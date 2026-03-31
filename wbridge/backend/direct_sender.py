import json
import logging
import socket

import requests
import torch
import torch.distributed as dist

from wbridge.utils.data import WeightData
from wbridge.utils.distributed import init_custom_process_group, get_local_ip, get_full_group_port

logger = logging.getLogger(__name__)


class DirectSender:
    def __init__(
        self,
        receiver_urls: list[str],
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.receiver_urls = receiver_urls
        self._sender_coord_init_method = f"tcp://{master_addr}:{master_port}"

        self.connected = False
        self.group: dist.ProcessGroup | None = None
        self.overlaps: dict[int, WeightData] = {}
        self.metadata: WeightData | None = None
        self.backend = None
        self.device = None

    def _dedup_sender_metadata(
        self, sender_metadata: WeightData, sender_coord_group: dist.ProcessGroup | None = None,
    ) -> WeightData:
        """All-gather metadata across senders and deduplicate identical shards.

        For each named tensor present on multiple senders, the intersection of
        their shard specs must be either empty (the common case) or exactly
        equal.  When equal, only the lowest-rank sender keeps the entry;
        higher-rank senders drop it.  Any partial overlap raises an error.
        """
        if self.world_size == 1:
            return sender_metadata

        meta_dict = dict(sender_metadata.meta_dict)
        all_meta_dicts: list[dict | None] = [None] * self.world_size
        dist.all_gather_object(all_meta_dicts, meta_dict, group=sender_coord_group)

        names_to_remove: set[str] = set()
        for peer_rank, peer_dict in enumerate(all_meta_dicts):
            if peer_rank == self.rank:
                continue
            peer_wd = WeightData(peer_dict)
            overlap = WeightData.compute_overlap(sender_metadata, peer_wd)

            for name in overlap.meta_dict:
                if meta_dict[name]["shard"] == peer_dict[name]["shard"]:
                    if self.rank > peer_rank:
                        names_to_remove.add(name)
                else:
                    raise ValueError(
                        f"Partial shard overlap for '{name}' between sender "
                        f"rank {self.rank} and rank {peer_rank}. Senders must "
                        f"have identical or non-overlapping shards."
                    )

        if not names_to_remove:
            return sender_metadata

        new_meta_dict = {
            k: v for k, v in sender_metadata.meta_dict.items()
            if k not in names_to_remove
        }
        return WeightData(new_meta_dict)


    def connect(self, sender_metadata: WeightData) -> None:
        """Join receivers over NCCL after a short-lived Gloo group for sender coordination.

        The Gloo process group uses ``tcp://{master_addr}:{master_port}`` from
        :meth:`__init__` so all sender ranks rendezvous before rank 0 drives
        HTTP metadata/connect and broadcasts rendezvous info for the main group.
        """
        sender_coord_group = None
        if self.world_size > 1:
            sender_coord_group = init_custom_process_group(
                backend="gloo",
                init_method=self._sender_coord_init_method,
                world_size=self.world_size,
                rank=self.rank,
                group_name="wbridge_sender_coord",
            )

        sender_metadata = self._dedup_sender_metadata(sender_metadata, sender_coord_group)
        self.metadata = sender_metadata

        group_name = "wbridge"

        if self.rank == 0:            
            # Query receiver metadata and build per-worker list
            receiver_metas: list[tuple[int, dict]] = []
            receiver_worker_counts: list[int] = []
            base_rank = self.world_size
            for url in self.receiver_urls:
                resp = requests.get(f"{url}/wbridge/metadata")
                resp.raise_for_status()
                workers = sorted(resp.json(), key=lambda w: w["rank"])
                receiver_worker_counts.append(len(workers))
                receiver_metas.extend([(
                    base_rank + worker["rank"], WeightData(worker["metadata"])
                ) for worker in workers])
                base_rank += len(workers)

            total_world_size = self.world_size + sum(receiver_worker_counts)
            
            # Tell each receiver to join, assigning ranks starting after all senders
            base_rank = self.world_size
            master_address, master_port = get_local_ip(), get_full_group_port()
            for url, count in zip(self.receiver_urls, receiver_worker_counts):
                resp = requests.post(
                    f"{url}/wbridge/connect",
                    json={
                        "backend": self.backend,
                        "init_method": f"tcp://{master_address}:{master_port}",
                        "rank": base_rank,
                        "world_size": total_world_size,
                        "group_name": group_name,
                        "sender_world_size": self.world_size,
                    },
                )
                resp.raise_for_status()
                base_rank += count

            connect_info = [
                master_address,
                master_port,
                total_world_size,
                group_name,
                receiver_metas,
            ]
        else:
            connect_info = [None, None, None, None, None]

        if sender_coord_group is not None:
            dist.broadcast_object_list(connect_info, src=0, group=sender_coord_group)
            dist.destroy_process_group(sender_coord_group)
        master_address, master_port, total_world_size, group_name, receiver_metas = (
            connect_info
        )
        
        self.group = init_custom_process_group(
            backend=self.backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=total_world_size,
            rank=self.rank,
            group_name=group_name,
        )
        
        # Compute overlap with each receiver and send the sizes of the overlap metadata.
        # Always send a size (0 when no overlap) so receivers don't block on a recv that never comes.
        size_ops: list[dist.P2POp] = []
        for r_rank, r_meta in receiver_metas:
            overlap = WeightData.compute_overlap(sender_metadata, r_meta)
            if overlap:
                self.overlaps[r_rank] = overlap
                size_bytes = len(bytes(overlap))
                byte_size = torch.tensor([size_bytes], dtype=torch.long, device=self.device)
            else:
                size_bytes = 0
                byte_size = torch.tensor([size_bytes], dtype=torch.long, device=self.device)
            size_ops.append(dist.P2POp(dist.isend, byte_size, r_rank, self.group))
        for h in dist.batch_isend_irecv(size_ops):
            h.wait()
            
        dist.barrier(group=self.group)

        # Send the overlap metadata bytes to the receivers
        meta_ops: list[dist.P2POp] = []
        for r_rank, overlap in self.overlaps.items():
            overlap_bytes = bytearray(bytes(overlap))
            overlap_bytes = torch.frombuffer(overlap_bytes, dtype=torch.uint8).to(self.device)
            meta_ops.append(dist.P2POp(dist.isend, overlap_bytes, r_rank, self.group))
        for h in dist.batch_isend_irecv(meta_ops):
            h.wait()

        self.connected = True

    def send(self, state_dict: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError


class GPUDirectSender(DirectSender):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.device = "cuda"
        self.backend = "nccl"

    def send(self, state_dict: dict[str, torch.Tensor]) -> None:
        if not self.connected or self.metadata is None:
            raise RuntimeError("GPUDirectSender.send requires connect() first")
        if self.rank == 0:
            for url in self.receiver_urls:
                resp = requests.post(f"{url}/wbridge/receive")
                resp.raise_for_status()

        chunks = self.metadata(state_dict)[self.overlaps]
        ops = [
            dist.P2POp(dist.isend, chunk, receiver_rank, self.group)
            for receiver_rank, chunk in chunks.items()
        ]
        if ops:
            for h in dist.batch_isend_irecv(ops):
                h.wait()


class CPUDirectSender(DirectSender):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.device = "cpu"
        self.backend = "gloo"

    def send(self, state_dict: dict[str, torch.Tensor]) -> None:
        pass # TODO: Implement CPUDirectSender.send
