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
    ):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.receiver_urls = receiver_urls
        
        self.connected = False
        self.group: dist.ProcessGroup | None = None
        self.overlaps: dict[int, WeightData] = {}
        self.metadata: WeightData | None = None
        self.backend = None
        self.device = None

    def _dedup_sender_metadata(self, sender_metadata: WeightData) -> WeightData:
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
        dist.all_gather_object(all_meta_dicts, meta_dict)

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
        sender_metadata = self._dedup_sender_metadata(sender_metadata)
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
                        "master_address": master_address,
                        "master_port": master_port,
                        "base_rank": base_rank,
                        "world_size": total_world_size,
                        "group_name": group_name,
                        "sender_world_size": self.world_size,
                        "backend": self.backend,
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

        dist.broadcast_object_list(connect_info, src=0)
        master_address, master_port, total_world_size, group_name, receiver_metas = (
            connect_info
        )
        
        logger.info("Sender %d initializing process group with master %s:%d", self.rank, master_address, master_port)
        self.group = init_custom_process_group(
            backend=self.backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=total_world_size,
            rank=self.rank,
            group_name=group_name,
        )
        logger.info("Sender %d initialized process group", self.rank)
        # Compute overlap with each receiver and send the sizes of the overlap metadata to the receiver
        handles: list = []
        for r_rank, r_meta in receiver_metas:
            overlap = WeightData.compute_overlap(sender_metadata, r_meta)
            if not overlap:
                continue
            self.overlaps[r_rank] = overlap
            byte_size = torch.tensor([len(bytes(overlap))], dtype=torch.long, device=self.device)
            handles.append(dist.isend(byte_size, dst=r_rank, group=self.group))
        for h in handles:
            h.wait()
            
        # Send the overlap metadata bytes to the receivers
        handles: list = []
        for r_rank, overlap in self.overlaps.items():
            overlap_bytes = torch.frombuffer(bytes(overlap), dtype=torch.uint8).to(self.device)
            handles.append(dist.isend(overlap_bytes, dst=r_rank, group=self.group))
        for h in handles:
            h.wait()

        logger.info(
            "Sender %d connected (group=%s, backend=%s, world_size=%d, receiver_workers=%d)",
            self.rank,
            group_name,
            self.backend,
            total_world_size,
            len(receiver_metas),
        )
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
        handles = [
            dist.isend(chunk, dst=receiver_rank, group=self.group)
            for receiver_rank, chunk in chunks.items()
        ]
        for h in handles:
            h.wait()


class CPUDirectSender(DirectSender):
    def __init__(
        self,
        receiver_urls: list[str],
    ):
        super().__init__(receiver_urls)
        self.device = "cpu"
        self.backend = "gloo"

    def send(self, state_dict: dict[str, torch.Tensor]) -> None:
        pass # TODO: Implement CPUDirectSender.send
