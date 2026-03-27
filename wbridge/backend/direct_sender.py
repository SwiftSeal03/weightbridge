import json
import logging
import socket

import requests
import torch
import torch.distributed as dist

from wbridge.utils.data import WeightData
from wbridge.utils.distributed import init_custom_process_group

logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    """Return the IP address of the interface used for outbound traffic."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


class DirectSender:
    def __init__(
        self,
        receiver_urls: list[str],
    ):
        self.receiver_urls = receiver_urls

    def connect(self, sender_metadata: WeightData) -> None:
        raise NotImplementedError

    def send(self, state_dict: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError


class GPUDirectSender(DirectSender):
    def __init__(
        self,
        receiver_urls: list[str],
    ):
        super().__init__(receiver_urls)
        self.rank = dist.get_rank()
        self.connected = False
        self.world_size = dist.get_world_size()
        self.group: dist.ProcessGroup | None = None
        self.overlaps: dict[int, WeightData] = {}
        self._sender_metadata: WeightData | None = None

    backend = "nccl"

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
        self._sender_metadata = sender_metadata

        group_name = "wbridge"

        if self.rank == 0:
            master_address = _get_local_ip()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            # Query receiver metadata and build per-worker list
            all_receiver_workers: list[tuple[int, dict]] = []
            receiver_worker_counts: list[int] = []
            base_rank = self.world_size
            for url in self.receiver_urls:
                resp = requests.get(f"{url}/wbridge/metadata")
                resp.raise_for_status()
                workers = sorted(resp.json(), key=lambda w: w["rank"])
                receiver_worker_counts.append(len(workers))
                for worker in workers:
                    all_receiver_workers.append(
                        (base_rank + worker["rank"], worker["metadata"])
                    )
                base_rank += len(workers)

            total_world_size = self.world_size + sum(receiver_worker_counts)

            # Tell each receiver to join, assigning ranks starting after all senders
            base_rank = self.world_size
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
                all_receiver_workers,
            ]
        else:
            connect_info = [None, None, None, None, None]

        dist.broadcast_object_list(connect_info, src=0)
        master_address, master_port, total_world_size, group_name, all_receiver_workers = (
            connect_info
        )

        self.group = init_custom_process_group(
            backend=self.backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=total_world_size,
            rank=self.rank,
            group_name=group_name,
        )

        # Compute overlap with each receiver and send it via the process group.
        # Receivers iterate over sender ranks in the same order, so the
        # point-to-point send/recv pairs are matched without deadlock.
        device = "cuda" if self.backend == "nccl" else "cpu"
        for receiver_rank, receiver_meta_dict in all_receiver_workers:
            receiver_wd = WeightData(receiver_meta_dict)
            overlap = WeightData.compute_overlap(sender_metadata, receiver_wd)
            self.overlaps[receiver_rank] = overlap

            overlap_bytes = json.dumps(overlap.meta_dict, default=str).encode("utf-8")
            size_t = torch.tensor(
                [len(overlap_bytes)], dtype=torch.long, device=device
            )
            data_t = torch.frombuffer(
                bytearray(overlap_bytes), dtype=torch.uint8
            ).to(device)
            dist.send(size_t, dst=receiver_rank, group=self.group)
            dist.send(data_t, dst=receiver_rank, group=self.group)

        logger.info(
            "Sender %d connected (group=%s, backend=%s, world_size=%d, receiver_workers=%d)",
            self.rank,
            group_name,
            self.backend,
            total_world_size,
            len(all_receiver_workers),
        )
        self.connected = True

    def send(self, state_dict: dict[str, torch.Tensor]) -> None:
        if not self.connected or self._sender_metadata is None:
            raise RuntimeError("GPUDirectSender.send requires connect() first")
        if self.rank == 0:
            for url in self.receiver_urls:
                resp = requests.post(f"{url}/wbridge/receive")
                resp.raise_for_status()

        device = "cuda" if self.backend == "nccl" else "cpu"
        handles: list = []
        for receiver_rank, overlap in self.overlaps.items():
            if not overlap.meta_dict:
                continue
            packed = WeightData.pack_for(
                self._sender_metadata, state_dict, overlap
            )
            if packed.device.type != device:
                packed = packed.to(device)
            nbytes = packed.numel()
            logger.info(
                "Sender %d -> Receiver %d: %d bytes",
                self.rank, receiver_rank, nbytes,
            )
            handles.append(
                dist.isend(packed, dst=receiver_rank, group=self.group)
            )
        for h in handles:
            h.wait()


class CPUDirectSender(DirectSender):
    backend = "gloo"
