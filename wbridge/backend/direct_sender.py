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
        self.init_method = f"tcp://{master_addr}:{master_port}"

        self.connected = False
        self.group: dist.ProcessGroup | None = None
        self.overlaps: dict[int, WeightData] = {}
        self.metadata: WeightData | None = None
        self.backend = None
        self.device = None
            

    def connect(self, sender_metadata: WeightData) -> None:
        """Join receivers over NCCL after a short-lived Gloo group for sender coordination.

        The Gloo process group uses ``tcp://{master_addr}:{master_port}`` from
        :meth:`__init__` so all sender ranks rendezvous before rank 0 drives
        HTTP metadata/connect and broadcasts rendezvous info for the main group.
        """
        self.metadata = sender_metadata
        if self.group is not None:
            dist.destroy_process_group(self.group)

        # Query receiver metadata and build per-worker list
        rollout_num_workers = []
        resps = [requests.get(f"{url}/wbridge/metadata") for url in self.receiver_urls]
        assert all(resp.status_code == 200 for resp in resps), "Failed to get receiver metadata"
        rollout_num_workers = [resp.json()["world_size"] for resp in resps]
        total_world_size = self.world_size + sum(rollout_num_workers)
        
        pg_init_args = {
            "backend": self.backend,
            "init_method": self.init_method,
            "world_size": total_world_size,
            "rank": self.rank,
            "group_name": "wbridge",
        }
            
        if self.rank == 0:
            base_rank = self.world_size
            for url, num_workers in zip(self.receiver_urls, rollout_num_workers):
                connect_args = {
                    **pg_init_args,
                    "rank": base_rank,
                    "sender_world_size": self.world_size,
                }
                resp = requests.post(f"{url}/wbridge/connect", json=connect_args)
                resp.raise_for_status()
                base_rank += num_workers
        
        self.group = init_custom_process_group(**pg_init_args)
        
        all_metas = [None] * total_world_size
        dist.all_gather_object(all_metas, self.metadata, group=self.group)
        
        self.overlaps = {
            rank: overlap for rank, meta in enumerate(all_metas) 
            if rank >= self.world_size and (overlap := WeightData.compute_overlap(self.metadata, meta))
        }
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
