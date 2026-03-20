import socket
import logging
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

    def send(
        self,
        params: dict[str, torch.Tensor],
    ):
        pass


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

    def connect(self) -> None:
        group_name = "wbridge"

        if self.rank == 0:
            master_address = _get_local_ip()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            # Count receiver workers via metadata endpoint
            receiver_worker_counts: list[int] = []
            for url in self.receiver_urls:
                resp = requests.get(f"{url}/wbridge/metadata")
                resp.raise_for_status()
                receiver_worker_counts.append(len(resp.json()))

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
                    },
                )
                resp.raise_for_status()
                base_rank += count

            connect_info = [master_address, master_port, total_world_size, group_name]
        else:
            connect_info = [None, None, None, None]

        dist.broadcast_object_list(connect_info, src=0)
        master_address, master_port, total_world_size, group_name = connect_info

        self.group = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=total_world_size,
            rank=self.rank,
            group_name=group_name,
        )
        
        logger.info(f"Sender {self.rank} joined group {group_name} as rank {self.rank} (world_size={total_world_size})")

    def send(
        self,
        params: WeightData,
    ):
        if not self.connected:
            self.connect()
            self.connected = True
        self.sender.send(params)


class CPUDirectSender(DirectSender):
    pass
