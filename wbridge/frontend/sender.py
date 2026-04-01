import torch

from wbridge.backend.direct_sender import CPUDirectSender, GPUDirectSender
from wbridge.utils.data import ShardSpec


class WeightSender:
    def __init__(
        self,
        transfer_mode: str,
        receiver_urls: list[str],
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ):
        self.transfer_mode = transfer_mode
        self.receiver_urls = receiver_urls
        sender_cls = GPUDirectSender if transfer_mode == "gpu_direct" else CPUDirectSender
        self.sender = sender_cls(receiver_urls, rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port)

    def connect(self, sender_metadata: ShardSpec) -> None:
        self.sender.connect(sender_metadata)

    def send(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.sender.send(state_dict)
