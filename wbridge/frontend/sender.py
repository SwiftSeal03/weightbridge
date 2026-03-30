import torch

from wbridge.backend.direct_sender import CPUDirectSender, GPUDirectSender
from wbridge.utils.data import WeightData


class WeightSender:
    def __init__(
        self,
        transfer_mode: str,
        receiver_urls: list[str],
        rank: int,
        world_size: int,
    ):
        self.transfer_mode = transfer_mode
        self.receiver_urls = receiver_urls
        if transfer_mode == "gpu_direct":
            self.sender = GPUDirectSender(receiver_urls, rank=rank, world_size=world_size)
        elif transfer_mode == "cpu_direct":
            self.sender = CPUDirectSender(receiver_urls, rank=rank, world_size=world_size)
        else:
            raise ValueError(f"Invalid transfer mode: {transfer_mode}")

    def connect(self, sender_metadata: WeightData, sender_init_method: str) -> None:
        self.sender.connect(sender_metadata, sender_init_method=sender_init_method)

    def send(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.sender.send(state_dict)
