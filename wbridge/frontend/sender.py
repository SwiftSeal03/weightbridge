import torch

from wbridge.backend.direct_sender import CPUDirectSender, GPUDirectSender
from wbridge.utils.data import WeightData


class WeightSender:
    def __init__(
        self,
        transfer_mode: str,
        receiver_urls: list[str],
    ):
        self.transfer_mode = transfer_mode
        self.receiver_urls = receiver_urls
        if transfer_mode == "gpu_direct":
            self.sender = GPUDirectSender(receiver_urls)
        elif transfer_mode == "cpu_direct":
            self.sender = CPUDirectSender(receiver_urls)
        else:
            raise ValueError(f"Invalid transfer mode: {transfer_mode}")

    def connect(self, sender_metadata: WeightData) -> None:
        self.sender.connect(sender_metadata)

    def send(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.sender.send(state_dict)