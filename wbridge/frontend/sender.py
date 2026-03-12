import torch
import requests

from wbridge.backend.direct_sender import CPUDirectSender, GPUDirectSender


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
    
    def get_metadata(self) -> dict[str, dict[str, ...]]:
        metadatas = requests.get(f"{self.receiver_urls[0]}/wbridge/metadata").json()
        return metadatas
    
    def send(
        self,
        params: dict[str, torch.Tensor],
    ):
        metadatas = self.get_metadata()
        print(metadatas)
        # self.sender.send(params)