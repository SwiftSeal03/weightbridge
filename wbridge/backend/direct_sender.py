import torch

from wbridge.utils.data import WeightData

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
    def send(
        self,
        params: WeightData,
    ):
        print("Sending weights to GPU: ", params)
    
class CPUDirectSender(DirectSender):
    pass