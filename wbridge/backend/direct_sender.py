import torch


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
    pass
    
class CPUDirectSender(DirectSender):
    pass