import torch
from torch import Tensor, nn
from torchaudio import transforms


class TimeMasking(nn.Module):
    def __init__(self, p, max_mask_len):
        super().__init__()
        self.p = p
        self._aug = transforms.TimeMasking(
            time_mask_param=max_mask_len,
        )

    def __call__(self, data: Tensor):
        if torch.rand(1) < self.p:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        return data
