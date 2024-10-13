import torch_audiomentations
from torch import Tensor, nn

class LowPassFilter(nn.Module):
    def __init__(self, p, min_cutoff_freq, max_cutoff_freq, sample_rate):
        super().__init__()
        self._aug = torch_audiomentations.LowPassFilter(
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            sample_rate=sample_rate,
            p=p,
        )

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)