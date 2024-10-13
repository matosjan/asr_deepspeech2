import torch
from torch import Tensor, nn

from src.model.deep_speech2.utils import after_conv, apply_mask


class ConvModule(nn.Module):
    def __init__(self, layers: nn.Sequential) -> None:
        super().__init__()

        self.layers = layers

    def forward(self, x, lengths):
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                lengths = after_conv(n_features=lengths, conv=layer, dim=1)
            x = apply_mask(x, lengths)
        return x, lengths
