from math import floor

import torch
from torch import Tensor, nn

from src.model.deep_speech2.conv_module import ConvModule
from src.model.deep_speech2.gru_block import GRUBlock
from src.model.deep_speech2.utils import len_after_conv


class DeepSpeech2(nn.Module):
    def __init__(
        self, n_feats, n_tokens, gru_layers_num=7, gru_hidden_size=512
    ) -> None:
        super().__init__()

        assert gru_layers_num > 0

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(41, 11),
                stride=(2, 2),
                padding=(20, 5),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=96),
            nn.Hardtanh(0, 20),
        )

        self.conv_module = ConvModule(layers=self.conv_layers)

        gru_input_size = n_feats
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                gru_input_size = len_after_conv(
                    n_features=gru_input_size, conv=layer, dim=0
                )
        gru_input_size *= 96

        self.first_gru = GRUBlock(
            gru_input_size=gru_input_size,
            gru_hidden_size=gru_hidden_size,
            batch_norm=False,
        )
        self.gru_layers = nn.ModuleList(
            [
                GRUBlock(
                    gru_input_size=gru_hidden_size, gru_hidden_size=gru_hidden_size
                )
                for _ in range(gru_layers_num - 1)
            ]
        )

        self.proj = nn.Linear(in_features=gru_hidden_size, out_features=n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        spectrogram_length = spectrogram_length.to(spectrogram.device)
        x, new_lengths = self.conv_module(spectrogram.unsqueeze(1), spectrogram_length)
        b_size, n_channels, n_freqs, n_time = x.size()

        x = x.reshape(b_size, n_channels * n_freqs, n_time)
        x = x.permute(0, 2, 1)

        x, h_last = self.first_gru(x, None, new_lengths)
        for gru in self.gru_layers:
            x, h_last = gru(x, h_last, new_lengths)

        logits = self.proj(x)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": new_lengths.detach().cpu()}
