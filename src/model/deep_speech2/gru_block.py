import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUBlock(nn.Module):
    def __init__(self, gru_input_size, gru_hidden_size, batch_norm=True) -> None:
        super().__init__()

        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=gru_input_size)
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, h_last, lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        packed_x = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        layer_hiddens_packed, h_last = self.gru(packed_x, h_last)
        layer_hiddens = pad_packed_sequence(layer_hiddens_packed, batch_first=True)[0]

        b_size, n_time, _ = layer_hiddens.size()
        layer_hiddens = layer_hiddens.reshape(b_size, n_time, 2, -1)
        layer_hiddens = torch.sum(
            layer_hiddens, dim=2
        )  # суммируем хиддены с разных сторон
        return layer_hiddens, h_last
