import torch

def len_after_conv(n_features, conv, dim):
    padding, kernel_size, stride = (
        conv.padding[dim],
        conv.kernel_size[dim],
        conv.stride[dim],
    )
    n_features = (n_features + padding * 2 - kernel_size) // stride + 1

    return n_features


def form_and_apply_mask(x, lengths):
    mask = torch.arange(x.size(-1), device=x.device)[None, None, None, :].expand_as(x)
    mask = mask < lengths[:, None, None, None]

    x = x.masked_fill(~mask, 0)

    return x
