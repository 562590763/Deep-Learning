import torch
import torch.nn as nn


class Identical(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.zeros_like(x, requires_grad=True)
        return x


class Shift(nn.Module):
    def __init__(self, n_div=12):
        super().__init__()
        self.n_div = n_div

    def forward(self, x):
        shortcut = x[:]
        B, C, H, W = x.shape
        g = C // self.n_div

        x[:, g * 0:g * 1, :, :-1] = x[:, g * 0:g * 1, :, 1:]  # shift left
        x[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # shift right
        x[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # shift up
        x[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # shift down
        x[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # no shift
        return x - shortcut
