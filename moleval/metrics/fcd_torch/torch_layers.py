import torch
from torch import nn


class Reverse(nn.Module):
    def forward(self, x):
        return torch.flip(x, [1])


class IndexTuple(nn.Module):
    def __init__(self, pos):
        super().__init__()
        self.pos = pos

    def forward(self, x):
        return x[self.pos]


class IndexTensor(nn.Module):
    def __init__(self, pos, dim):
        super().__init__()
        self.pos = pos
        self.dim = dim

    def forward(self, x):
        return torch.select(x, self.dim, self.pos)


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


# https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch
class SamePadding1d(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        if x.shape[2] % self.stride == 0:
            pad = max(self.kernel_size - self.stride, 0)
        else:
            pad = max(self.kernel_size - (x.shape[2] % self.stride), 0)

        if pad % self.stride == 0:
            pad_val = pad // self.stride
            padding = (pad_val, pad_val)
        else:
            pad_val_start = pad // self.stride
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end)
        return torch.nn.functional.pad(x, padding, "constant", 0)
