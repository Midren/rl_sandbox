import torch
from torch import nn

from rl_sandbox.utils.dists import DistLayer


class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def Dist(val):
    return DistLayer('onehot')(val)


class Normalizer(nn.Module):

    def __init__(self, momentum=0.99, scale=1.0, eps=1e-8):
        super().__init__()
        self.momentum = momentum
        self.scale = scale
        self.eps = eps
        self.register_buffer('mag', torch.ones(1, dtype=torch.float32))
        self.mag.requires_grad = False

    def forward(self, x):
        self.update(x)
        return (x / (self.mag + self.eps)) * self.scale

    def update(self, x):
        self.mag = self.momentum * self.mag + (1 -
                                               self.momentum) * (x.abs().mean()).detach()
