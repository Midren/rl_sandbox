import torch
from torch import nn
import torch.distributions as td
import numpy as np

from rl_sandbox.utils.dists import DistLayer

def get_position_encoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def Dist(val):
    return td.Independent(DistLayer('onehot')(val), 1)


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


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, norm=False, update_bias=-1, **kwargs):
        super().__init__()
        self._size = hidden_size
        self._act = torch.tanh
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(input_size + hidden_size,
                                3 * hidden_size,
                                bias=norm is not None,
                                **kwargs)
        if norm:
            self._norm = nn.LayerNorm(3 * hidden_size)

    @property
    def state_size(self):
        return self._size

    def forward(self, x, h):
        state = h
        parts = self._layer(torch.concat([x, state], -1))
        if self._norm:
            dtype = parts.dtype
            parts = self._norm(parts.float())
            parts = parts.to(dtype=dtype)
        reset, cand, update = parts.chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, output


