import torch
import typing as t
from dataclasses import dataclass
from jaxtyping import Float, Bool
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


@dataclass
class State:
    determ: Float[torch.Tensor, 'seq batch num_slots determ']
    stoch_logits: Float[torch.Tensor, 'seq batch num_slots latent_classes latent_dim']
    stoch_: t.Optional[Bool[torch.Tensor, 'seq batch num_slots stoch_dim']] = None

    @property
    def combined(self):
        return torch.concat([self.determ, self.stoch], dim=-1).flatten(2, 3)

    @property
    def combined_slots(self):
        return torch.concat([self.determ, self.stoch], dim=-1)

    @property
    def stoch(self):
        if self.stoch_ is None:
            self.stoch_ = Dist(self.stoch_logits).rsample().reshape(self.stoch_logits.shape[:3] + (-1,))
        return self.stoch_

    @property
    def stoch_dist(self):
        return Dist(self.stoch_logits)

    @classmethod
    def stack(cls, states: list['State'], dim = 0):
        if states[0].stoch_ is not None:
            stochs = torch.cat([state.stoch for state in states], dim=dim)
        else:
            stochs = None
        return State(torch.cat([state.determ for state in states], dim=dim),
                     torch.cat([state.stoch_logits for state in states], dim=dim),
                     stochs)

class Normalizer(nn.Module):
    def __init__(self, momentum=0.99, scale=1.0, eps=1e-8):
        super().__init__()
        self.momentum = momentum
        self.scale = scale
        self.eps= eps
        self.register_buffer('mag', torch.ones(1, dtype=torch.float32))
        self.mag.requires_grad = False

    def forward(self, x):
        self.update(x)
        return (x / (self.mag + self.eps))*self.scale

    def update(self, x):
        self.mag = self.momentum * self.mag  + (1 - self.momentum) * (x.abs().mean()).detach()


