import math
from numbers import Number
import typing as t

import numpy as np
import torch
import torch.distributions as td
from torch import nn
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch.distributions.utils import _standard_normal

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

class TruncatedNormal(td.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, sample_shape=torch.Size(), clip=None):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Sigmoid2(nn.Module):
    def forward(self, x):
        return 2*torch.sigmoid(x/2)

class NormalWithOffset(nn.Module):
    def __init__(self, min_std: float, std_trans: str = 'sigmoid2', transform: t.Optional[str] = None):
        super().__init__()
        self.min_std = min_std
        match std_trans:
            case 'identity':
                self.std_trans = nn.Identity()
            case 'softplus':
                self.std_trans = nn.Softplus()
            case 'sigmoid':
                self.std_trans = nn.Sigmoid()
            case 'sigmoid2':
                self.std_trans = Sigmoid2()
            case _:
                raise RuntimeError("Unknown std transformation")

        match transform:
            case 'tanh':
                self.trans = [td.TanhTransform(cache_size=1)]
            case None:
                self.trans = None
            case _:
                raise RuntimeError("Unknown distribution transformation")

    def forward(self, x):
        mean, std = x.chunk(2, dim=-1)
        dist = td.Normal(mean, self.std_trans(std) + self.min_std)
        if self.trans is None:
            return dist
        else:
            return td.TransformedDistribution(dist, self.trans)

class DistLayer(nn.Module):
    def __init__(self, type: str):
        super().__init__()
        self._dist = type
        match type:
            case 'mse':
                self.dist = lambda x: td.Normal(x.float(), 1.0)
            case 'normal':
                self.dist = NormalWithOffset(min_std=0.1)
            case 'onehot':
                # Forcing float32 on AMP
                self.dist = lambda x: td.OneHotCategoricalStraightThrough(logits=x.float())
            case 'normal_tanh':
                def get_tanh_normal(x, min_std=0.1):
                    mean, std = x.chunk(2, dim=-1)
                    init_std = np.log(np.exp(5) - 1)
                    raise NotImplementedError()
                    # return TanhNormal(torch.clamp(mean, -9.0, 9.0).float(), (F.softplus(std + init_std) + min_std).float(), upscale=5)
                self.dist = get_tanh_normal
            case 'normal_trunc':
                def get_trunc_normal(x, min_std=0.1):
                    mean, std = x.chunk(2, dim=-1)
                    return TruncatedNormal(loc=torch.tanh(mean).float(), scale=(2*torch.sigmoid(std/2) + min_std).float())
                self.dist = get_trunc_normal
            case 'binary':
                self.dist = lambda x: td.Bernoulli(logits=x.float())
            case _:
                raise RuntimeError("Invalid dist layer")

    def forward(self, x):
        match self._dist:
            case 'onehot':
                return self.dist(x)
            case _:
                # FIXME: verify dimensionality of independent
                return td.Independent(self.dist(x), 1)

