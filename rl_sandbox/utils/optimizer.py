import typing as t
from collections.abc import Iterable
import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import GradScaler

from torch.optim.lr_scheduler import LinearLR, LambdaLR

class WarmupScheduler(LinearLR):
    def __init__(self, optimizer, warmup_steps):
        super().__init__(optimizer, start_factor=1/warmup_steps, total_iters=int(warmup_steps))

# class WarmupScheduler(LambdaLR):
#     def __init__(self, optimizer, warmup_steps):
#         super().__init__(optimizer, lambda epoch: min(1, np.interp(epoch, [1, warmup_steps], [0, 1])) )

class DecayScheduler(LambdaLR):
    def __init__(self, optimizer, decay_steps, decay_rate):
        super().__init__(optimizer, lambda epoch: decay_rate**(epoch/decay_steps))

class Optimizer:
    def __init__(self, model,
                 lr=1e-4,
                 eps=1e-8,
                 weight_decay=0.01,
                 lr_scheduler: t.Optional[t.Type[LRScheduler] | t.Iterable[t.Type[LRScheduler]]] = None,
                 scaler: bool = False,
                 log_grad: bool = False,
                 clip: t.Optional[float] = None):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
        self.lr_scheduler = lr_scheduler
        if lr_scheduler is not None and not isinstance(lr_scheduler, Iterable):
            self.lr_scheduler = lr_scheduler(optimizer=self.optimizer)
        elif isinstance(lr_scheduler, Iterable):
            self.lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([lr_sched(optimizer=self.optimizer) for lr_sched in lr_scheduler])
        self.log_grad = log_grad
        self.scaler = GradScaler() if scaler else None
        self.clip = clip

    def step(self, loss):
        metrics = {}
        self.optimizer.zero_grad(set_to_none=True)

        if self.scaler:
            loss = self.scaler.scale(loss)
        loss.backward()

        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        if self.log_grad:
            for tag, value in self.model.named_parameters():
                metrics[f"grad/{tag.replace('.', '/')}"] = value.detach()

        if self.clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()
            metrics[f'lr/{self.model.__class__.__name__}'] = torch.Tensor(self.lr_scheduler.get_last_lr())

        return metrics
