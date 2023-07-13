import typing as t
from collections import deque
from dataclasses import dataclass, field
from unpackable import unpack

import torch
import numpy as np
from jaxtyping import Bool, Float, Int

Observation = Int[torch.Tensor, 'n n 3']
State = Float[torch.Tensor, 'n']
Action = Int[torch.Tensor, 'n']

Observations = Int[torch.Tensor, 'batch n n 3']
States = Float[torch.Tensor, 'batch n']
Actions = Int[torch.Tensor, 'batch n']
Rewards = Float[torch.Tensor, 'batch']
TerminationFlags = Bool[torch.Tensor, 'batch']
IsFirstFlags = TerminationFlags

@dataclass
class EnvStep:
    obs: Observation
    action: Action
    reward: float
    is_finished: bool
    is_first: bool
    additional_data: dict[str, Float[torch.Tensor, '...']] = field(default_factory=dict)

@dataclass
class Rollout:
    obs: Observations
    actions: Actions
    rewards: Rewards
    is_finished: TerminationFlags
    is_first: IsFirstFlags
    additional_data: dict[str, Float[torch.Tensor, 'batch ...']] = field(default_factory=dict)

    def __len__(self):
        return len(self.obs)

    def to(self, device: str, non_blocking: bool = False):
        self.obs = self.obs.to(device, non_blocking=True)
        self.actions = self.actions.to(device, non_blocking=True)
        self.rewards = self.rewards.to(device, non_blocking=True)
        self.is_finished = self.is_finished.to(device, non_blocking=True)
        self.is_first = self.is_first.to(device, non_blocking=True)
        for k, v in self.additional_data.items():
            self.additional_data[k] = v.to(device, non_blocking = True)
        if not non_blocking:
            torch.cuda.current_stream().synchronize()
        return self

@dataclass
class RolloutChunks(Rollout):
    pass

class ReplayBuffer:

    def __init__(self, max_len=2e6,
                       prioritize_ends: bool = False,
                       min_ep_len: int = 1,
                       preprocess_func: t.Callable[[Rollout], Rollout] = lambda x: x,
                       device: str = 'cpu'):
        self.rollouts: deque[Rollout] = deque()
        self.rollouts_len: deque[int] = deque()
        self.curr_rollout = None
        self.min_ep_len = min_ep_len
        self.prioritize_ends = prioritize_ends
        self.max_len = max_len
        self.total_num = 0
        self.device = device
        self.preprocess_func = preprocess_func

    def __len__(self):
        return self.total_num

    def add_rollout(self, rollout: Rollout):
        if len(rollout.obs) <= self.min_ep_len:
            return
        self.rollouts.append(self.preprocess_func(rollout).to(device='cpu'))
        self.total_num += len(self.rollouts[-1].rewards)
        self.rollouts_len.append(len(self.rollouts[-1].rewards))

        while self.total_num >= self.max_len:
            self.total_num -= self.rollouts_len[0]
            self.rollouts_len.popleft()
            self.rollouts.popleft()

    # Add sample expects that each subsequent sample
    # will be continuation of last rollout util termination flag true
    # is encountered
    def add_sample(self, env_step: EnvStep):
        s, a, r, n, f, additional = unpack(env_step)
        if self.curr_rollout is None:
            self.curr_rollout = Rollout([s], [a], [r], [n], [f], {k: [v] for k,v in additional.items()})
        else:
            self.curr_rollout.obs.append(s)
            self.curr_rollout.actions.append(a)
            self.curr_rollout.rewards.append(r)
            self.curr_rollout.is_finished.append(n)
            self.curr_rollout.is_first.append(f)
            for k,v in additional.items():
                self.curr_rollout.additional_data[k].append(v)

            if f:
                self.add_rollout(
                    Rollout(
                        torch.stack(self.curr_rollout.obs),
                        torch.stack(self.curr_rollout.actions).reshape(-1, 1),
                        torch.Tensor(self.curr_rollout.rewards),
                        torch.Tensor(self.curr_rollout.is_finished),
                        torch.Tensor(self.curr_rollout.is_first),
                        {k: torch.stack(v) for k,v in self.curr_rollout.additional_data.items()})
                        )
                self.curr_rollout = None

    def can_sample(self, num: int):
        return self.total_num >= num

    def sample(
        self,
        batch_size: int,
        cluster_size: int = 1
    ) -> RolloutChunks:
        # NOTE: constant creation of numpy arrays from self.rollout_len seems terrible for me
        s, a, r, t, is_first, additional = [], [], [], [], [], {}
        r_indeces = np.random.choice(len(self.rollouts), batch_size, p=np.array(self.rollouts_len) / self.total_num)
        s_indeces = []
        for r_idx in r_indeces:
            rollout, r_len = self.rollouts[r_idx], self.rollouts_len[r_idx]

            assert r_len > cluster_size - 1, "Rollout it too small"
            max_idx = r_len - cluster_size + 1
            if self.prioritize_ends:
                s_idx = np.random.choice(max_idx - cluster_size + 1, 1).item() + cluster_size - 1
            else:
                s_idx = np.random.choice(max_idx, 1).item()
            s_indeces.append(s_idx)

            is_first.append(torch.zeros(cluster_size))
            if s_idx == 0:
                is_first[-1][0] = 1

            s.append(rollout.obs[s_idx:s_idx + cluster_size])
            a.append(rollout.actions[s_idx:s_idx + cluster_size])
            r.append(rollout.rewards[s_idx:s_idx + cluster_size])
            t.append(rollout.is_finished[s_idx:s_idx + cluster_size])
            for k,v in rollout.additional_data.items():
                if k not in additional:
                    additional[k] = []
                additional[k].append(v[s_idx:s_idx + cluster_size])

        return RolloutChunks(
                obs=torch.cat(s),
                actions=torch.cat(a),
                rewards=torch.cat(r).float(),
                is_finished=torch.cat(t),
                is_first=torch.cat(is_first),
                additional_data={k: torch.cat(v) for k,v in additional.items()}
                ).to(self.device, non_blocking=False)


# TODO:
# [ ] (Optional) Utilize torch's dataloader for async sampling
