import typing as t
from collections import deque
from dataclasses import dataclass

import numpy as np
from nptyping import Bool, Float, Int, NDArray, Shape

Observation = NDArray[Shape["*,*,3"], Int]
State = NDArray[Shape["*"], Float] | Observation
Action = NDArray[Shape["*"], Int]

Observations = NDArray[Shape["*,*,*,3"], Int]
States = NDArray[Shape["*,*"], Float] | Observations
Actions = NDArray[Shape["*,*"], Int]
Rewards = NDArray[Shape["*"], Float]
TerminationFlags = NDArray[Shape["*"], Bool]
IsFirstFlags = TerminationFlags


@dataclass
class Rollout:
    states: States
    actions: Actions
    rewards: Rewards
    next_states: States
    is_finished: TerminationFlags
    observations: t.Optional[Observations] = None

    def __len__(self):
        return len(self.states)

# TODO: make buffer concurrent-friendly
class ReplayBuffer:

    def __init__(self, max_len=2e6,
                       prioritize_ends: bool = False,
                       min_ep_len: int = 1,
                       device: str = 'cpu'):
        self.rollouts: deque[Rollout] = deque()
        self.rollouts_len: deque[int] = deque()
        self.curr_rollout = None
        self.min_ep_len = min_ep_len
        self.prioritize_ends = prioritize_ends
        self.max_len = max_len
        self.total_num = 0
        self.device = device

    def __len__(self):
        return self.total_num

    def add_rollout(self, rollout: Rollout):
        if len(rollout.next_states) <= self.min_ep_len:
            return
        # NOTE: only last next state is stored, all others are induced
        #       from state on next step
        rollout.next_states = np.expand_dims(rollout.next_states[-1], 0)
        self.rollouts.append(rollout)
        self.total_num += len(self.rollouts[-1].rewards)
        self.rollouts_len.append(len(self.rollouts[-1].rewards))

        while self.total_num >= self.max_len:
            self.total_num -= self.rollouts_len[0]
            self.rollouts_len.popleft()
            self.rollouts.popleft()

    # Add sample expects that each subsequent sample
    # will be continuation of last rollout util termination flag true
    # is encountered
    def add_sample(self, s: State, a: Action, r: float, n: State, f: bool):
        if self.curr_rollout is None:
            self.curr_rollout = Rollout([s], [a], [r], None, [f])
        else:
            self.curr_rollout.states.append(s)
            self.curr_rollout.actions.append(a)
            self.curr_rollout.rewards.append(r)
            self.curr_rollout.is_finished.append(f)

            if f:
                self.add_rollout(
                    Rollout(np.array(self.curr_rollout.states),
                            np.array(self.curr_rollout.actions).reshape(len(self.curr_rollout.actions), -1),
                            np.array(self.curr_rollout.rewards, dtype=np.float32),
                            np.array([n]), np.array(self.curr_rollout.is_finished)))
                self.curr_rollout = None

    def can_sample(self, num: int):
        return self.total_num >= num

    def sample(
        self,
        batch_size: int,
        cluster_size: int = 1
    ) -> tuple[States, Actions, Rewards, States, TerminationFlags, IsFirstFlags]:
        # NOTE: constant creation of numpy arrays from self.rollout_len seems terrible for me
        s, a, r, n, t, is_first = [], [], [], [], [], []
        do_add_curr = self.curr_rollout is not None and len(self.curr_rollout.states) > (cluster_size * (self.prioritize_ends + 1))
        tot = self.total_num + (len(self.curr_rollout.states) if do_add_curr else 0)
        r_indeces = np.random.choice(len(self.rollouts) + int(do_add_curr),
                                     batch_size,
                                     p=np.array(self.rollouts_len + deque([len(self.curr_rollout.states)] if do_add_curr else [])) / tot)
        s_indeces = []
        for r_idx in r_indeces:
            if r_idx != len(self.rollouts):
                rollout, r_len = self.rollouts[r_idx], self.rollouts_len[r_idx]
            else:
                # -1 because we don't have next_state on terminal
                rollout, r_len = self.curr_rollout, len(self.curr_rollout.states) - 1

            assert r_len > cluster_size - 1, "Rollout it too small"
            max_idx = r_len - cluster_size + 1
            if self.prioritize_ends:
                s_idx = np.random.choice(max_idx - cluster_size + 1, 1).item() + cluster_size - 1
            else:
                s_idx = np.random.choice(max_idx, 1).item()
            s_indeces.append(s_idx)

            if r_idx == len(self.rollouts):
                r_len += 1
                # FIXME: hot-fix for 1d action space, better to find smarter solution
                actions = np.array(rollout.actions[s_idx:s_idx + cluster_size]).reshape(cluster_size, -1)
            else:
                actions = rollout.actions[s_idx:s_idx + cluster_size]

            is_first.append(np.zeros(cluster_size))
            if s_idx == 0:
                is_first[-1][0] = 1
            s.append(rollout.states[s_idx:s_idx + cluster_size])
            a.append(actions)
            r.append(rollout.rewards[s_idx:s_idx + cluster_size])
            t.append(rollout.is_finished[s_idx:s_idx + cluster_size])
            if s_idx != r_len - cluster_size:
                n.append(rollout.states[s_idx+1:s_idx+1 + cluster_size])
            else:
                if cluster_size != 1:
                    n.append(rollout.states[s_idx+1:s_idx+1 + cluster_size - 1])
                n.append(rollout.next_states)
        return (np.concatenate(s), np.concatenate(a), np.concatenate(r, dtype=np.float32),
            np.concatenate(n), np.concatenate(t), np.concatenate(is_first))
