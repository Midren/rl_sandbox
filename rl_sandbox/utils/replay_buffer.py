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


@dataclass
class Rollout:
    states: States
    actions: Actions
    rewards: Rewards
    next_states: States
    is_finished: TerminationFlags
    observations: t.Optional[Observations] = None


class ReplayBuffer:

    def __init__(self, max_len=2_000):
        self.rollouts: deque[Rollout] = deque()
        self.rollouts_len: deque[int] = deque()
        self.curr_rollout = None
        self.max_len = max_len
        self.total_num = 0

    def __len__(self):
        return self.total_num

    def add_rollout(self, rollout: Rollout):
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
                self.curr_rollout = None
                self.add_rollout(
                    Rollout(np.array(self.curr_rollout.states),
                            np.array(self.curr_rollout.actions),
                            np.array(self.curr_rollout.rewards, dtype=np.float32),
                            np.array([n]), np.array(self.curr_rollout.is_finished)))

    def can_sample(self, num: int):
        return self.total_num >= num

    def sample(
        self,
        batch_size: int,
        cluster_size: int = 1
    ) -> tuple[States, Actions, Rewards, States, TerminationFlags]:
        seq_num = batch_size // cluster_size
        # NOTE: constant creation of numpy arrays from self.rollout_len seems terrible for me
        s, a, r, n, t = [], [], [], [], []
        r_indeces = np.random.choice(len(self.rollouts),
                                     seq_num,
                                     p=np.array(self.rollouts_len) / self.total_num)
        for r_idx in r_indeces:
            # NOTE: maybe just no add such small rollouts to buffer
            assert self.rollouts_len[r_idx] - cluster_size + 1 > 0, "Rollout it too small"
            s_idx = np.random.choice(self.rollouts_len[r_idx] - cluster_size + 1, 1).item()

            s.append(self.rollouts[r_idx].states[s_idx:s_idx + cluster_size])
            a.append(self.rollouts[r_idx].actions[s_idx:s_idx + cluster_size])
            r.append(self.rollouts[r_idx].rewards[s_idx:s_idx + cluster_size])
            t.append(self.rollouts[r_idx].is_finished[s_idx:s_idx + cluster_size])
            if s_idx != self.rollouts_len[r_idx] - cluster_size:
                n.append(self.rollouts[r_idx].states[s_idx+1:s_idx+1 + cluster_size])
            else:
                if cluster_size != 1:
                    n.append(self.rollouts[r_idx].states[s_idx+1:s_idx+1 + cluster_size - 1])
                n.append(self.rollouts[r_idx].next_states)

        return (np.concatenate(s), np.concatenate(a), np.concatenate(r),
            np.concatenate(n), np.concatenate(t))
