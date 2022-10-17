import random
import typing as t
from collections import deque

import numpy as np
from nptyping import Bool, Int, Float, NDArray, Shape

State = NDArray[Shape["*"],Float]
Action = NDArray[Shape["*"],Int]

States = NDArray[Shape["*,*"],Float]
Actions = NDArray[Shape["*,*"],Int]
Rewards = NDArray[Shape["*"],Float]
TerminationFlag = NDArray[Shape["*"],Bool]


# ReplayBuffer consists of next triplets: (s, a, r)
class ReplayBuffer:
    def __init__(self, max_len=10_000):
        self.max_len = max_len
        self.states: States = np.array([])
        self.actions: Actions = np.array([])
        self.rewards: Rewards = np.array([])
        self.next_states: States = np.array([])

    def add_rollout(self, s: States, a: Actions, r: Rewards, n: States, f: TerminationFlag):
        if len(self.states) == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = n
            self.is_finished = f
        else:
            self.states = np.concatenate([self.states, s])
            self.actions = np.concatenate([self.actions, a])
            self.rewards = np.concatenate([self.rewards, r])
            self.next_states = np.concatenate([self.next_states, n])
            self.is_finished = np.concatenate([self.is_finished, f])

    def can_sample(self, num: int):
        return len(self.states) >= num

    def sample(self, num: int) -> t.Tuple[States, Actions, Rewards, States, TerminationFlag]:
        indeces = list(range(len(self.states)))
        random.shuffle(indeces)
        indeces = indeces[:num]
        return self.states[indeces], self.actions[indeces], self.rewards[indeces], self.next_states[indeces], self.is_finished[indeces]
