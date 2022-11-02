import typing as t
from collections import deque
from dataclasses import dataclass

import numpy as np
from nptyping import Bool, Float, Int, NDArray, Shape

Observation = NDArray[Shape["*,*,3"],Int]
State = NDArray[Shape["*"],Float]
Action = NDArray[Shape["*"],Int]

Observations = NDArray[Shape["*,*,*,3"],Int]
States = NDArray[Shape["*,*"],Float]
Actions = NDArray[Shape["*,*"],Int]
Rewards = NDArray[Shape["*"],Float]
TerminationFlags = NDArray[Shape["*"],Bool]

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
        self.max_len = max_len
        self.states: States = np.array([])
        self.actions: Actions = np.array([])
        self.rewards: Rewards = np.array([])
        self.next_states: States = np.array([])

    def add_rollout(self, rollout: Rollout):
        if len(self.states) == 0:
            self.states = rollout.states
            self.actions = rollout.actions
            self.rewards = rollout.rewards
            self.next_states = rollout.next_states
            self.is_finished = rollout.is_finished
        else:
            self.states = np.concatenate([self.states, rollout.states])
            self.actions = np.concatenate([self.actions, rollout.actions])
            self.rewards = np.concatenate([self.rewards, rollout.rewards])
            self.next_states = np.concatenate([self.next_states, rollout.next_states])
            self.is_finished = np.concatenate([self.is_finished, rollout.is_finished])

        if len(self.states) >= self.max_len:
            self.states = self.states
            self.actions = self.actions
            self.rewards = self.rewards
            self.next_states = self.next_states
            self.is_finished = self.is_finished

    def add_sample(self, s: State, a: Action, r: float, n: State, f: bool):
        rollout = Rollout(np.array([s]), np.expand_dims(np.array([a]), 0), np.array([r], dtype=np.float32), np.array([n]), np.array([f]))
        self.add_rollout(rollout)

    def can_sample(self, num: int):
        return len(self.states) >= num

    def sample(self, num: int) -> t.Tuple[States, Actions, Rewards, States, TerminationFlags]:
        indeces = np.random.choice(len(self.states), num)
        return self.states[indeces], self.actions[indeces], self.rewards[indeces], self.next_states[indeces], self.is_finished[indeces]
