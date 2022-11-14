import typing as t
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
        self.max_len = max_len
        self.states: States = np.array([])
        self.actions: Actions = np.array([])
        self.rewards: Rewards = np.array([])
        self.next_states: States = np.array([])
        self.observations: t.Optional[Observations]

    def add_rollout(self, rollout: Rollout):
        if len(self.states) == 0:
            self.states = rollout.states
            self.actions = rollout.actions
            self.rewards = rollout.rewards
            self.next_states = rollout.next_states
            self.is_finished = rollout.is_finished
            self.observations = rollout.observations
        else:
            self.states = np.concatenate([self.states, rollout.states])
            self.actions = np.concatenate([self.actions, rollout.actions])
            self.rewards = np.concatenate([self.rewards, rollout.rewards])
            self.next_states = np.concatenate([self.next_states, rollout.next_states])
            self.is_finished = np.concatenate([self.is_finished, rollout.is_finished])
            if self.observations is not None:
                self.observations = np.concatenate(
                    [self.observations, rollout.observations])

        if len(self.states) >= self.max_len:
            self.states = self.states[:self.max_len]
            self.actions = self.actions[:self.max_len]
            self.rewards = self.rewards[:self.max_len]
            self.next_states = self.next_states[:self.max_len]
            self.is_finished = self.is_finished[:self.max_len]
            if self.observations is not None:
                self.observations = self.observations[:self.max_len]

    def add_sample(self, s: State, a: Action, r: float, n: State, f: bool,
                   o: t.Optional[Observation] = None):
        rollout = Rollout(np.array([s]), np.array([a]),
                          np.array([r], dtype=np.float32), np.array([n]), np.array([f]),
                          np.array([o]) if o is not None else None)
        self.add_rollout(rollout)

    def can_sample(self, num: int):
        return len(self.states) >= num

    def sample(
        self,
        batch_size: int,
        cluster_size: int = 1
    ) -> t.Tuple[States, Actions, Rewards, States, TerminationFlags]:
        # TODO: add warning if batch_size % cluster_size != 0
        # FIXME: currently doesn't take into account discontinuations between between rollouts
        indeces = np.random.choice(len(self.states) - (cluster_size - 1), batch_size//cluster_size)
        indeces = np.stack([indeces + i for i in range(cluster_size)]).flatten(order='F')
        return self.states[indeces], self.actions[indeces], self.rewards[indeces], self.next_states[
            indeces], self.is_finished[indeces]
