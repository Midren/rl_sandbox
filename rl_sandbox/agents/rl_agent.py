from typing import Any
from abc import ABCMeta, abstractmethod

from rl_sandbox.utils.replay_buffer import Action, State, States, Actions, Rewards

class RlAgent(metaclass=ABCMeta):
    @abstractmethod
    def get_action(self, obs: State) -> Action:
        pass

    @abstractmethod
    def train(self, s: States, a: Actions, r: Rewards, next: States) -> dict[str, Any]:
        """
        Return dict with losses for logging
        """
        pass

    # Some models can have internal state which should be
    # properly reseted between rollouts
    def reset(self):
        pass
