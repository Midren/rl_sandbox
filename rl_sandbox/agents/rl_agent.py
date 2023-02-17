from typing import Any
from abc import ABCMeta, abstractmethod
from pathlib import Path

from rl_sandbox.utils.replay_buffer import Action, State, States, Actions, Rewards, TerminationFlags

class RlAgent(metaclass=ABCMeta):
    @abstractmethod
    def get_action(self, obs: State) -> Action:
        pass

    @abstractmethod
    def train(self, s: States, a: Actions, r: Rewards, next: States, is_finished: TerminationFlags) -> dict[str, Any]:
        """
        Return dict with losses for logging
        """
        pass

    # Some models can have internal state which should be
    # properly reseted between rollouts
    def reset(self):
        pass

    @abstractmethod
    def save_ckpt(self, epoch_num: int, losses: dict[str, float]):
        pass

    @abstractmethod
    def load_ckpt(self, ckpt_path: Path):
        pass
