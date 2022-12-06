import numpy as np
from nptyping import Float, NDArray, Shape

from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.env import Env
from rl_sandbox.utils.replay_buffer import (Action, Actions, Rewards, State,
                                            States, TerminationFlags)


class RandomAgent(RlAgent):
    def __init__(self, env: Env):
        self.action_space = env.action_space

    def get_action(self, obs: State) -> Action | NDArray[Shape["*"],Float]:
        return self.action_space.sample()

    def train(self, s: States, a: Actions, r: Rewards, next: States, is_finished: TerminationFlags):
        return dict()

    def save_ckpt(self, epoch_num: int, losses: dict[str, float]):
        pass
