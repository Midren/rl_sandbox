import gym
import numpy as np
from dm_control.composer.environment import Environment as dmEnv
from nptyping import Float, NDArray, Shape

from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.dm_control import ActionDiscritizer
from rl_sandbox.utils.replay_buffer import (Action, Actions, Rewards, State,
                                            States, TerminationFlags)


class RandomAgent(RlAgent):
    def __init__(self, env: gym.Env | dmEnv):
        self.action_space = None
        self.action_spec = None
        if isinstance(env, gym.Env):
            self.action_space = env.action_space
        else:
            self.action_spec = env.action_spec()

    def get_action(self, obs: State) -> Action | NDArray[Shape["*"],Float]:
        if self.action_space is not None:
            return self.action_space.sample()
        else:
            return np.random.uniform(self.action_spec.minimum,
                             self.action_spec.maximum,
                             size=self.action_spec.shape)

    def train(self, s: States, a: Actions, r: Rewards, next: States, is_finished: TerminationFlags):
        pass
