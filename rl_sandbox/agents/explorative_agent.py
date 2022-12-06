import numpy as np
from nptyping import Float, NDArray, Shape

from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.schedulers import Scheduler
from rl_sandbox.utils.replay_buffer import (Action, Actions, Rewards, State,
                                            States, TerminationFlags)


class ExplorativeAgent(RlAgent):
    def __init__(self, policy_agent: RlAgent,
                       exploration_agent: RlAgent,
                       scheduler: Scheduler):
        self.policy_ag = policy_agent
        self.expl_ag = exploration_agent
        self.scheduler = scheduler

    def get_action(self, obs: State) -> Action | NDArray[Shape["*"],Float]:
        if np.random.random() > self.scheduler.step():
            return self.expl_ag.get_action(obs)
        return self.policy_ag.get_action(obs)

    def train(self, s: States, a: Actions, r: Rewards, next: States, is_finished: TerminationFlags):
        return self.expl_ag.train(s, a, r, next, is_finished) | self.policy_ag.train(s, a, r, next, is_finished)

    def save_ckpt(self, epoch_num: int, losses: dict[str, float]):
        self.policy_ag.save_ckpt(epoch_num, losses)
        self.expl_ag.save_ckpt(epoch_num, losses)
