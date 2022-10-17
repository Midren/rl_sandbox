import numpy as np
import torch

from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.fc_nn import fc_nn_generator
from rl_sandbox.utils.replay_buffer import (Action, Actions, Rewards, State,
                                            States, TerminationFlags)


class DqnAgent(RlAgent):
    def __init__(self, actions_num: int,
                    obs_space_num: int,
                    hidden_layer_size: int,
                    num_layers: int,
                    discount_factor: float):
        self.gamma = discount_factor
        self.value_func = fc_nn_generator(obs_space_num,
                                          actions_num,
                                          hidden_layer_size,
                                          num_layers)
        self.optimizer = torch.optim.Adam(self.value_func.parameters(), lr=1e-3)
        self.loss = torch.nn.MSELoss()

    def get_action(self, obs: State) -> Action:
        return np.array(torch.argmax(self.value_func(torch.from_numpy(obs)), dim=1))

    def train(self, s: States, a: Actions, r: Rewards, next: States, is_finished: TerminationFlags):
        # Bellman error: MSE( (r + gamma * max_a Q(S_t+1, a)) -  Q(s_t, a) )
        # check for is finished

        s = torch.from_numpy(s)
        a = torch.from_numpy(a)
        r = torch.from_numpy(r)
        next = torch.from_numpy(next)
        is_finished = torch.from_numpy(is_finished)

        values = self.value_func(next)
        indeces = torch.argmax(values, dim=1)
        x = r + (self.gamma * torch.gather(values, dim=1, index=indeces.unsqueeze(1)).squeeze(1)) * torch.logical_not(is_finished)

        loss = self.loss(x, torch.gather(self.value_func(s), dim=1, index=a).squeeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach()
