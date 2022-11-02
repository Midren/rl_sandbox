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
                    discount_factor: float,
                    device_type: str = 'cpu'):
        self.gamma = discount_factor
        self.value_func = fc_nn_generator(obs_space_num,
                                          actions_num,
                                          hidden_layer_size,
                                          num_layers).to(device_type)
        self.optimizer = torch.optim.Adam(self.value_func.parameters(), lr=1e-3)
        self.loss = torch.nn.MSELoss()
        self.device_type = device_type

    def get_action(self, obs: State) -> Action:
        return np.array(torch.argmax(self.value_func(torch.from_numpy(obs.reshape(1, -1)).to(self.device_type)), dim=1).detach().cpu())[0]

    def train(self, s: States, a: Actions, r: Rewards, next: States, is_finished: TerminationFlags):
        # Bellman error: MSE( (r + gamma * max_a Q(S_t+1, a)) -  Q(s_t, a) )
        # check for is finished

        s = torch.from_numpy(s).to(self.device_type)
        a = torch.from_numpy(a).to(self.device_type)
        r = torch.from_numpy(r).to(self.device_type)
        next = torch.from_numpy(next).to(self.device_type)
        is_finished = torch.from_numpy(is_finished).to(self.device_type)

        # TODO: normalize input
        # TODO: double dqn with target network
        values = self.value_func(next)
        indeces = torch.argmax(values, dim=1)
        target = r + (self.gamma * torch.gather(values, dim=1, index=indeces.unsqueeze(1)).squeeze(1)) * torch.logical_not(is_finished)

        loss = self.loss(torch.gather(self.value_func(s), dim=1, index=a).squeeze(1), target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu()
