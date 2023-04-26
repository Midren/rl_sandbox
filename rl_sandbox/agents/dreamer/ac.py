import typing as t

import torch
import torch.distributions as td
from torch import nn

from rl_sandbox.utils.dists import DistLayer
from rl_sandbox.utils.fc_nn import fc_nn_generator


class ImaginativeCritic(nn.Module):

    def __init__(self, discount_factor: float, update_interval: int,
                 soft_update_fraction: float, value_target_lambda: float, latent_dim: int,
                 layer_norm: bool):
        super().__init__()
        self.gamma = discount_factor
        self.critic_update_interval = update_interval
        self.lambda_ = value_target_lambda
        self.critic_soft_update_fraction = soft_update_fraction
        self._update_num = 0

        self.critic = fc_nn_generator(latent_dim,
                                      1,
                                      400,
                                      5,
                                      intermediate_activation=nn.ELU,
                                      layer_norm=layer_norm,
                                      final_activation=DistLayer('mse'))
        self.target_critic = fc_nn_generator(latent_dim,
                                             1,
                                             400,
                                             5,
                                             intermediate_activation=nn.ELU,
                                             layer_norm=layer_norm,
                                             final_activation=DistLayer('mse'))
        self.target_critic.requires_grad_(False)

    def update_target(self):
        if self._update_num == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
            # for target_param, local_param in zip(self.target_critic.parameters(),
            #                                      self.critic.parameters()):
            #     mix = self.critic_soft_update_fraction
            #     target_param.data.copy_(mix * local_param.data +
            #                             (1 - mix) * target_param.data)
        self._update_num = (self._update_num + 1) % self.critic_update_interval

    def estimate_value(self, z) -> td.Distribution:
        return self.critic(z)

    def _lambda_return(self, vs: torch.Tensor, rs: torch.Tensor, ds: torch.Tensor):
        # Formula is actually slightly different than in paper
        # https://github.com/danijar/dreamerv2/issues/25
        v_lambdas = [vs[-1]]
        for i in range(rs.shape[0] - 1, -1, -1):
            v_lambda = rs[i] + ds[i] * (
                (1 - self.lambda_) * vs[i + 1] + self.lambda_ * v_lambdas[-1])
            v_lambdas.append(v_lambda)

        # FIXME: it copies array, so it is quite slow
        return torch.stack(v_lambdas).flip(dims=(0, ))[:-1]

    def lambda_return(self, zs, rs, ds):
        vs = self.target_critic(zs).mode
        return self._lambda_return(vs, rs, ds)

    def calculate_loss(self, zs: torch.Tensor, vs: torch.Tensor,
                       discount_factors: torch.Tensor):
        predicted_vs_dist = self.estimate_value(zs.detach())
        losses = {
            'loss_critic':
            -(predicted_vs_dist.log_prob(vs.detach()).unsqueeze(2) *
              discount_factors).mean()
        }
        metrics = {
            'critic/avg_target_value': self.target_critic(zs).mode.mean(),
            'critic/avg_lambda_value': vs.mean(),
            'critic/avg_predicted_value': predicted_vs_dist.mode.mean()
        }
        return losses, metrics


class ImaginativeActor(nn.Module):

    def __init__(self, latent_dim: int, actions_num: int, is_discrete: bool,
                 layer_norm: bool, reinforce_fraction: t.Optional[float],
                 entropy_scale: float):
        super().__init__()
        self.rho = reinforce_fraction
        if self.rho is None:
            self.rho = is_discrete
        self.eta = entropy_scale
        self.actor = fc_nn_generator(
            latent_dim,
            actions_num if is_discrete else actions_num * 2,
            400,
            5,
            layer_norm=layer_norm,
            intermediate_activation=nn.ELU,
            final_activation=DistLayer('onehot' if is_discrete else 'normal_trunc'))

    def forward(self, z: torch.Tensor) -> td.Distribution:
        return self.actor(z)

    def calculate_loss(self, zs: torch.Tensor, vs: torch.Tensor, baseline: torch.Tensor,
                       discount_factors: torch.Tensor, actions: torch.Tensor):
        losses = {}
        metrics = {}
        action_dists = self.actor(zs.detach())
        # baseline =
        advantage = (vs - baseline).detach()
        losses['loss_actor_reinforce'] = -(self.rho * action_dists.log_prob(
            actions.detach()).unsqueeze(2) * discount_factors * advantage).mean()
        losses['loss_actor_dynamics_backprop'] = -((1 - self.rho) *
                                                   (vs * discount_factors)).mean()

        def calculate_entropy(dist):
            # return dist.base_dist.base_dist.entropy().unsqueeze(2)
            return dist.entropy().unsqueeze(2)

        losses['loss_actor_entropy'] = -(self.eta * calculate_entropy(action_dists) *
                                         discount_factors).mean()
        losses['loss_actor'] = losses['loss_actor_reinforce'] + losses[
            'loss_actor_dynamics_backprop'] + losses['loss_actor_entropy']

        # mean and std are estimated statistically as tanh transformation is used
        sample = action_dists.rsample((128, ))
        act_avg = sample.mean(0)
        metrics['actor/avg_val'] = act_avg.mean()
        # metrics['actor/mode_val'] = action_dists.mode.mean()
        metrics['actor/mean_val'] = action_dists.mean.mean()
        metrics['actor/avg_sd'] = (((sample - act_avg)**2).mean(0).sqrt()).mean()
        metrics['actor/min_val'] = sample.min()
        metrics['actor/max_val'] = sample.max()

        return losses, metrics
