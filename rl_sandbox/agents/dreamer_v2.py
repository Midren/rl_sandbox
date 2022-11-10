import itertools
import typing as t
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.fc_nn import fc_nn_generator
from rl_sandbox.utils.replay_buffer import (Action, Actions, Observation,
                                            Observations, Rewards,
                                            TerminationFlags)


class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class DebugShapeLayer(nn.Module):

    def __init__(self, note=""):
        super().__init__()
        self.note = note

    def forward(self, x):
        if len(self.note):
            print(self.note, x.shape)
        else:
            print(x.shape)
        return x


class Quantize(nn.Module):

    def forward(self, logits):
        dist = torch.distributions.one_hot_categorical.OneHotCategoricalStraightThrough(
            logits=logits)
        return dist


class RSSM(nn.Module):
    """
    Recurrent State Space Model

    h_t    <- deterministic state which is updated inside GRU
    s^_t   <- stohastic discrete prior state (used for KL divergence:
                                               better predict future and encode smarter)
    s_t    <- stohastic discrete posterior state (latent representation of current state)

      h_1            --->     h_2            --->     h_3            --->
         \\    x_1               \\    x_2               \\    x_3
       |  \\    |     ^        |  \\    |     ^        |  \\    |     ^
       v   MLP CNN    |        v   MLP CNN    |        v   MLP CNN    |
            \\  |     |             \\  |     |             \\  |     |
    Ensemble \\ |     |     Ensemble \\ |     |     Ensemble \\ |     |
              \\|     |               \\|     |               \\|     |
       |        |     |        |        |     |        |        |     |
       v        v     |        v        v     |        v        v     |
                      |                       |                       |
      s^_1     s_1 ---|       s^_2     s_2 ---|       s^_3     s_3 ---|

    """

    def __init__(self, latent_dim, hidden_size, actions_num, latent_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_classes = latent_classes
        self.ensemble_num = 5
        self.hidden_size = hidden_size

        # Calculate deterministic state from prev stochastic, prev action and prev deterministic
        self.pre_determ_recurrent = nn.Sequential(
            nn.Linear(latent_dim * latent_classes + actions_num,
                      hidden_size),  # Dreamer 'img_in'
            nn.LayerNorm(hidden_size),
        )
        self.determ_recurrent = nn.GRU(input_size=hidden_size,
                                       hidden_size=hidden_size)  # Dreamer gru '_cell'

        # Calculate stochastic state from prior embed
        # shared between all ensemble models
        self.ensemble_prior_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),  # Dreamer 'img_out_{k}'
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size,
                          latent_dim * self.latent_classes),  # Dreamer 'img_dist_{k}'
                View((-1, latent_dim, self.latent_classes)),
                Quantize()) for _ in range(self.ensemble_num)
        ])

        # For observation we do not have ensemble
        # FIXME: very bad magic number
        img_sz = 4 * 384  # 384*2x2
        self.stoch_net = nn.Sequential(
            nn.Linear(hidden_size + img_sz, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),  # Dreamer 'obs_out'
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size,
                      latent_dim * self.latent_classes),  # Dreamer 'obs_dist'
            View((-1, latent_dim, self.latent_classes)),
            # NOTE: Maybe worth having some LogSoftMax as activation
            #       before using input as logits for distribution
            Quantize())

    def estimate_stochastic_latent(self, prev_determ):
        dists_per_model = [model(prev_determ) for model in self.ensemble_prior_estimator]
        # NOTE: Maybe something smarter can be used instead of
        #       taking only one random between all ensembles
        idx = torch.randint(0, self.ensemble_num, ())
        return dists_per_model[idx]

    def predict_next(self,
                     stoch_latent,
                     action,
                     deter_state: t.Optional[torch.Tensor] = None):
        if deter_state is None:
            deter_state = torch.zeros(*stoch_latent.shape[:2], self.hidden_size).to(
                next(self.stoch_net.parameters()).device)
        x = self.pre_determ_recurrent(torch.concat([stoch_latent, action], dim=2))
        _, determ = self.determ_recurrent(x, deter_state)

        # used for KL divergence
        predicted_stoch_latent = self.estimate_stochastic_latent(determ)
        return deter_state, predicted_stoch_latent

    def update_current(self, determ, embed):  # Dreamer 'obs_out'
        return self.stoch_net(torch.concat([determ, embed], dim=2))

    def forward(self, h_prev: t.Optional[tuple[torch.Tensor, torch.Tensor]], embed,
                action):
        """
            'h' <- internal state of the world
            'z' <- latent embedding of current observation
            'a' <- action taken on prev step
            Returns 'h_next' <- the next next of the world
        """

        # Use zero vector for prev_state of first
        if h_prev is None:
            h_prev = (torch.zeros((*action.shape[:-1], self.hidden_size)),
                      torch.zeros(
                          (*action.shape[:-1], self.latent_dim * self.latent_classes)))
        deter_prev, stoch_prev = h_prev
        determ, prior_stoch_dist = self.predict_next(stoch_prev,
                                                     action,
                                                     deter_state=deter_prev)
        posterior_stoch_dist = self.update_current(determ, embed)

        return [determ, prior_stoch_dist, posterior_stoch_dist]


# NOTE: residual blocks are not used inside dreamer
class Encoder(nn.Module):

    def __init__(self, kernel_sizes=[4, 4, 4, 4]):
        super().__init__()
        layers = []

        channel_step = 48
        in_channels = 3
        for i, k in enumerate(kernel_sizes):
            out_channels = 2**i * channel_step
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=2))
            layers.append(nn.ELU(inplace=True))
            # FIXME: change to layer norm when sizes will be known
            layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class Decoder(nn.Module):

    def __init__(self, kernel_sizes=[5, 5, 6, 6]):
        super().__init__()
        layers = []
        self.channel_step = 48
        # 2**(len(kernel_sizes)-1)*channel_step
        self.convin = nn.Linear(32 * 32, 32 * self.channel_step)

        in_channels = 32 * self.channel_step  #2**(len(kernel_sizes) - 1) * self.channel_step
        for i, k in enumerate(kernel_sizes):
            out_channels = 2**(len(kernel_sizes) - i - 2) * self.channel_step
            if i == len(kernel_sizes) - 1:
                out_channels = 3
                layers.append(nn.ConvTranspose2d(in_channels, 3, kernel_size=k, stride=2))
            else:
                layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k,
                                       stride=2))
                layers.append(nn.ELU(inplace=True))
                layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        x = self.convin(X)
        x = x.view(-1, 32 * self.channel_step, 1, 1)
        return self.net(x)


class WorldModel(nn.Module):

    def __init__(self, batch_cluster_size, latent_dim, latent_classes, rssm_dim,
                 actions_num, kl_loss_scale):
        super().__init__()
        self.kl_beta = kl_loss_scale
        self.rssm_dim = rssm_dim
        self.latent_dim = latent_dim
        self.latent_classes = latent_classes
        self.cluster_size = batch_cluster_size
        # kl loss balancing (prior/posterior)
        self.alpha = 0.8

        self.recurrent_model = RSSM(latent_dim,
                                    rssm_dim,
                                    actions_num,
                                    latent_classes=latent_classes)
        self.encoder = Encoder()
        self.image_predictor = Decoder()
        self.reward_predictor = fc_nn_generator(rssm_dim + latent_dim * latent_classes,
                                                1,
                                                hidden_size=400,
                                                num_layers=4,
                                                intermediate_activation=nn.ELU)
        self.discount_predictor = fc_nn_generator(rssm_dim + latent_dim * latent_classes,
                                                  1,
                                                  hidden_size=400,
                                                  num_layers=4,
                                                  intermediate_activation=nn.ELU,
                                                  final_activation=nn.Sigmoid)

        self.optimizer = torch.optim.Adam(itertools.chain(
            self.recurrent_model.parameters(), self.encoder.parameters(),
            self.image_predictor.parameters(), self.reward_predictor.parameters(),
            self.discount_predictor.parameters()),
                                          lr=2e-4)

    def predict_next(self, latent_repr, action, world_state: t.Optional[torch.Tensor]):
        determ_state, next_repr_dist = self.recurrent_model.predict_next(
            latent_repr.unsqueeze(0), action.unsqueeze(0), world_state)

        next_repr = next_repr_dist.rsample().reshape(
            -1, self.latent_dim * self.latent_classes)
        reward = self.reward_predictor(
            torch.concat([determ_state.squeeze(0), next_repr], dim=1))
        is_finished = self.discount_predictor(
            torch.concat([determ_state.squeeze(0), next_repr], dim=1))
        return determ_state, next_repr, reward, is_finished

    def get_latent(self, obs: torch.Tensor, state):
        embed = self.encoder(obs)
        determ, _, latent_repr_dist = self.recurrent_model(state, embed.unsqueeze(0),
                                                           self._last_action)
        latent_repr = latent_repr_dist.rsample().reshape(-1, 32 * 32)
        return determ, latent_repr.unsqueeze(0)

    def train(self, obs: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
              is_finished: torch.Tensor):
        b, h, w, _ = obs.shape  # s <- BxHxWx3

        embed = self.encoder(obs)
        embed = embed.view(b // self.cluster_size, self.cluster_size, -1)

        obs = obs.view(-1, self.cluster_size, 3, h, w)
        a = a.view(-1, self.cluster_size, a.shape[1])
        r = r.view(-1, self.cluster_size, 1)
        f = is_finished.view(-1, self.cluster_size, 1)

        h_prev = None
        losses = defaultdict(lambda: torch.zeros(1).to(next(self.parameters()).device))

        def KL(dist1, dist2):
            KL_ = torch.distributions.kl_divergence
            Dist = torch.distributions.OneHotCategoricalStraightThrough
            return self.kl_beta * (
                self.alpha * KL_(dist1, Dist(logits=dist2.logits.detach())).mean() +
                (1 - self.alpha) * KL_(Dist(logits=dist1.logits.detach()), dist2).mean())

        latent_vars = []

        for t in range(self.cluster_size):
            # s_t <- 1xB^xHxWx3
            x_t, embed_t, a_t, r_t, f_t = obs[:, t], embed[:, t].unsqueeze(
                0), a[:, t].unsqueeze(0), r[:, t], f[:, t]

            determ_t, prior_stoch_dist, posterior_stoch_dist = self.recurrent_model(
                h_prev, embed_t, a_t)
            posterior_stoch = posterior_stoch_dist.rsample().reshape(
                -1, self.latent_dim * self.latent_classes)

            r_t_pred = self.reward_predictor(
                torch.concat([determ_t.squeeze(0), posterior_stoch], dim=1))
            f_t_pred = self.discount_predictor(
                torch.concat([determ_t.squeeze(0), posterior_stoch], dim=1))

            x_r = self.image_predictor(posterior_stoch)

            losses['loss_reconstruction'] = nn.functional.mse_loss(x_t, x_r)
            losses['loss_reward_pred'] += F.mse_loss(r_t, r_t_pred)
            losses['loss_discount_pred'] += F.cross_entropy(f_t.type(torch.float32),
                                                            f_t_pred)
            losses['loss_kl_reg'] += KL(prior_stoch_dist, posterior_stoch_dist)

            h_prev = [determ_t, posterior_stoch.unsqueeze(0)]
            latent_vars.append(posterior_stoch.detach())

        loss = torch.Tensor(1)
        for l in losses.values():
            loss += l

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        discovered_latents = torch.stack(latent_vars).reshape(
            -1, self.latent_dim * self.latent_classes)
        return {l: val.detach() for l, val in losses.items()}, discovered_latents


class ImaginativeCritic(nn.Module):

    def __init__(self, discount_factor: float, update_interval: int,
                 soft_update_fraction: float, value_target_lambda: float, latent_dim: int,
                 actions_num: int):
        super().__init__()
        self.gamma = discount_factor
        self.critic_update_interval = update_interval
        self.lambda_ = value_target_lambda
        self.critic_soft_update_fraction = soft_update_fraction
        self._update_num = 0

        self.critic = fc_nn_generator(latent_dim,
                                      actions_num,
                                      400,
                                      1,
                                      intermediate_activation=nn.ELU)
        self.target_critic = fc_nn_generator(latent_dim,
                                             actions_num,
                                             400,
                                             1,
                                             intermediate_activation=nn.ELU)

    def update_target(self):
        if self._update_num == 0:
            for target_param, local_param in zip(self.target_critic.parameters(),
                                                 self.critic.parameters()):
                mix = self.critic_soft_update_fraction
                target_param.data.copy_(mix * local_param.data +
                                        (1 - mix) * target_param.data)
        self._update_num = (self._update_num + 1) % self.critic_update_interval

    def estimate_value(self, z) -> torch.Tensor:
        return self.critic(z)

    def lambda_return(self, zs, rs, ts):
        v_lambdas = [self.target_critic(zs[-1])]
        for r, z, t in zip(reversed(rs[:-1]), reversed(zs[:-1]), reversed(ts[:-1])):
            v_lambda = r + t * self.gamma * (
                (1 - self.lambda_) * self.target_critic(z) + self.lambda_ * v_lambdas[-1])
            v_lambdas.append(v_lambda)
        return torch.concat(list(reversed(v_lambdas)), dim=0)


class DreamerV2(RlAgent):

    def __init__(
            self,
            obs_space_num: int,  # NOTE: encoder/decoder will work only with 64x64 currently
            actions_num: int,
            batch_cluster_size: int,
            latent_dim: int,
            latent_classes: int,
            rssm_dim: int,
            discount_factor: float,
            kl_loss_scale: float,
            imagination_horizon: int,
            critic_update_interval: int,
            actor_reinforce_fraction: float,
            actor_entropy_scale: float,
            critic_soft_update_fraction: float,
            critic_value_target_lambda: float,
            device_type: str = 'cpu'):

        self._state = None
        self._last_action = torch.zeros(actions_num)
        self.actions_num = actions_num
        self.imagination_horizon = imagination_horizon
        self.cluster_size = batch_cluster_size
        self.actions_num = actions_num
        self.rho = actor_reinforce_fraction
        if actor_reinforce_fraction != 0:
            raise NotImplementedError("Reinforce part is not implemented")
        self.eta = actor_entropy_scale

        self.world_model = WorldModel(batch_cluster_size, latent_dim, latent_classes,
                                      rssm_dim, actions_num,
                                      kl_loss_scale).to(device_type)
        self.actor = fc_nn_generator(latent_dim,
                                     actions_num,
                                     400,
                                     4,
                                     intermediate_activation=nn.ELU,
                                     final_activation=Quantize)
        # TODO: Leave only ImaginativeCritic and move Actor to DreamerV2
        self.critic = ImaginativeCritic(discount_factor, critic_update_interval,
                                        critic_soft_update_fraction,
                                        critic_value_target_lambda,
                                        latent_dim * latent_classes, actions_num)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=4e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

    def imagine_trajectory(
        self, z_0
    ) -> list[tuple[torch.Tensor, torch.distributions.Distribution, torch.Tensor,
                    torch.Tensor]]:
        rollout = []
        world_state = None
        z = z_0.detach().unsqueeze(0)
        for _ in range(self.imagination_horizon):
            a = self.actor(z)
            world_state, next_z, reward, is_finished = self.world_model.predict_next(
                z, a.rsample(), world_state)
            rollout.append(
                (z.detach(), a, next_z.detach(), reward.detach(), is_finished.detach()))
            z = next_z.detach()
        return rollout

    def reset(self):
        self._state = None
        self._last_action = torch.zeros((1, 1, self.actions_num))

    def preprocess_obs(self, obs: torch.Tensor):
        order = list(range(obs.shape))
        # Swap channel from last to 3 from last
        order = order[:-3] + [order[-1]] + [order[-3:-1]]
        return ((obs.type(torch.float32) / 255.0) - 0.5).permute(order)

    def get_action(self, obs: Observation) -> Action:
        # NOTE: pytorch fails without .copy() only when get_action is called
        obs = torch.from_numpy(obs.copy()).to(next(self.world_model.parameters()).device)
        obs = self.preprocess_obs(obs)

        self._state = self.world_model.get_latent(obs, self._state)
        self._last_action = self.actor(self._state[1]).rsample().unsqueeze(0)

        return self._last_action.squeeze().detach().cpu().numpy().argmax()

    def from_np(self, arr: np.ndarray):
        return torch.from_numpy(arr).to(next(self.world_model.parameters()).device)

    def train(self, obs: Observations, a: Actions, r: Rewards, next_obs: Observations,
              is_finished: TerminationFlags):

        obs = self.preprocess_obs(self.from_np(obs))
        a = self.from_np(a)
        a = F.one_hot(a, num_classes=self.actions_num).squeeze()
        r = self.from_np(r)
        next_obs = self.from_np(next_obs)
        is_finished = self.from_np(is_finished)

        # take some latent embeddings as initial step
        losses, discovered_latents = self.world_model.train(next_obs, a, r, is_finished)

        idx = torch.randperm(discovered_latents.size(0))
        initial_states = discovered_latents[idx]

        losses_ac = defaultdict(
            lambda: torch.zeros(1).to(next(self.critic.parameters()).device))

        for z_0 in initial_states:
            rollout = self.imagine_trajectory(z_0)
            zs, action_dists, next_zs, rewards, terminal_flags = zip(*rollout)
            vs = self.critic.lambda_return(next_zs, rewards, terminal_flags)

            losses_ac['loss_critic'] += F.mse_loss(self.critic.estimate_value(
                torch.stack(next_zs).squeeze(1)),
                                                   vs.detach(),
                                                   reduction='sum')

            losses_ac['loss_actor_reinforce'] += 0  # unused in dm_control
            losses_ac['loss_actor_dynamics_backprop'] += (-(1 - self.rho) * vs[-1]).mean()
            losses_ac['loss_actor_entropy'] += -self.eta * torch.stack(
                [a.entropy() for a in action_dists[:-1]]).mean()
            losses_ac['loss_actor'] += losses_ac['loss_actor_reinforce'] + losses_ac[
                'loss_actor_dynamics_backprop'] + losses_ac['loss_actor_entropy']

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        losses_ac['loss_critic'].backward()
        losses_ac['loss_actor'].backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.critic.update_target()

        losses_ac = {l: val.detach() for l, val in losses_ac.items()}

        return losses | losses_ac
