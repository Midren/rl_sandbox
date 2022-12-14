import itertools
import typing as t
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td

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


class Quantize(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return logits
        # return torch.distributions.one_hot_categorical.OneHotCategoricalStraightThrough(logits=logits)
        # return td.Independent(torch.distributions.OneHotCategoricalStraightThrough(logits=logits), 1)

def Dist(val):
    return td.Independent(torch.distributions.OneHotCategoricalStraightThrough(logits=val), 1)


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
            nn.ELU(inplace=True)
        )
        self.determ_recurrent = nn.GRU(input_size=hidden_size,
                                       hidden_size=hidden_size)  # Dreamer gru '_cell'

        # Calculate stochastic state from prior embed
        # shared between all ensemble models
        self.ensemble_prior_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),  # Dreamer 'img_out_{k}'
                nn.LayerNorm(hidden_size),
                nn.ELU(inplace=True),
                nn.Linear(hidden_size,
                          latent_dim * self.latent_classes),  # Dreamer 'img_dist_{k}'
                View((-1, latent_dim, self.latent_classes)),
                Quantize()) for _ in range(self.ensemble_num)
        ])

        # For observation we do not have ensemble
        # FIXME: very bad magic number
        img_sz = 4 * 384  # 384*2x2
        self.stoch_net = nn.Sequential(
            nn.Linear(hidden_size + img_sz, hidden_size), # Dreamer 'obs_out'
            nn.LayerNorm(hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size,
                      latent_dim * self.latent_classes),  # Dreamer 'obs_dist'
            View((-1, latent_dim, self.latent_classes)),
            # NOTE: Maybe worth having some LogSoftMax as activation
            #       before using input as logits for distribution
            Quantize())

    def estimate_stochastic_latent(self, prev_determ):
        # NOTE: Maybe something smarter can be used instead of
        #       taking only one random between all ensembles
        # FIXME: temporary use one model
        dists_per_model = [model(prev_determ) for model in self.ensemble_prior_estimator]
        idx = torch.randint(0, self.ensemble_num, ())
        return dists_per_model[idx]
        # return self.ensemble_prior_estimator[0](prev_determ)

    def predict_next(self,
                     stoch_latent,
                     action,
                     deter_state: t.Optional[torch.Tensor] = None):
        x = self.pre_determ_recurrent(torch.concat([stoch_latent, action], dim=1))
        _, determ = self.determ_recurrent(x.unsqueeze(0), deter_state.unsqueeze(0))
        determ = determ.squeeze(0)

        # used for KL divergence
        predicted_stoch_latent = self.estimate_stochastic_latent(determ)
        return determ, predicted_stoch_latent

    def update_current(self, determ, embed):  # Dreamer 'obs_out'
        return self.stoch_net(torch.concat([determ, embed], dim=1))

    def forward(self, h_prev: t.Optional[tuple[torch.Tensor, torch.Tensor]], embed,
                action):
        """
            'h' <- internal state of the world
            'z' <- latent embedding of current observation
            'a' <- action taken on prev step
            Returns 'h_next' <- the next next of the world
        """

        # FIXME: Use zero vector for prev_state of first
        # Move outside of rssm to omit checking
        deter_prev, stoch_prev = h_prev
        determ, prior_stoch_logits = self.predict_next(stoch_prev,
                                                     action,
                                                     deter_state=deter_prev)
        posterior_stoch_logits = self.update_current(determ, embed)

        return [determ, prior_stoch_logits, posterior_stoch_logits]


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

    def __init__(self, input_size, kernel_sizes=[5, 5, 6, 6]):
        super().__init__()
        layers = []
        self.channel_step = 48
        # 2**(len(kernel_sizes)-1)*channel_step
        self.convin = nn.Linear(input_size, 32 * self.channel_step)

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
                 actions_num, kl_loss_scale, kl_loss_balancing):
        super().__init__()
        self.kl_beta = kl_loss_scale
        self.rssm_dim = rssm_dim
        self.latent_dim = latent_dim
        self.latent_classes = latent_classes
        self.cluster_size = batch_cluster_size
        self.actions_num = actions_num
        # kl loss balancing (prior/posterior)
        self.alpha = kl_loss_balancing

        self.recurrent_model = RSSM(latent_dim,
                                    rssm_dim,
                                    actions_num,
                                    latent_classes=latent_classes)
        self.encoder = Encoder()
        self.image_predictor = Decoder(rssm_dim + latent_dim * latent_classes)
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

    def predict_next(self, latent_repr, action, world_state: t.Optional[torch.Tensor]):
        determ_state, next_repr_logits = self.recurrent_model.predict_next(
            latent_repr, action, world_state)

        next_repr = Dist(next_repr_logits).rsample().reshape(
            -1, self.latent_dim * self.latent_classes)
        reward = self.reward_predictor(
            torch.concat([determ_state, next_repr], dim=1))
        is_finished = self.discount_predictor(
            torch.concat([determ_state, next_repr], dim=1))
        return determ_state, next_repr, reward, is_finished

    def next_state(self, obs: torch.Tensor, action, state):
        embed = self.encoder(obs)
        determ, _, latent_repr_logits = self.recurrent_model(state, embed, action)
        return (determ, Dist(latent_repr_logits).rsample().reshape(-1, 32 * 32)), latent_repr_logits

    def calculate_loss(self, obs: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                       is_finished: torch.Tensor):
        b, _, h, w = obs.shape  # s <- BxHxWx3

        embed = self.encoder(obs)
        obs_c = obs.view(b // self.cluster_size, self.cluster_size, 3, h, w)
        embed_c = embed.view(b // self.cluster_size, self.cluster_size, -1)

        a_c = a.view(-1, self.cluster_size, self.actions_num)

        device = next(self.encoder.parameters()).device
        h_prev = [torch.zeros((b // self.cluster_size, 200), device=device),
                  torch.zeros((b // self.cluster_size, 32*32), device=device)]
        losses = defaultdict(lambda: torch.zeros(1).to(next(self.parameters()).device))

        def KL(dist1, dist2):
            KL_ = torch.distributions.kl_divergence
            one = torch.zeros(1,device=next(self.parameters()).device)
            # kl_lhs = torch.maximum(KL_(Dist(dist2.detach()), Dist(dist1)), one).view(-1, self.cluster_size).mean(dim=0).sum()
            # kl_rhs = torch.maximum(KL_(Dist(dist2), Dist(dist1.detach())), one).view(-1, self.cluster_size).mean(dim=0).sum()
            kl_lhs = torch.maximum(KL_(Dist(dist2.detach()), Dist(dist1)), one).mean()
            kl_rhs = torch.maximum(KL_(Dist(dist2), Dist(dist1.detach())), one).mean()
            return self.kl_beta * (self.alpha * kl_lhs + (1 - self.alpha) * kl_rhs)

        determ_vars = []
        prior_logits_vars = []
        posterior_logits_vars = []
        latent_vars = []

        for t in range(self.cluster_size):
            # s_t <- 1xB^xHxWx3
            embed_t, a_t = embed_c[:, t], a_c[:, t]

            determ_t, prior_stoch_logits, posterior_stoch_logits = self.recurrent_model(
                h_prev, embed_t, a_t)
            posterior_stoch = Dist(posterior_stoch_logits).rsample().reshape(
                -1, self.latent_dim * self.latent_classes)
            h_prev = [determ_t, posterior_stoch]

            obs_t = obs_c[:, t]
            obs_r = self.image_predictor(torch.concat([determ_t, posterior_stoch], dim=1))
            losses['loss_reconstruction'] += F.mse_loss(obs_t, obs_r)
            losses['loss_kl_reg'] += KL(prior_stoch_logits, posterior_stoch_logits)

            determ_vars.append(determ_t)
            prior_logits_vars.append(prior_stoch_logits)
            posterior_logits_vars.append(posterior_stoch_logits)
            latent_vars.append(posterior_stoch)

        latents = torch.stack(latent_vars)
        inp = torch.concat([torch.stack(determ_vars), latents], dim=2)
        inp = torch.flatten(inp, 0, 1)
        r_pred = self.reward_predictor(inp)
        f_pred = self.discount_predictor(inp)
        obs_r = self.image_predictor(inp)

        # losses['loss_reconstruction'] = F.mse_loss(obs, obs_r, reduction='none').view(-1, self.cluster_size, 3, h, w).mean(dim=(0, 2, 3, 4)).sum()
        losses['loss_reward_pred'] = F.mse_loss(r, r_pred.squeeze(1), reduction='none').mean(dim=0).sum()
        losses['loss_discount_pred'] = F.cross_entropy(is_finished.type(torch.float32), f_pred.squeeze(1), reduction='none').mean(dim=0).sum()
        # NOTE: entropy can be added as metric
        # losses['loss_kl_reg'] = KL(torch.concat(prior_logits_vars), torch.concat(posterior_logits_vars))


        return losses, latents.reshape(-1, self.latent_dim * self.latent_classes).detach()


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
                                      1,
                                      400,
                                      1,
                                      intermediate_activation=nn.ELU)
        self.target_critic = fc_nn_generator(latent_dim,
                                             1,
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
        for i in range(zs.shape[0] - 2, -1, -1):
            v_lambda = rs[i] + ts[i] * self.gamma * (
                (1 - self.lambda_) * self.target_critic(zs[i]).detach() +
                self.lambda_ * v_lambdas[-1])
            v_lambdas.append(v_lambda)

        return torch.stack(list(reversed(v_lambdas)))


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
            kl_loss_balancing: float,
            imagination_horizon: int,
            critic_update_interval: int,
            actor_reinforce_fraction: float,
            actor_entropy_scale: float,
            critic_soft_update_fraction: float,
            critic_value_target_lambda: float,
            world_model_lr: float,
            actor_lr: float,
            critic_lr: float,
            device_type: str = 'cpu'):

        self.actions_num = actions_num
        self.imagination_horizon = imagination_horizon
        self.cluster_size = batch_cluster_size
        self.actions_num = actions_num
        self.rho = actor_reinforce_fraction
        if actor_reinforce_fraction != 0:
            raise NotImplementedError("Reinforce part is not implemented")
        self.eta = actor_entropy_scale
        self.device_type = device_type

        self.world_model = WorldModel(batch_cluster_size, latent_dim, latent_classes,
                                      rssm_dim, actions_num, kl_loss_scale,
                                      kl_loss_balancing).to(device_type)
        # TODO: final activation should depend whether agent
        # action space in one hot or identity if real-valued
        self.actor = fc_nn_generator(latent_dim * latent_classes,
                                     actions_num,
                                     400,
                                     4,
                                     intermediate_activation=nn.ELU,
                                     final_activation=Quantize).to(device_type)
        self.critic = ImaginativeCritic(discount_factor, critic_update_interval,
                                        critic_soft_update_fraction,
                                        critic_value_target_lambda,
                                        latent_dim * latent_classes,
                                        actions_num).to(device_type)

        self.world_model_optimizer = torch.optim.AdamW(self.world_model.parameters(),
                                                       lr=world_model_lr,
                                                       eps=1e-5,
                                                       weight_decay=1e-6)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                 lr=actor_lr,
                                                 eps=1e-5,
                                                 weight_decay=1e-6)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),
                                                  lr=critic_lr,
                                                  eps=1e-5,
                                                  weight_decay=1e-6)
        self.reset()

    def imagine_trajectory(
        self, z_0
    ) -> tuple[torch.Tensor, torch.distributions.Distribution, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        world_state = torch.zeros(1, self.world_model.rssm_dim, device=z_0.device)
        zs, action_logits, next_zs, rewards, ts, determs = [], [], [], [], [], []
        z = z_0.detach()
        # FIXME: reward, ts should be predicted once at the end
        for _ in range(self.imagination_horizon):
            a = self.actor(z)
            world_state, next_z, reward, is_finished = self.world_model.predict_next(z, Dist(a).rsample(), world_state)

            zs.append(z)
            action_logits.append(a)
            next_zs.append(next_z)
            rewards.append(reward)
            ts.append(is_finished)
            determs.append(world_state)

            z = next_z.detach()
        return (torch.stack(zs), torch.stack(action_logits), torch.stack(next_zs),
                torch.stack(rewards).detach(), torch.stack(ts).detach(), torch.stack(determs))

    def reset(self):
        self._state = [torch.zeros((1, self.world_model.rssm_dim), device=self.device_type),
                       torch.zeros((1, 32*32), device=self.device_type)]
        self._last_action = torch.zeros((1, self.actions_num), device=self.device_type)
        self._latent_probs = torch.zeros((32, 32), device=self.device_type)
        self._action_probs = torch.zeros((self.actions_num), device=self.device_type)
        self._stored_steps = 0

    @staticmethod
    def preprocess_obs(obs: torch.Tensor):
        # FIXME: move to dataloader in replay buffer
        order = list(range(len(obs.shape)))
        # Swap channel from last to 3 from last
        order = order[:-3] + [order[-1]] + order[-3:-1]
        # return np.transpose((obs.astype(np.float32) / 255.0) - 0.5, axes=order)
        return ((obs.type(torch.float32) / 255.0) - 0.5).permute(order)

    def get_action(self, obs: Observation) -> Action:
        # NOTE: pytorch fails without .copy() only when get_action is called
        obs = torch.from_numpy(obs).to(next(self.world_model.parameters()).device)
        obs = self.preprocess_obs(obs).unsqueeze(0)
        # obs = obs.unsqueeze(0)

        self._state, latent_repr_logits = self.world_model.next_state(obs, self._last_action, self._state)

        actor_logits = self.actor(self._state[1])
        self._last_action = Dist(actor_logits).rsample()

        to_probs = lambda x: x.exp() / (1 + x.exp())
        self._action_probs += to_probs(actor_logits.squeeze())
        self._latent_probs += to_probs(latent_repr_logits.squeeze())
        self._stored_steps += 1

        return np.array([self._last_action.squeeze().detach().cpu().numpy().argmax()])

    def _generate_video(self, obs: Observation, init_action: Action):
        obs = torch.from_numpy(obs).to(next(self.world_model.parameters()).device)
        obs = self.preprocess_obs(obs).unsqueeze(0)
        # obs = obs.unsqueeze(0)

        action = F.one_hot(self.from_np(init_action).to(torch.int64),
                           num_classes=self.actions_num).squeeze()
        state = [torch.zeros(1, self.world_model.rssm_dim, device=obs.device), torch.zeros(1, 32*32, device=obs.device)]
        z_0 = self.world_model.next_state(obs, action.unsqueeze(0), state)[0][1]
        zs, _, _, _, _, determs = self.imagine_trajectory(z_0)
        video_r = self.world_model.image_predictor(torch.concat([determs, zs], dim=2)).cpu().detach().numpy()
        video_r = ((video_r + 0.5) * 255.0).astype(np.uint8)
        return video_r

    def viz_log(self, rollout, logger, epoch_num):
        init_indeces = np.random.choice(len(rollout.states) - self.imagination_horizon, 3)

        videos_r = np.concatenate([self._generate_video(obs_0, a_0) for obs_0, a_0 in zip(rollout.states[init_indeces], rollout.actions[init_indeces]) ], axis=-1)
        videos = np.concatenate([rollout.states[init_idx:init_idx + self.imagination_horizon] for init_idx in init_indeces], axis=-2).transpose(0, 3, 1, 2)

        videos_comparison = np.expand_dims(np.concatenate([videos, videos_r], axis=-2), 0)
        latent_hist = (self._latent_probs / self._stored_steps).detach().cpu().numpy()
        latent_hist = ((latent_hist / latent_hist.max() * 255.0 )).astype(np.uint8)
        action_hist = (self._action_probs / self._stored_steps).detach().cpu().numpy()

        # logger.add_histogram('val/action_probs', action_hist, epoch_num)
        fig = plt.Figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(np.arange(self.actions_num), action_hist)
        logger.add_figure('val/action_probs', fig, epoch_num)
        logger.add_image('val/latent_probs', latent_hist, epoch_num, dataformats='HW')
        logger.add_image('val/latent_probs_sorted', np.sort(latent_hist, axis=1), epoch_num, dataformats='HW')
        logger.add_video('val/dreamed_rollout', videos_comparison, epoch_num)

    def from_np(self, arr: np.ndarray):
        arr = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
        return arr.to(next(self.world_model.parameters()).device, non_blocking=True)

    def train(self, obs: Observations, a: Actions, r: Rewards, next_obs: Observations,
              is_finished: TerminationFlags):

        obs = self.preprocess_obs(self.from_np(obs))
        # obs = self.from_np(self.preprocess_obs(obs))
        # obs = self.from_np(obs)
        a = self.from_np(a).to(torch.int64)
        a = F.one_hot(a, num_classes=self.actions_num).squeeze()
        r = self.from_np(r)
        next_obs = self.preprocess_obs(self.from_np(next_obs))
        # next_obs = self.from_np(next_obs)
        is_finished = self.from_np(is_finished)

        # take some latent embeddings as initial step
        losses, discovered_latents = self.world_model.calculate_loss(
            next_obs, a, r, is_finished)

        # NOTE: 'aten::nonzero' inside KL divergence is not currently supported on M1 Pro MPS device
        world_model_loss = losses['loss_reconstruction'] + losses['loss_reward_pred'] + losses['loss_discount_pred'] + losses['loss_kl_reg']
        losses = {l: val.detach() for l, val in losses.items()}

        # idx = torch.randperm(discovered_latents.size(0))
        # initial_states = discovered_latents[idx]

        # losses_ac = defaultdict(
        #     lambda: torch.zeros(1).to(next(self.critic.parameters()).device))

        # zs, action_logits, next_zs, rewards, terminal_flags, _ = self.imagine_trajectory(
        #     initial_states)
        # vs = self.critic.lambda_return(next_zs, rewards, terminal_flags)

        # losses_ac['loss_critic'] = F.mse_loss(self.critic.estimate_value(next_zs.detach()),
        #                                       vs.detach(),
        #                                       reduction='sum') / zs.shape[1]
        # losses_ac['loss_actor_reinforce'] += 0  # unused in dm_control
        # losses_ac['loss_actor_dynamics_backprop'] = -(
        #     (1 - self.rho) * vs).sum() / zs.shape[1]
        # FIXME: use single entropy with currently returned logits
        # losses_ac['loss_actor_entropy'] = -self.eta * torch.stack(
        #     [a.entropy() for a in action_dists[:-1]]).sum() / zs.shape[1]
        # losses_ac['loss_actor'] += losses_ac['loss_actor_reinforce'] + losses_ac[
        #     'loss_actor_dynamics_backprop'] + losses_ac['loss_actor_entropy']

        self.world_model_optimizer.zero_grad()
        # self.actor_optimizer.zero_grad()
        # self.critic_optimizer.zero_grad()
        world_model_loss.backward()
        self.world_model_optimizer.step()

        # losses_ac['loss_critic'].backward()
        # losses_ac['loss_actor'].backward()
        # self.actor_optimizer.step()
        # self.critic_optimizer.step()
        # self.critic.update_target()

        losses = {l: val.cpu().item() for l, val in losses.items()}
        # losses_ac = {l: val.detach().cpu().item() for l, val in losses_ac.items()}

        return losses
        # return losses | losses_ac

    def save_ckpt(self, epoch_num: int, losses: dict[str, float]):
        torch.save(
            {
                'epoch': epoch_num,
                'world_model_state_dict': self.world_model.state_dict(),
                'world_model_optimizer_state_dict':
                self.world_model_optimizer.state_dict(),
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'losses': losses
            }, f'dreamerV2-{epoch_num}-{sum(losses.values())}.ckpt')
