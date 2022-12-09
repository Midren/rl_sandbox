import itertools
import typing as t
from collections import defaultdict

import matplotlib.pyplot as plt
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
        return torch.distributions.one_hot_categorical.OneHotCategoricalStraightThrough(
            logits=logits)


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
        # FIXME: check whether it is trully correct
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
        dists_per_model = [model(prev_determ) for model in self.ensemble_prior_estimator]
        # NOTE: Maybe something smarter can be used instead of
        #       taking only one random between all ensembles
        # FIXME: temporary use the same model
        idx = torch.randint(0, self.ensemble_num, ())
        return dists_per_model[0]

    def predict_next(self,
                     stoch_latent,
                     action,
                     deter_state: t.Optional[torch.Tensor] = None):
        # FIXME: Move outside of rssm to omit checking
        if deter_state is None:
            deter_state = torch.zeros(*stoch_latent.shape[:2], self.hidden_size).to(
                next(self.stoch_net.parameters()).device)
        x = self.pre_determ_recurrent(torch.concat([stoch_latent, action], dim=2))
        _, determ = self.determ_recurrent(x, deter_state)

        # used for KL divergence
        predicted_stoch_latent = self.estimate_stochastic_latent(determ)
        return determ, predicted_stoch_latent

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

        # FIXME: Use zero vector for prev_state of first
        # Move outside of rssm to omit checking
        if h_prev is None:
            h_prev = (torch.zeros((
                *action.shape[:-1],
                self.hidden_size,
            ),
                                  device=next(self.stoch_net.parameters()).device),
                      torch.zeros(
                          (*action.shape[:-1], self.latent_dim * self.latent_classes),
                          device=next(self.stoch_net.parameters()).device))
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
        determ_state, next_repr_dist = self.recurrent_model.predict_next(
            latent_repr.unsqueeze(0), action.unsqueeze(0), world_state)

        next_repr = next_repr_dist.rsample().reshape(
            -1, self.latent_dim * self.latent_classes)
        reward = self.reward_predictor(
            torch.concat([determ_state.squeeze(0), next_repr], dim=1))
        is_finished = self.discount_predictor(
            torch.concat([determ_state.squeeze(0), next_repr], dim=1))
        return determ_state, next_repr, reward, is_finished

    def get_latent(self, obs: torch.Tensor, action, state):
        embed = self.encoder(obs)
        determ, _, latent_repr_dist = self.recurrent_model(state, embed.unsqueeze(0),
                                                           action)
        return determ, latent_repr_dist

    def calculate_loss(self, obs: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                       is_finished: torch.Tensor):
        b, _, h, w = obs.shape  # s <- BxHxWx3

        embed = self.encoder(obs)
        embed = embed.view(b // self.cluster_size, self.cluster_size, -1)

        obs = obs.view(-1, self.cluster_size, 3, h, w)
        a = a.view(-1, self.cluster_size, self.actions_num)
        r = r.view(-1, self.cluster_size, 1)
        f = is_finished.view(-1, self.cluster_size, 1)

        h_prev = None
        losses = defaultdict(lambda: torch.zeros(1).to(next(self.parameters()).device))

        def KL(dist1, dist2):
            KL_ = torch.distributions.kl_divergence
            Dist = torch.distributions.OneHotCategoricalStraightThrough
            one = torch.ones(1,device=next(self.parameters()).device)
            kl_lhs = torch.maximum(KL_(Dist(logits=dist2.logits.detach()), dist1), one)
            kl_rhs = torch.maximum(KL_(dist2, Dist(logits=dist1.logits.detach())), one)
            return self.kl_beta * (self.alpha * kl_lhs.mean()+ (1 - self.alpha) * kl_rhs.mean())

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
                torch.concat([determ_t.squeeze(0), posterior_stoch], dim=1).detach())
            f_t_pred = self.discount_predictor(
                torch.concat([determ_t.squeeze(0), posterior_stoch], dim=1))

            x_r = self.image_predictor(torch.concat([determ_t.squeeze(0), posterior_stoch], dim=1))

            losses['loss_reconstruction'] += nn.functional.mse_loss(x_t, x_r)
            losses['loss_reward_pred'] += F.mse_loss(r_t, r_t_pred)
            losses['loss_discount_pred'] += F.cross_entropy(f_t.type(torch.float32),
                                                            f_t_pred)
            # NOTE: entropy can be added as metric
            losses['loss_kl_reg'] += KL(prior_stoch_dist, posterior_stoch_dist)

            h_prev = [determ_t, posterior_stoch.unsqueeze(0)]
            latent_vars.append(posterior_stoch.detach())

        discovered_latents = torch.stack(latent_vars).reshape(
            -1, self.latent_dim * self.latent_classes)
        return losses, discovered_latents


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
        world_state = None
        zs, actions, next_zs, rewards, ts, determs = [], [], [], [], [], []
        z = z_0.detach()
        for _ in range(self.imagination_horizon):
            a = self.actor(z)
            world_state, next_z, reward, is_finished = self.world_model.predict_next(
                z, a.rsample(), world_state)

            zs.append(z)
            actions.append(a)
            next_zs.append(next_z)
            rewards.append(reward)
            ts.append(is_finished)
            determs.append(world_state[0])

            z = next_z.detach()
        return (torch.stack(zs), actions, torch.stack(next_zs),
                torch.stack(rewards).detach(), torch.stack(ts).detach(), torch.stack(determs))

    def reset(self):
        self._state = None
        self._last_action = torch.zeros((1, 1, self.actions_num),
                                        device=next(self.world_model.parameters()).device)
        self._latent_probs = torch.zeros((32, 32), device=next(self.world_model.parameters()).device)
        self._action_probs = torch.zeros((self.actions_num), device=next(self.world_model.parameters()).device)
        self._stored_steps = 0

    @staticmethod
    def preprocess_obs(obs: torch.Tensor):
        # FIXME: move to dataloader in replay buffer
        order = list(range(len(obs.shape)))
        # Swap channel from last to 3 from last
        order = order[:-3] + [order[-1]] + order[-3:-1]
        return ((obs.type(torch.float32) / 255.0) - 0.5).permute(order)

    def get_action(self, obs: Observation) -> Action:
        # NOTE: pytorch fails without .copy() only when get_action is called
        obs = torch.from_numpy(obs.copy()).to(next(self.world_model.parameters()).device)
        obs = self.preprocess_obs(obs).unsqueeze(0)

        determ, latent_repr_dist = self.world_model.get_latent(obs, self._last_action,
                                                               self._state)
        self._state = (determ, latent_repr_dist.rsample().reshape(-1,
                                                                  32 * 32).unsqueeze(0))

        actor_dist = self.actor(self._state[1])
        self._last_action = actor_dist.rsample()

        self._action_probs += actor_dist.probs.squeeze()
        self._latent_probs += latent_repr_dist.probs.squeeze()
        self._stored_steps += 1

        return np.array([self._last_action.squeeze().detach().cpu().numpy().argmax()])

    def _generate_video(self, obs: Observation, init_action: Action):
        obs = torch.from_numpy(obs.copy()).to(next(self.world_model.parameters()).device)
        obs = self.preprocess_obs(obs).unsqueeze(0)

        action = F.one_hot(self.from_np(init_action).to(torch.int64),
                           num_classes=self.actions_num).squeeze()
        z_0 = self.world_model.get_latent(obs,
                                          action.unsqueeze(0).unsqueeze(0),
                                          None)[1].rsample().reshape(-1,
                                                                     32 * 32).unsqueeze(0)
        zs, _, _, _, _, determs = self.imagine_trajectory(z_0.squeeze(0))
        video_r = self.world_model.image_predictor(torch.concat([determs, zs], dim=2)).cpu().detach().numpy()
        video_r = ((video_r + 0.5) * 255.0).astype(np.uint8)
        return video_r

    def viz_log(self, rollout, logger, epoch_num):
        init_indeces = np.random.choice(len(rollout.states) - self.imagination_horizon, 3)
        videos_r = np.concatenate([
            self._generate_video(obs_0.copy(), a_0) for obs_0, a_0 in zip(
                rollout.states[init_indeces], rollout.actions[init_indeces])
        ],
                                  axis=3)

        videos = np.concatenate([
            rollout.states[init_idx:init_idx + self.imagination_horizon].transpose(
                0, 3, 1, 2) for init_idx in init_indeces
        ],
                                axis=3)
        videos_comparison = np.expand_dims(np.concatenate([videos, videos_r], axis=2), 0)
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
        a = self.from_np(a).to(torch.int64)
        a = F.one_hot(a, num_classes=self.actions_num).squeeze()
        r = self.from_np(r)
        next_obs = self.preprocess_obs(self.from_np(next_obs))
        is_finished = self.from_np(is_finished)

        # take some latent embeddings as initial step
        losses, discovered_latents = self.world_model.calculate_loss(
            next_obs, a, r, is_finished)

        # NOTE: 'aten::nonzero' inside KL divergence is not currently supported on M1 Pro MPS device
        world_model_loss = torch.Tensor(1).to(next(self.world_model.parameters()).device)
        for l in losses.values():
            world_model_loss += l

        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        self.world_model_optimizer.step()

        idx = torch.randperm(discovered_latents.size(0))
        initial_states = discovered_latents[idx]

        losses_ac = defaultdict(
            lambda: torch.zeros(1).to(next(self.critic.parameters()).device))

        zs, action_dists, next_zs, rewards, terminal_flags, _ = self.imagine_trajectory(
            initial_states)
        vs = self.critic.lambda_return(next_zs, rewards, terminal_flags)

        losses_ac['loss_critic'] = F.mse_loss(self.critic.estimate_value(next_zs.detach()),
                                              vs.detach(),
                                              reduction='sum') / zs.shape[1]
        losses_ac['loss_actor_reinforce'] += 0  # unused in dm_control
        losses_ac['loss_actor_dynamics_backprop'] = -(
            (1 - self.rho) * vs).sum() / zs.shape[1]
        losses_ac['loss_actor_entropy'] = -self.eta * torch.stack(
            [a.entropy() for a in action_dists[:-1]]).sum() / zs.shape[1]
        losses_ac['loss_actor'] += losses_ac['loss_actor_reinforce'] + losses_ac[
            'loss_actor_dynamics_backprop'] + losses_ac['loss_actor_entropy']

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        losses_ac['loss_critic'].backward()
        losses_ac['loss_actor'].backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.critic.update_target()

        losses = {l: val.detach().cpu().item() for l, val in losses.items()}
        losses_ac = {l: val.detach().cpu().item() for l, val in losses_ac.items()}

        return losses | losses_ac

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
