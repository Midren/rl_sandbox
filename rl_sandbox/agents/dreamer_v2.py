import typing as t
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
from torch import nn
from torch.nn import functional as F

from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.fc_nn import fc_nn_generator
from rl_sandbox.utils.replay_buffer import (Action, Actions, Observation,
                                            Observations, Rewards,
                                            TerminationFlags, IsFirstFlags)
from rl_sandbox.utils.dists import TruncatedNormal

class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Sigmoid2(nn.Module):
    def forward(self, x):
        return 2*torch.sigmoid(x/2)

class NormalWithOffset(nn.Module):
    def __init__(self, min_std: float, std_trans: str = 'sigmoid2', transform: t.Optional[str] = None):
        super().__init__()
        self.min_std = min_std
        match std_trans:
            case 'identity':
                self.std_trans = nn.Identity()
            case 'softplus':
                self.std_trans = nn.Softplus()
            case 'sigmoid':
                self.std_trans = nn.Sigmoid()
            case 'sigmoid2':
                self.std_trans = Sigmoid2()
            case _:
                raise RuntimeError("Unknown std transformation")

        match transform:
            case 'tanh':
                self.trans = [td.TanhTransform(cache_size=1)]
            case None:
                self.trans = None
            case _:
                raise RuntimeError("Unknown distribution transformation")

    def forward(self, x):
        mean, std = x.chunk(2, dim=-1)
        dist = td.Normal(mean, self.std_trans(std) + self.min_std)
        if self.trans is None:
            return dist
        else:
            return td.TransformedDistribution(dist, self.trans)

class DistLayer(nn.Module):
    def __init__(self, type: str):
        super().__init__()
        match type:
            case 'mse':
                self.dist = lambda x: td.Normal(x.float(), 1.0)
            case 'normal':
                self.dist = NormalWithOffset(min_std=0.1)
            case 'onehot':
                # Forcing float32 on AMP
                self.dist = lambda x: td.OneHotCategoricalStraightThrough(logits=x.float())
            case 'normal_tanh':
                def get_tanh_normal(x, min_std=0.1):
                    mean, std = x.chunk(2, dim=-1)
                    init_std = np.log(np.exp(5) - 1)
                    raise NotImplementedError()
                    # return TanhNormal(torch.clamp(mean, -9.0, 9.0).float(), (F.softplus(std + init_std) + min_std).float(), upscale=5)
                self.dist = get_tanh_normal
            case 'normal_trunc':
                def get_trunc_normal(x, min_std=0.1):
                    mean, std = x.chunk(2, dim=-1)
                    return TruncatedNormal(loc=torch.tanh(mean).float(), scale=(2*torch.sigmoid(std/2) + min_std).float(), a=-1, b=1)
                self.dist = get_trunc_normal
            case 'binary':
                self.dist = lambda x: td.Bernoulli(logits=x)
            case _:
                raise RuntimeError("Invalid dist layer")

    def forward(self, x):
        return td.Independent(self.dist(x), 1)

def Dist(val):
    return DistLayer('onehot')(val)


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
        self.ensemble_num = 1
        self.hidden_size = hidden_size

        # Calculate deterministic state from prev stochastic, prev action and prev deterministic
        self.pre_determ_recurrent = nn.Sequential(
            nn.Linear(latent_dim * latent_classes + actions_num,
                      hidden_size),  # Dreamer 'img_in'
            # nn.LayerNorm(hidden_size),
            nn.ELU(inplace=True)
        )
        self.determ_recurrent = nn.GRU(input_size=hidden_size,
                                       hidden_size=hidden_size)  # Dreamer gru '_cell'

        # Calculate stochastic state from prior embed
        # shared between all ensemble models
        self.ensemble_prior_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),  # Dreamer 'img_out_{k}'
                # nn.LayerNorm(hidden_size),
                nn.ELU(inplace=True),
                nn.Linear(hidden_size,
                          latent_dim * self.latent_classes),  # Dreamer 'img_dist_{k}'
                View((-1, latent_dim, self.latent_classes))) for _ in range(self.ensemble_num)
        ])

        # For observation we do not have ensemble
        # FIXME: very bad magic number
        img_sz = 4 * 384  # 384*2x2
        self.stoch_net = nn.Sequential(
            nn.Linear(hidden_size + img_sz, hidden_size), # Dreamer 'obs_out'
            # nn.LayerNorm(hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size,
                      latent_dim * self.latent_classes),  # Dreamer 'obs_dist'
            View((-1, latent_dim, self.latent_classes)))

    def estimate_stochastic_latent(self, prev_determ):
        dists_per_model = [model(prev_determ) for model in self.ensemble_prior_estimator]
        # NOTE: Maybe something smarter can be used instead of
        #       taking only one random between all ensembles
        # NOTE: in Dreamer ensemble_num is always 1
        idx = torch.randint(0, self.ensemble_num, ())
        return dists_per_model[idx]

    def predict_next(self,
                     stoch_latent,
                     action,
                     deter_state: t.Optional[torch.Tensor] = None):
        # FIXME: Move outside of rssm to omit checking
        if deter_state is None:
            deter_state = torch.zeros(*stoch_latent.shape[:2], self.hidden_size).to(
                next(self.stoch_net.parameters()).device)
        x = self.pre_determ_recurrent(torch.concat([stoch_latent, action], dim=-1))
        # NOTE: x and determ are actually the same value if sequence of 1 is inserted
        x, determ = self.determ_recurrent(x, deter_state)

        # used for KL divergence
        predicted_stoch_logits = self.estimate_stochastic_latent(x)
        return determ, predicted_stoch_logits

    def update_current(self, determ, embed):  # Dreamer 'obs_out'
        return self.stoch_net(torch.concat([determ, embed], dim=-1))

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
                *embed.shape[:-1],
                self.hidden_size,
            ),
                                  device=next(self.stoch_net.parameters()).device),
                      torch.zeros(
                          (*action.shape[:-1], self.latent_dim * self.latent_classes),
                          device=next(self.stoch_net.parameters()).device))
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
            # layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ELU(inplace=True))
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
                # layers.append(nn.BatchNorm2d(in_channels))
                layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k,
                                       stride=2))
                layers.append(nn.ELU(inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        x = self.convin(X)
        x = x.view(-1, 32 * self.channel_step, 1, 1)
        return td.Independent(td.Normal(self.net(x), 1.0), 3)


class Normalizer(nn.Module):
    def __init__(self, momentum=0.99, scale=1.0, eps=1e-8):
        super().__init__()
        self.momentum = momentum
        self.scale = scale
        self.eps= eps
        self.mag = torch.ones(1, dtype=torch.float32)
        self.mag.requires_grad = False

    def forward(self, x):
        self.update(x)
        return (x / (self.mag + self.eps))*self.scale

    def update(self, x):
        self.mag = self.momentum * self.mag.to(x.device)  + (1 - self.momentum) * (x.abs().mean()).detach()


class WorldModel(nn.Module):

    def __init__(self, batch_cluster_size, latent_dim, latent_classes, rssm_dim,
                 actions_num, kl_loss_scale, kl_loss_balancing, kl_free_nats):
        super().__init__()
        self.kl_free_nats = kl_free_nats
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
                                                intermediate_activation=nn.ELU,
                                                final_activation=DistLayer('mse'))
        self.discount_predictor = fc_nn_generator(rssm_dim + latent_dim * latent_classes,
                                                  1,
                                                  hidden_size=400,
                                                  num_layers=4,
                                                  intermediate_activation=nn.ELU,
                                                  final_activation=DistLayer('binary'))
        self.reward_normalizer = Normalizer(momentum=1.00, scale=1.0, eps=1e-8)

    def predict_next(self, latent_repr, action, world_state: t.Optional[torch.Tensor]):
        determ_state, next_repr_logits = self.recurrent_model.predict_next(
            latent_repr.unsqueeze(0), action.unsqueeze(0), world_state)

        next_repr = Dist(next_repr_logits).rsample().reshape(
            -1, self.latent_dim * self.latent_classes)
        inp = torch.concat([determ_state.squeeze(0), next_repr], dim=-1)
        reward = self.reward_predictor(inp).mode
        discount_factors = self.discount_predictor(inp).sample()
        return determ_state, next_repr, reward, discount_factors

    def get_latent(self, obs: torch.Tensor, action, state):
        embed = self.encoder(obs)
        determ, _, latent_repr_logits = self.recurrent_model.forward(state, embed.unsqueeze(0),
                                                           action)
        return determ, latent_repr_logits

    def calculate_loss(self, obs: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                       discount: torch.Tensor, first: torch.Tensor):
        b, _, h, w = obs.shape  # s <- BxHxWx3

        embed = self.encoder(obs)
        embed_c = embed.reshape(b // self.cluster_size, self.cluster_size, -1)

        a_c = a.reshape(-1, self.cluster_size, self.actions_num)
        r_c = r.reshape(-1, self.cluster_size, 1)
        d_c = discount.reshape(-1, self.cluster_size, 1)
        first_c = first.reshape(-1, self.cluster_size, 1)

        h_prev = None
        losses = defaultdict(lambda: torch.zeros(1).to(next(self.parameters()).device))
        metrics = defaultdict(lambda: torch.zeros(1).to(next(self.parameters()).device))

        def KL(dist1, dist2, free_nat = True):
            KL_ = torch.distributions.kl_divergence
            one = self.kl_free_nats * torch.ones(1, device=next(self.parameters()).device)
            # TODO: kl_free_avg is used always
            if free_nat:
                kl_lhs = torch.maximum(KL_(Dist(dist2.detach()), Dist(dist1)).mean(), one)
                kl_rhs = torch.maximum(KL_(Dist(dist2), Dist(dist1.detach())).mean(), one)
            else:
                kl_lhs = KL_(Dist(dist2.detach()), Dist(dist1)).mean()
                kl_rhs = KL_(Dist(dist2), Dist(dist1.detach())).mean()
            return (self.kl_beta * (self.alpha * kl_lhs + (1 - self.alpha) * kl_rhs))

        latent_vars = []
        determ_vars = []
        prior_logits = []
        posterior_logits = []

        # inps = []
        # reconstructed = []

        for t in range(self.cluster_size):
            # s_t <- 1xB^xHxWx3
            embed_t, a_t, first_t = embed_c[:, t].unsqueeze(0), a_c[:, t].unsqueeze(0), first_c[:, t].unsqueeze(0)
            a_t = a_t * (1 - first_t)

            determ_t, prior_stoch_logits, posterior_stoch_logits = self.recurrent_model.forward(
                h_prev, embed_t, a_t)
            posterior_stoch = Dist(posterior_stoch_logits).rsample().reshape(
                -1, self.latent_dim * self.latent_classes)

            h_prev = [determ_t, posterior_stoch.unsqueeze(0)]
            determ_vars.append(determ_t.squeeze(0))
            latent_vars.append(posterior_stoch)

            prior_logits.append(prior_stoch_logits)
            posterior_logits.append(posterior_stoch_logits)

        # inp = torch.concat([determ_vars.squeeze(0), posterior_stoch], dim=1)
        inp = torch.concat([torch.stack(determ_vars, dim=1), torch.stack(latent_vars, dim=1)], dim=-1)
        r_pred = self.reward_predictor(inp)
        f_pred = self.discount_predictor(inp)
        x_r = self.image_predictor(torch.flatten(inp, 0, 1))
        prior_logits = torch.flatten(torch.stack(prior_logits, dim=1), 0, 1)
        posterior_logits = torch.flatten(torch.stack(posterior_logits, dim=1), 0, 1)

        losses['loss_reconstruction'] = -x_r.log_prob(obs).mean()
        losses['loss_reward_pred'] = -r_pred.log_prob(r_c).mean()
        losses['loss_discount_pred'] = -f_pred.log_prob(d_c).mean()
        losses['loss_kl_reg'] = KL(prior_logits, posterior_logits)

        metrics['reward_sae'] = (torch.abs(r_pred.mode - r_c)).mean()
        metrics['prior_entropy'] = Dist(prior_logits).entropy().mean()
        metrics['posterior_entropy'] = Dist(posterior_logits).entropy().mean()

        return losses, inp.flatten(0, 1).detach(), metrics


class ImaginativeCritic(nn.Module):

    def __init__(self, discount_factor: float, update_interval: int,
                 soft_update_fraction: float, value_target_lambda: float, latent_dim: int):
        super().__init__()
        self.gamma = discount_factor
        self.critic_update_interval = update_interval
        self.lambda_ = value_target_lambda
        self.critic_soft_update_fraction = soft_update_fraction
        self._update_num = 0

        self.critic = fc_nn_generator(latent_dim,
                                      1,
                                      400,
                                      4,
                                      intermediate_activation=nn.ELU,
                                      final_activation=DistLayer('mse'))
        self.target_critic = fc_nn_generator(latent_dim,
                                             1,
                                             400,
                                             4,
                                             intermediate_activation=nn.ELU,
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
        for i in range(rs.shape[0] - 2, -1, -1):
            v_lambda = rs[i] + ds[i] * (
                (1 - self.lambda_) * vs[i+1] +
                self.lambda_ * v_lambdas[-1])
            v_lambdas.append(v_lambda)

        return torch.stack(list(reversed(v_lambdas)))

    def lambda_return(self, zs, rs, ds):
        vs = self.target_critic(zs).mode
        return self._lambda_return(vs, rs, ds)


class DreamerV2(RlAgent):

    def __init__(
            self,
            obs_space_num: int,  # NOTE: encoder/decoder will work only with 64x64 currently
            actions_num: int,
            action_type: str,
            batch_cluster_size: int,
            latent_dim: int,
            latent_classes: int,
            rssm_dim: int,
            discount_factor: float,
            kl_loss_scale: float,
            kl_loss_balancing: float,
            kl_loss_free_nats: float,
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

        self.device = device_type
        self.imagination_horizon = imagination_horizon
        self.cluster_size = batch_cluster_size
        self.actions_num = actions_num
        self.rho = actor_reinforce_fraction
        # if actor_reinforce_fraction != 0:
        #     raise NotImplementedError("Reinforce part is not implemented")
        self.eta = actor_entropy_scale

        self.world_model = WorldModel(batch_cluster_size, latent_dim, latent_classes,
                                      rssm_dim, actions_num, kl_loss_scale,
                                      kl_loss_balancing, kl_loss_free_nats).to(device_type)

        self.actor = fc_nn_generator(rssm_dim + latent_dim * latent_classes,
                                     actions_num * 2 if action_type == 'continuous' else actions_num,
                                     400,
                                     4,
                                     intermediate_activation=nn.ELU,
                                     final_activation=DistLayer('normal_trunc' if action_type == 'continuous' else 'onehot')).to(device_type)

        self.critic = ImaginativeCritic(discount_factor, critic_update_interval,
                                        critic_soft_update_fraction,
                                        critic_value_target_lambda,
                                        rssm_dim + latent_dim * latent_classes).to(device_type)

        self.scaler = torch.cuda.amp.GradScaler()
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
            self, z_0, precomp_actions: t.Optional[list[Action]] = None, horizon: t.Optional[int] = None
    ) -> tuple[torch.Tensor, torch.distributions.Distribution, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        if horizon is None:
            horizon = self.imagination_horizon
        world_state = torch.zeros(1, z_0.shape[0], 200, device=z_0.device)
        zs, actions_dists, actions, next_zs, rewards, ts, determs = [], [], [], [], [], [], []
        z = z_0
        for i in range(horizon):
            # FIXME: if somebody sees it, you have no credibility as programmer
            if z.shape[1] == 1224:
                z, world_state = z[:, :1024], z[:, 1024:].unsqueeze(0)
            if precomp_actions is not None:
                a_dist = None
                a = precomp_actions[i].unsqueeze(0)
            else:
                a_dist = self.actor(torch.cat([world_state.squeeze(), z], dim=-1).detach())
                a = a_dist.rsample()
            world_state, next_z, reward, discount = self.world_model.predict_next(
                z, a, world_state)

            zs.append(z)
            actions_dists.append(a_dist)
            next_zs.append(next_z)
            rewards.append(reward)
            ts.append(discount)
            determs.append(world_state[0])
            actions.append(a)

            z = next_z
        return (torch.stack(zs), actions_dists, torch.stack(next_zs),
                torch.stack(rewards), torch.stack(ts), torch.stack(determs), torch.stack(actions))

    def reset(self):
        self._state = None
        # FIXME: instead of zero, it should be mode of distribution
        self._last_action = torch.zeros((1, 1, self.actions_num), device=self.device)
        self._latent_probs = torch.zeros((32, 32), device=self.device)
        self._action_probs = torch.zeros((self.actions_num), device=self.device)
        self._stored_steps = 0

    @staticmethod
    def preprocess_obs(obs: torch.Tensor):
        # FIXME: move to dataloader in replay buffer
        order = list(range(len(obs.shape)))
        # Swap channel from last to 3 from last
        order = order[:-3] + [order[-1]] + order[-3:-1]
        return ((obs.type(torch.float32) / 255.0) - 0.5).permute(order)
        # return obs.type(torch.float32).permute(order)

    def get_action(self, obs: Observation) -> Action:
        # NOTE: pytorch fails without .copy() only when get_action is called
        obs = torch.from_numpy(obs.copy()).to(self.device)
        obs = self.preprocess_obs(obs).unsqueeze(0)

        determ, latent_repr_logits = self.world_model.get_latent(obs, self._last_action,
                                                               self._state)
        latent_repr_dist = Dist(latent_repr_logits)
        self._state = (determ, latent_repr_dist.rsample().reshape(-1,
                                                                  32 * 32).unsqueeze(0))

        actor_dist = self.actor(torch.cat(self._state, dim=-1))

        if False:
            self._action_probs += actor_dist.base_dist.probs.squeeze()
        self._latent_probs += latent_repr_dist.base_dist.probs.squeeze()
        self._stored_steps += 1

        if False:
            return np.array([self._last_action.squeeze().detach().cpu().numpy().argmax()])
        else:
            return self._last_action.squeeze().detach().cpu().numpy()

    def _generate_video(self, obs: Observation, actions: list[Action]):
        obs = torch.from_numpy(obs.copy()).to(self.device)
        obs = self.preprocess_obs(obs).unsqueeze(0)

        if False:
            actions = F.one_hot(self.from_np(actions).to(torch.int64),
                               num_classes=self.actions_num).squeeze()
        else:
            actions = self.from_np(actions)
        z_0 = Dist(self.world_model.get_latent(obs, actions[0].unsqueeze(0).unsqueeze(0), None)[1]).rsample().reshape(-1, 32 * 32).unsqueeze(0)
        zs, _, _, rews, _, determs, _ = self.imagine_trajectory(z_0.squeeze(0), actions[1:], horizon=self.imagination_horizon - 1)
        # video_r = self.world_model.image_predictor(torch.concat([determs, zs], dim=2)).rsample().cpu().detach().numpy()
        video_r = self.world_model.image_predictor(torch.concat([torch.concat([torch.zeros_like(determs[0]).unsqueeze(0), determs]), torch.concat([z_0, zs])], dim=-1)).mode.cpu().detach().numpy()
        video_r = ((video_r + 0.5) * 255.0).astype(np.uint8)
        return video_r, rews.sum()

    def _generate_video_with_update(self, obs: list[Observation], init_action: list[Action]):
        obs = torch.from_numpy(obs.copy()).to(self.device)
        obs = self.preprocess_obs(obs)

        if False:
            action = F.one_hot(self.from_np(init_action).to(torch.int64),
                               num_classes=self.actions_num).squeeze()
        else:
            action = self.from_np(init_action)
        state = None
        video = []
        rews = []
        for o, a in zip(obs, action):
            determ, stoch_logits = self.world_model.get_latent(o.unsqueeze(0), a.unsqueeze(0).unsqueeze(0), state)
            z_0 = Dist(stoch_logits).rsample().reshape(-1, 32 * 32).unsqueeze(0)
            state = (determ, z_0)
            inp = torch.concat([determ, z_0], dim=-1)
            video_r = self.world_model.image_predictor(inp).mode.cpu().detach().numpy()
            rews.append(self.world_model.reward_predictor(inp).mode.item())
            video_r = ((video_r + 0.5) * 255.0).astype(np.uint8)
            video.append(video_r)
        return np.concatenate(video), sum(rews)

    def viz_log(self, rollout, logger, epoch_num):
        init_indeces = np.random.choice(len(rollout.states) - self.imagination_horizon, 3)
        real_rewards = [rollout.rewards[idx:idx+ self.imagination_horizon].sum() for idx in init_indeces]
        videos, imagined_rewards = zip(*[self._generate_video(obs_0.copy(), a_0) for obs_0, a_0 in zip(
                rollout.next_states[init_indeces], [rollout.actions[idx:idx+ self.imagination_horizon] for idx in init_indeces])])
        videos_r = np.concatenate(videos, axis=3)

        videos_update, imagined_update_rewards = zip(*[self._generate_video_with_update(obs_0.copy(), a_0) for obs_0, a_0 in zip(
                [rollout.next_states[idx:idx+ self.imagination_horizon] for idx in init_indeces],
                [rollout.actions[idx:idx+ self.imagination_horizon] for idx in init_indeces])
        ])
        videos_r_update = np.concatenate(videos_update, axis=3)

        videos = np.concatenate([
            rollout.next_states[init_idx:init_idx + self.imagination_horizon].transpose(
                0, 3, 1, 2) for init_idx in init_indeces
        ],
                                axis=3)
        videos_comparison = np.expand_dims(np.concatenate([videos, videos_r_update, videos_r], axis=2), 0)
        latent_hist = (self._latent_probs / self._stored_steps).detach().cpu().numpy()
        latent_hist = ((latent_hist / latent_hist.max() * 255.0 )).astype(np.uint8)

        # if discrete action space
        if False:
            action_hist = (self._action_probs / self._stored_steps).detach().cpu().numpy()
            fig = plt.Figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.bar(np.arange(self.actions_num), action_hist)
            logger.add_figure('val/action_probs', fig, epoch_num)
        else:
            # log mean +- std
            pass
        logger.add_image('val/latent_probs', latent_hist, epoch_num, dataformats='HW')
        logger.add_image('val/latent_probs_sorted', np.sort(latent_hist, axis=1), epoch_num, dataformats='HW')
        logger.add_video('val/dreamed_rollout', videos_comparison, epoch_num)

        rewards_err = torch.Tensor([torch.abs(imagined_rewards[i] - real_rewards[i]) for i in range(len(imagined_rewards))]).mean()
        rewards_update_err = np.mean([np.abs(imagined_update_rewards[i] - real_rewards[i]) for i in range(len(imagined_rewards))])
        logger.add_scalar('val/img_reward_err', rewards_err.item(), epoch_num)
        logger.add_scalar('val/img_update_reward_err', rewards_update_err.item(), epoch_num)

        logger.add_scalar(f'val/reward', real_rewards[0], epoch_num)

    def from_np(self, arr: np.ndarray):
        arr = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
        return arr.to(self.device, non_blocking=True)

    def train(self, obs: Observations, a: Actions, r: Rewards, next_obs: Observations,
              is_finished: TerminationFlags, is_first: IsFirstFlags):

        obs = self.preprocess_obs(self.from_np(obs))
        a = self.from_np(a).to(torch.int64)
        if False:
            a = F.one_hot(a, num_classes=self.actions_num).squeeze()
        r = self.from_np(r)
        next_obs = self.preprocess_obs(self.from_np(next_obs))
        discount_factors = (1 - self.from_np(is_finished).type(torch.float32))
        first_flags = self.from_np(is_first).type(torch.float32)

        number_of_zero_discounts = (1 - discount_factors).sum()
        if number_of_zero_discounts > 0:
            pass

        # take some latent embeddings as initial
        with torch.cuda.amp.autocast(enabled=True):
            losses, discovered_latents, wm_metrics = self.world_model.calculate_loss(
                    obs, a, r, discount_factors, first_flags)

            # NOTE: 'aten::nonzero' inside KL divergence is not currently supported on M1 Pro MPS device
            # world_model_loss = torch.Tensor(1).to(self.device)
            world_model_loss = (losses['loss_reconstruction'] +
                                losses['loss_reward_pred'] +
                                losses['loss_kl_reg'] +
                                losses['loss_discount_pred'])
        # for l in losses.values():
        #     world_model_loss += l

        self.world_model_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(world_model_loss).backward()

        # FIXME: clip gradient should be parametrized
        self.scaler.unscale_(self.world_model_optimizer)
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 100)
        self.scaler.step(self.world_model_optimizer)

        metrics = defaultdict(
            lambda: torch.zeros(1).to(self.device))
        metrics |= wm_metrics

        with torch.cuda.amp.autocast(enabled=True):
            # idx = torch.randperm(discovered_latents.size(0))
            # initial_states = discovered_latents[idx]
            # Dreamer does not shuffle
            initial_states = discovered_latents

            losses_ac = defaultdict(
                lambda: torch.zeros(1).to(self.device))

            zs, _, next_zs, rewards, discount_factors, determs, actions = self.imagine_trajectory(
                initial_states)
            zs = torch.cat([determs.squeeze(), zs], dim=-1)
            rewards = self.world_model.reward_normalizer(rewards)

            # Discount prediction is disabled for dmc vision in Dreamer
            # as trajectory will not abruptly stop
            discount_factors = self.critic.gamma * torch.ones_like(rewards)

            # Discounted factors should be shifted as they predict whether next state is terminal
            # First discount factor on contrary is always 1 as it cannot lead to trajectory finish
            discount_factors = torch.cat([torch.ones_like(discount_factors[:1]), discount_factors[:-2]], dim=0).detach()

            vs = self.critic.lambda_return(zs, rewards[:-1], discount_factors)

            # Ignore all factors after first is_finished state
            discount_factors = torch.cumprod(discount_factors, dim=0)

            predicted_vs_dist = self.critic.estimate_value(zs[:-1].detach())
            losses_ac['loss_critic'] = -(predicted_vs_dist.log_prob(vs.detach())).mean()

            metrics['critic/avg_target_value'] = self.critic.target_critic(zs[1:]).mode.mean()
            metrics['critic/avg_lambda_value'] = vs.mean()
            metrics['critic/avg_predicted_value'] = predicted_vs_dist.mode.mean()

            # last action should be ignored as it is not used to predict next state, thus no feedback
            # first value should be ignored as it is comes from replay buffer
            action_dists = self.actor(zs[1:-1].detach())
            baseline = self.critic.target_critic(zs[1:-1]).mode
            advantage = (vs[1:] - baseline).detach()
            losses_ac['loss_actor_reinforce'] += 0# -(self.rho * action_dists.base_dist.log_prob(actions[1:-1].detach()).unsqueeze(2) * discount_factors[:-2] * advantage).mean()
            losses_ac['loss_actor_dynamics_backprop'] = -((1 - self.rho) * (vs[1:]*discount_factors[:-1])).mean()

            def calculate_entropy(dist):
                return dist.entropy().unsqueeze(2)
                # return dist.base_dist.base_dist.entropy().unsqueeze(2)

            losses_ac['loss_actor_entropy'] += -(self.eta * calculate_entropy(action_dists)*discount_factors[:-1]).mean()
            losses_ac['loss_actor'] = losses_ac['loss_actor_reinforce'] + losses_ac['loss_actor_dynamics_backprop'] + losses_ac['loss_actor_entropy']

            # mean and std are estimated statistically as tanh transformation is used
            sample = action_dists.rsample((128,))
            act_avg = sample.mean(0)
            metrics['actor/avg_val'] = act_avg.mean()
            # metrics['actor/mode_val'] = action_dists.mode.mean()
            metrics['actor/mean_val'] = action_dists.mean.mean()
            metrics['actor/avg_sd'] = (((sample - act_avg)**2).mean(0).sqrt()).mean()
            metrics['actor/min_val'] = sample.min()
            metrics['actor/max_val'] = sample.max()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)

        self.scaler.scale(losses_ac['loss_critic']).backward()
        self.scaler.scale(losses_ac['loss_actor']).backward()

        self.scaler.unscale_(self.actor_optimizer)
        self.scaler.unscale_(self.critic_optimizer)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100)

        self.scaler.step(self.actor_optimizer)
        self.scaler.step(self.critic_optimizer)

        self.critic.update_target()
        self.scaler.update()

        losses = {l: val.detach().cpu().item() for l, val in losses.items()}
        losses_ac = {l: val.detach().cpu().item() for l, val in losses_ac.items()}
        metrics = {l: val.detach().cpu().item() for l, val in metrics.items()}

        return losses | losses_ac | metrics

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
