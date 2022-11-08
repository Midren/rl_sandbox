import itertools
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.fc_nn import fc_nn_generator
from rl_sandbox.utils.replay_buffer import (Action, Actions, Observations,
                                            Rewards, State, States,
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
        return dist.rsample()


class RSSM(nn.Module):
    """
    Recurrent State Space Model

    h_t    <- deterministic state which is updated inside GRU
    s^_t   <- stohastic discrete prior state (used for KL divergence:
                                               better predict future and encode smarter)
    s_t    <- stohastic discrete posterior state (latent representation of current state)

      h_1            --->     h_2            --->     h_3            --->
          \    x_1                \    x_2                \    x_3
       |   \    |     ^        |   \    |     ^        |   \    |     ^
       v   MLP CNN    |        v   MLP CNN    |        v   MLP CNN    |
             \  |     |              \  |     |              \  |     |
    Ensemble  \ |     |     Ensemble  \ |     |     Ensemble  \ |     |
               \|     |                \|     |                \|     |
       |        |     |        |        |     |        |        |     |
       v        v     |        v        v     |        v        v     |
                      |                       |                       |
      s^_1     s_1 ---|       s^_2     s_2 ---|       s^_3     s_3 ---|

    """

    def __init__(self, latent_dim, hidden_size, actions_num, categories_num):
        super().__init__()
        self.latent_dim = latent_dim
        self.categories_num = categories_num
        self.ensemble_num = 5

        # Calculate deterministic state from prev stochastic, prev action and prev deterministic
        self.pre_determ_recurrent = nn.Sequential(
            nn.Linear(latent_dim * categories_num + actions_num, hidden_size),  # Dreamer 'img_in'
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
                          latent_dim * self.categories_num),  # Dreamer 'img_dist_{k}'
                View((-1, latent_dim, self.categories_num)))
            for _ in range(self.ensemble_num)
        ])

        # For observation we do not have ensemble
        # FIXME: very band magic number
        img_sz = 4 * 384  # 384*2x2
        self.stoch_net = nn.Sequential(
            nn.Linear(hidden_size + img_sz, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),  # Dreamer 'obs_out'
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size,
                      latent_dim * self.categories_num),  # Dreamer 'obs_dist'
            View((-1, latent_dim, self.categories_num)),
            # NOTE: Maybe worth having some LogSoftMax as activation
            #       before using input as logits for distribution
            # Quantize()
        )

    def estimate_stochastic_latent(self, prev_determ):
        logits_per_model = torch.stack(
            [model(prev_determ) for model in self.ensemble_prior_estimator])
        # NOTE: Maybe something smarter can be used instead of
        #       taking only one random between all ensembles
        index = torch.randint(0, self.ensemble_num, ())
        return torch.distributions.one_hot_categorical.OneHotCategoricalStraightThrough(
            logits=logits_per_model[index])

    def forward(self, h_prev: tuple[torch.Tensor, torch.Tensor], embed, action):
        """
            'h' <- internal state of the world
            'z' <- latent embedding of current observation
            'a' <- action taken on prev step
            Returns 'h_next' <- the next next of the world
        """

        # Use zero vector for prev_state of first
        deter_prev, stoch_prev = h_prev
        x = self.pre_determ_recurrent(torch.concat([stoch_prev, action], dim=2))
        _, determ = self.determ_recurrent(x, deter_prev)

        # used for KL divergence
        prior_stoch_dist = self.estimate_stochastic_latent(determ)

        posterior_stoch_logits = self.stoch_net(torch.concat([determ, embed],
                                                             dim=2))  # Dreamer 'obs_out'
        posterior_stoch_dist = torch.distributions.one_hot_categorical.OneHotCategoricalStraightThrough(
            logits=posterior_stoch_logits)

        return [determ, prior_stoch_dist, posterior_stoch_dist]


# NOTE: In Dreamer ELU is used everywhere as activation func
# NOTE: In Dreamer 48**(lvl) filter size is used, 4 level of convolution,
#       Layer Normalizatin instead of Batch
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
        self.convin = nn.Linear(32*32, 32*self.channel_step)

        in_channels = 32*self.channel_step #2**(len(kernel_sizes) - 1) * self.channel_step
        for i, k in enumerate(kernel_sizes):
            out_channels = 2**(len(kernel_sizes) - i - 2) * self.channel_step
            if i == len(kernel_sizes) - 1:
                out_channels = 3
                layers.append(
                    nn.ConvTranspose2d(in_channels, 3, kernel_size=k, stride=2))
            else:
                layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=2))
                layers.append(nn.ELU(inplace=True))
                layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        x = self.convin(X)
        x = x.view(-1, 32*self.channel_step, 1, 1)
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
                                    categories_num=latent_classes)
        self.encoder = Encoder()
        from torchsummary import summary

        # summary(self.encoder, input_size=(3, 64, 64))
        self.image_predictor = Decoder()
        # FIXME: in Dreamer paper it is 4 hidden layers with 400 hidden units
        # FIXME: in Dramer paper it has Layer Normalization after Dense
        self.reward_predictor = fc_nn_generator(rssm_dim + latent_dim * latent_classes,
                                                1,
                                                hidden_size=128,
                                                num_layers=3)
        self.discount_predictor = fc_nn_generator(rssm_dim + latent_dim * latent_classes,
                                                  1,
                                                  hidden_size=128,
                                                  num_layers=3,
                                                  final_activation=nn.Sigmoid)

        self.optimizer = torch.optim.Adam(itertools.chain(
            self.recurrent_model.parameters(), self.encoder.parameters(),
            self.image_predictor.parameters(), self.reward_predictor.parameters(),
            self.discount_predictor.parameters()),
                                          lr=2e-4)

    def forward(self, X):
        pass

    def train(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
              is_finished: torch.Tensor):
        b, h, w, _ = s.shape  # s <- BxHxWx3

        s = ((s.type(torch.float32) / 255.0) - 0.5).permute(0, 3, 1, 2)
        embed = self.encoder(s)
        embed = embed.view(b // self.cluster_size, self.cluster_size, -1)

        s = s.view(-1, self.cluster_size, 3, h, w)
        a = a.view(-1, self.cluster_size, a.shape[1])
        r = r.view(-1, self.cluster_size, 1)
        f = is_finished.view(-1, self.cluster_size, 1)

        h_prev = [
            torch.zeros((1, b // self.cluster_size, self.rssm_dim)),
            torch.zeros((1, b // self.cluster_size, self.latent_dim*self.latent_classes))
        ]
        losses = defaultdict(lambda: torch.zeros(1))

        def KL(dist1, dist2):
            KL_ = torch.distributions.kl_divergence
            Dist = torch.distributions.OneHotCategoricalStraightThrough
            return self.kl_beta *(self.alpha * KL_(dist1, Dist(logits=dist2.logits.detach())).mean() +
            (1 - self.alpha) * KL_(Dist(logits=dist1.logits.detach()), dist2).mean())

        for t in range(self.cluster_size):
            # s_t <- 1xB^xHxWx3
            x_t, embed_t, a_t, r_t, f_t = s[:, t], embed[:, t].unsqueeze(
                0), a[:, t].unsqueeze(0), r[:, t], f[:, t]

            determ_t, prior_stoch_dist, posterior_stoch_dist = self.recurrent_model(
                h_prev, embed_t, a_t)
            posterior_stoch = posterior_stoch_dist.rsample().reshape(-1, self.latent_dim*self.latent_classes)

            r_t_pred = self.reward_predictor(torch.concat([determ_t.squeeze(0), posterior_stoch], dim=1))
            f_t_pred = self.discount_predictor(torch.concat([determ_t.squeeze(0), posterior_stoch], dim=1))

            x_r = self.image_predictor(posterior_stoch)

            losses['loss_reconstruction'] = nn.functional.mse_loss(x_t, x_r)
            losses['loss_reward_pred'] += F.mse_loss(r_t, r_t_pred)
            losses['loss_discount_pred'] += F.cross_entropy(f_t.type(torch.float32), f_t_pred)
            losses['loss_kl_reg'] += KL(prior_stoch_dist, posterior_stoch_dist)

            h_prev = [determ_t, posterior_stoch.unsqueeze(0)]

        loss = torch.Tensor(1)
        for l in losses.values():
            loss += l

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {l: val.detach() for l, val in losses.items()}


class DreamerV2(RlAgent):

    def __init__(self,
                 obs_space_num: int,
                 actions_num: int,
                 batch_cluster_size: int,
                 latent_dim: int,
                 latent_classes: int,
                 rssm_dim: int,
                 discount_factor: float,
                 kl_loss_scale: float,
                 device_type: str = 'cpu'):

        self.cluster_size = batch_cluster_size
        self.actions_num = actions_num
        self.gamma = discount_factor

        self.world_model = WorldModel(batch_cluster_size, latent_dim, latent_classes,
                                      rssm_dim, actions_num,
                                      kl_loss_scale).to(device_type)

    def get_action(self, obs: State) -> Action:
        return self.actions_num

    def from_np(self, arr: np.ndarray):
        return torch.from_numpy(arr).to(next(self.world_model.parameters()).device)

    def train(self, s: Observations, a: Actions, r: Rewards, next: States,
              is_finished: TerminationFlags):
        # NOTE: next is currently incorrect (state instead of img), but also unused
        # FIXME: next should be correct, as World model is trained on triplets
        #        (h_prev, action, next_state)

        s = self.from_np(s)
        a = self.from_np(a)
        r = self.from_np(r)
        next = self.from_np(next)
        is_finished = self.from_np(is_finished)

        return self.world_model.train(s, a, r, is_finished)
