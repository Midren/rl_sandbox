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
from rl_sandbox.vision.vq_vae import VQ_VAE


class RSSM(nn.Module):
    """
    Recurrent State Space Model
    """

    def __init__(self, latent_dim, hidden_size, actions_num):
        super().__init__()

        self.gru = nn.GRU(input_size=latent_dim + actions_num, hidden_size=hidden_size)

    def forward(self, h_prev, z, a):
        """
            'h' <- internal state of the world
            'z' <- latent embedding of current observation
            'a' <- action taken on current step
            Returns 'h_next' <- the next next of the world
        """

        _, h_n = self.gru(torch.concat([z, a]), h_prev)
        # NOTE: except deterministic step h_t, model should also return stochastic state concatenated
        # NOTE: to add stoshasticity for internal state, ensemble of MLP's is used
        return h_n


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
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=2, padding=1))
            layers.append(nn.ELU(inplace=True))
            layers.append(nn.LayerNorm(out_channels))
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)

class Decoder(nn.Module):

    def __init__(self, kernel_sizes=[4, 4, 4, 4]):
        super().__init__()
        layers = []

        channel_step = 48
        in_channels = 2**(len(kernel_sizes)-1) *channel_step
        for i, k in enumerate(kernel_sizes):
            out_channels = 2**(len(kernel_sizes) - i - 2) * channel_step
            if out_channels == channel_step:
                out_channels = 3
            layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=k,
                                             stride=2,
                                             padding=1
                ))
            layers.append(nn.ELU(inplace=True))
            layers.append(nn.LayerNorm(out_channels))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class WorldModel(nn.Module):

    def __init__(self, batch_cluster_size, latent_dim, latent_classes, rssm_dim,
                 actions_num, kl_loss_scale):
        self.kl_beta = kl_loss_scale
        self.rssm_dim = rssm_dim
        self.cluster_size = batch_cluster_size
        self.recurrent_model = RSSM(latent_dim, rssm_dim, actions_num)
        # NOTE: In Dreamer paper VQ-VAE has MLP after conv2d to get 1d embedding,
        #       which is concatenated with deterministic state and only after that
        #       sampled into discrete one-hot encoding (using TensorFlow.Distribution OneHotCategorical)
        # self.representation_network = VQ_VAE(
        #     latent_dim=latent_dim,
        #     latent_space_size=latent_classes)  # actually only 'encoder' part of VAE
        self.encoder = Encoder()
        self.image_predictor = Decoder()
        # self.image_predictor = 'decoder' part of VAE
        # FIXME: will not work until VQ-VAE internal embedding will not be changed from 2d to 1d
        # FIXME: in Dreamer paper it is 4 hidden layers with 400 hidden units
        # FIXME: in Dramer paper it has Layer Normalization after Dense
        self.transition_network = fc_nn_generator(rssm_dim,
                                                  latent_dim,
                                                  hidden_size=128,
                                                  num_layers=3)
        self.reward_predictor = fc_nn_generator(rssm_dim + latent_dim,
                                                1,
                                                hidden_size=128,
                                                num_layers=3)
        self.discount_predictor = fc_nn_generator(rssm_dim + latent_dim,
                                                  1,
                                                  hidden_size=128,
                                                  num_layers=3)

        self.optimizer = torch.optim.Adam(self.representation_network.parameters(),
                                          lr=2e-4)

    def forward(self, X):
        pass

    def train(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
              is_finished: torch.Tensor):
        b, h, w, _ = s.shape  # s <- BxHxWx3
        s = s.view(-1, self.cluster_size, h, w, 3)
        a = a.view(-1, self.cluster_size, a.shape[1])
        r = r.view(-1, self.cluster_size, 1)
        f = is_finished.view(-1, self.cluster_size, 1)

        h_prev = torch.zeros((b, self.rssm_dim))
        losses = defaultdict(lambda: torch.zeros(1))

        embed = self.encoder(s)
        for t in range(self.cluster_size):
            # s_t <- 1xB^xHxWx3
            embed_t, a_t, r_t, f_t = embed[:, t].unsqueeze(0), a[:, t].unsqueeze(
                0), r[:, t].unsqueeze(0), f[:, t].unsqueeze(0)

            # TODO: add in the future h_t into representation network
            # NOTE: can be moved out of the loop, *embed* is calculated solely by image
            # s_t_r, z_t, e_t = self.representation_network(s_t)
            h_t = self.recurrent_model(h_prev, z_t, a_t)

            r_t_pred = self.reward_predictor(torch.concat([h_t, z_t]))
            f_t_pred = self.discount_predictor(torch.concat([h_t, z_t]))
            z_t_prior = self.transition_network(h_t)

            vae_losses = self.representation_network.calculate_loss(s_t, s_t_r, z_t, e_t)
            # NOTE: regularization loss from VQ-VAE is not used in Dreamer paper
            losses['loss_reconstruction'] = vae_losses['loss_rec']
            losses['loss_reward_pred'] += F.mse_loss(r_t, r_t_pred)
            losses['loss_discount_pred'] += F.cross_entropy(f_t, f_t_pred)
            # TODO: add KL divergence loss between transition predictor and representation model
            # NOTE: remember about different learning rate for prior and posterior
            # NOTE: VQ-VAE should be changed to output the softmax of how close z is to each e,
            # so it can be used to count as probability for each distribution to calculate
            # the KL divergence
            # NOTE: DreamerV2 uses TensorFlow.Probability to calculate KL divergence
            losses['loss_kl_reg'] += self.kl_beta * 0

            h_prev = h_t

        loss = torch.Tensor(0)
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
        return torch.from_numpy(arr).to(self.device_type)

    def train(self, s: Observations, a: Actions, r: Rewards, next: States,
              is_finished: TerminationFlags):
        # NOTE: next is currently incorrect (state instead of img), but also unused

        s = self.from_np(s)
        a = self.from_np(a)
        r = self.from_np(r)
        next = self.from_np(next)
        is_finished = self.from_np(is_finished)

        self.world_model.train(s, a, r, is_finished)
