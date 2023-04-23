import typing as t
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
from torch import nn
from torch.nn import functional as F
import torchvision as tv
from jaxtyping import Float, Bool
from rl_sandbox.vision.dino import ViTFeat

from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.fc_nn import fc_nn_generator
from rl_sandbox.utils.replay_buffer import (Action, Actions, Observation,
                                            Observations, Rewards,
                                            TerminationFlags, IsFirstFlags)
from rl_sandbox.utils.schedulers import LinearScheduler
from rl_sandbox.utils.dists import DistLayer
from rl_sandbox.vision.slot_attention import SlotAttention, PositionalEmbedding

class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def Dist(val):
    return DistLayer('onehot')(val)

@dataclass
class State:
    determ: Float[torch.Tensor, 'seq batch num_slots determ']
    stoch_logits: Float[torch.Tensor, 'seq batch num_slots latent_classes latent_dim']
    stoch_: t.Optional[Bool[torch.Tensor, 'seq batch num_slots stoch_dim']] = None

    @property
    def combined(self):
        return torch.concat([self.determ, self.stoch], dim=-1).flatten(2, 3)

    @property
    def combined_slots(self):
        return torch.concat([self.determ, self.stoch], dim=-1)

    @property
    def stoch(self):
        if self.stoch_ is None:
            self.stoch_ = Dist(self.stoch_logits).rsample().reshape(self.stoch_logits.shape[:3] + (-1,))
        return self.stoch_

    @property
    def stoch_dist(self):
        return Dist(self.stoch_logits)

    @classmethod
    def stack(cls, states: list['State'], dim = 0):
        if states[0].stoch_ is not None:
            stochs = torch.cat([state.stoch for state in states], dim=dim)
        else:
            stochs = None
        return State(torch.cat([state.determ for state in states], dim=dim),
                     torch.cat([state.stoch_logits for state in states], dim=dim),
                     stochs)

class GRUCell(nn.Module):
  def __init__(self, input_size, hidden_size, norm=False, update_bias=-1, **kwargs):
    super().__init__()
    self._size = hidden_size
    self._act = torch.tanh
    self._norm = norm
    self._update_bias = update_bias
    self._layer = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=norm is not None, **kwargs)
    if norm:
      self._norm = nn.LayerNorm(3 * hidden_size)

  @property
  def state_size(self):
    return self._size

  def forward(self, x, h):
    state = h
    parts = self._layer(torch.concat([x, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = self._norm(parts.float())
      parts = parts.to(dtype=dtype)
    reset, cand, update = parts.chunk(3, dim=-1)
    reset = torch.sigmoid(reset)
    cand = self._act(reset * cand)
    update = torch.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, output


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.inp_in = nn.Linear(1024, self.n_embed*self.dim)
        self.inp_out = nn.Linear(self.n_embed*self.dim, 1024)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, inp):
        # input = self.inp_in(inp).reshape(-1, 1, self.n_embed, self.dim)
        input = inp.reshape(-1, 1, self.n_embed, self.dim)
        inp = input
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # quantize_out = self.inp_out(quantize.reshape(-1, self.n_embed*self.dim))
        quantize_out = quantize
        diff = 0.25*(quantize_out.detach() - inp).pow(2).mean() + (quantize_out - inp.detach()).pow(2).mean()
        quantize = inp + (quantize_out - inp).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


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

    def __init__(self, latent_dim, hidden_size, actions_num, latent_classes, discrete_rssm, norm_layer: nn.LayerNorm | nn.Identity):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_classes = latent_classes
        self.ensemble_num = 1
        self.hidden_size = hidden_size
        self.discrete_rssm = discrete_rssm

        # Calculate deterministic state from prev stochastic, prev action and prev deterministic
        self.pre_determ_recurrent = nn.Sequential(
            nn.Linear(latent_dim * latent_classes + actions_num,
                      hidden_size),  # Dreamer 'img_in'
            norm_layer(hidden_size),
            nn.ELU(inplace=True)
        )
        self.determ_recurrent = GRUCell(input_size=hidden_size, hidden_size=hidden_size, norm=True)  # Dreamer gru '_cell'

        # Calculate stochastic state from prior embed
        # shared between all ensemble models
        self.ensemble_prior_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),  # Dreamer 'img_out_{k}'
                norm_layer(hidden_size),
                nn.ELU(inplace=True),
                nn.Linear(hidden_size,
                          latent_dim * self.latent_classes),  # Dreamer 'img_dist_{k}'
                View((1, -1, latent_dim, self.latent_classes))) for _ in range(self.ensemble_num)
        ])

        # For observation we do not have ensemble
        # FIXME: very bad magic number
        # img_sz = 4 * 384  # 384x2x2
        img_sz = 192
        self.stoch_net = nn.Sequential(
            # nn.LayerNorm(hidden_size + img_sz, hidden_size),
            nn.Linear(hidden_size + img_sz, hidden_size), # Dreamer 'obs_out'
            norm_layer(hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size,
                      latent_dim * self.latent_classes),  # Dreamer 'obs_dist'
            View((1, -1, latent_dim, self.latent_classes)))
        # self.determ_discretizer = MlpVAE(self.hidden_size)
        self.determ_discretizer = Quantize(16, 16)
        self.discretizer_scheduler = LinearScheduler(1.0, 0.0, 1_000_000)
        self.determ_layer_norm = nn.LayerNorm(hidden_size)

    def estimate_stochastic_latent(self, prev_determ: torch.Tensor):
        dists_per_model = [model(prev_determ) for model in self.ensemble_prior_estimator]
        # NOTE: Maybe something smarter can be used instead of
        #       taking only one random between all ensembles
        # NOTE: in Dreamer ensemble_num is always 1
        idx = torch.randint(0, self.ensemble_num, ())
        return dists_per_model[0]

    def predict_next(self,
                     prev_state: State,
                     action) -> State:
        x = self.pre_determ_recurrent(torch.concat([prev_state.stoch, action.unsqueeze(2).repeat((1, 1, prev_state.determ.shape[2], 1))], dim=-1))
        # NOTE: x and determ are actually the same value if sequence of 1 is inserted
        x, determ_prior = self.determ_recurrent(x.flatten(1, 2), prev_state.determ.flatten(1, 2))
        if self.discrete_rssm:
            raise NotImplementedError("discrete rssm was not adopted for slot attention")
            # determ_post, diff, embed_ind = self.determ_discretizer(determ_prior)
            # determ_post = determ_post.reshape(determ_prior.shape)
            # determ_post = self.determ_layer_norm(determ_post)
            # alpha = self.discretizer_scheduler.val
            # determ_post = alpha * determ_prior + (1-alpha) * determ_post
        else:
            determ_post, diff = determ_prior, 0

        # used for KL divergence
        predicted_stoch_logits = self.estimate_stochastic_latent(x)
        # Size is 1 x B x slots_num x ...
        return State(determ_post.reshape(prev_state.determ.shape), predicted_stoch_logits.reshape(prev_state.stoch_logits.shape)), diff

    def update_current(self, prior: State, embed) -> State: # Dreamer 'obs_out'
        return State(prior.determ, self.stoch_net(torch.concat([prior.determ, embed], dim=-1)).flatten(1, 2).reshape(prior.stoch_logits.shape))

    def forward(self, h_prev: State, embed,
                action) -> tuple[State, State]:
        """
            'h' <- internal state of the world
            'z' <- latent embedding of current observation
            'a' <- action taken on prev step
            Returns 'h_next' <- the next next of the world
        """
        prior, diff = self.predict_next(h_prev, action)
        posterior = self.update_current(prior, embed)

        return prior, posterior, diff


class Encoder(nn.Module):

    def __init__(self, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[4, 4, 4]):
        super().__init__()
        layers = []

        channel_step = 48
        in_channels = 3
        for i, k in enumerate(kernel_sizes):
            out_channels = 2**i * channel_step
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=2))
            layers.append(norm_layer(1, out_channels))
            layers.append(nn.ELU(inplace=True))
            in_channels = out_channels
        # layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class Decoder(nn.Module):

    def __init__(self, input_size, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[5, 5, 6, 6]):
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
                layers.append(nn.ConvTranspose2d(in_channels, 4, kernel_size=k, stride=2))
            else:
                layers.append(norm_layer(1, in_channels))
                layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k,
                                       stride=2))
                layers.append(nn.ELU(inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        x = self.convin(X)
        x = x.view(-1, 32 * self.channel_step, 1, 1)
        return self.net(x)
        # return td.Independent(td.Normal(self.net(x), 1.0), 3)

class ViTDecoder(nn.Module):

    # def __init__(self, input_size, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[5, 5, 5, 3, 5, 3]):
    # def __init__(self, input_size, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[5, 5, 5, 5, 3]):
    def __init__(self, input_size, norm_layer: nn.GroupNorm | nn.Identity, kernel_sizes=[5, 5, 5, 3, 3]):
        super().__init__()
        layers = []
        self.channel_step = 12
        # 2**(len(kernel_sizes)-1)*channel_step
        self.convin = nn.Linear(input_size, 32 * self.channel_step)

        in_channels = 32 * self.channel_step  #2**(len(kernel_sizes) - 1) * self.channel_step
        for i, k in enumerate(kernel_sizes):
            out_channels = 2**(len(kernel_sizes) - i - 2) * self.channel_step
            if i == len(kernel_sizes) - 1:
                out_channels = 3
                layers.append(nn.ConvTranspose2d(in_channels, 384, kernel_size=k, stride=1, padding=1))
            else:
                layers.append(norm_layer(1, in_channels))
                layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=2, padding=2, output_padding=1))
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
        self.register_buffer('mag', torch.ones(1, dtype=torch.float32))
        self.mag.requires_grad = False

    def forward(self, x):
        self.update(x)
        return (x / (self.mag + self.eps))*self.scale

    def update(self, x):
        self.mag = self.momentum * self.mag  + (1 - self.momentum) * (x.abs().mean()).detach()


class WorldModel(nn.Module):

    def __init__(self, img_size, batch_cluster_size, latent_dim, latent_classes, rssm_dim,
                 actions_num, kl_loss_scale, kl_loss_balancing, kl_free_nats, discrete_rssm,
                 predict_discount, layer_norm: bool, encode_vit: bool, decode_vit: bool, vit_l2_ratio: float,
                 slots_num: int):
        super().__init__()
        self.register_buffer('kl_free_nats', kl_free_nats * torch.ones(1))
        self.kl_beta = kl_loss_scale
        self.rssm_dim = rssm_dim
        self.latent_dim = latent_dim
        self.latent_classes = latent_classes
        self.slots_num = slots_num
        self.cluster_size = batch_cluster_size
        self.actions_num = actions_num
        # kl loss balancing (prior/posterior)
        self.alpha = kl_loss_balancing
        self.predict_discount = predict_discount
        self.encode_vit = encode_vit
        self.decode_vit = decode_vit
        self.vit_l2_ratio = vit_l2_ratio

        self.recurrent_model = RSSM(latent_dim,
                                    rssm_dim,
                                    actions_num,
                                    latent_classes,
                                    discrete_rssm,
                                    norm_layer=nn.Identity if layer_norm else nn.LayerNorm)
        if encode_vit or decode_vit:
            # self.dino_vit = ViTFeat("/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth", feat_dim=768, vit_arch='base', patch_size=8)
            # self.dino_vit = ViTFeat("/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth", feat_dim=384, vit_arch='small', patch_size=8)
            self.dino_vit = ViTFeat("/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth", feat_dim=384, vit_arch='small', patch_size=16)
            self.vit_feat_dim = self.dino_vit.feat_dim
            self.vit_num_patches = self.dino_vit.model.patch_embed.num_patches
            self.dino_vit.requires_grad_(False)

        if encode_vit:
            self.encoder = nn.Sequential(
                self.dino_vit,
                nn.Flatten(),
                # fc_nn_generator(64*self.dino_vit.feat_dim,
                #                 64*384,
                #                 hidden_size=400,
                #                 num_layers=5,
                #                 intermediate_activation=nn.ELU,
                #                 layer_norm=layer_norm)
                )
        else:
            self.encoder = Encoder(norm_layer=nn.Identity if layer_norm else nn.GroupNorm)

        self.n_dim = 192
        self.slot_attention = SlotAttention(slots_num, self.n_dim, 5)
        self.positional_augmenter_inp = PositionalEmbedding(self.n_dim, (6, 6))
        # self.positional_augmenter_dec = PositionalEmbedding(self.n_dim, (8, 8))

        self.slot_mlp = nn.Sequential(
            nn.Linear(self.n_dim, self.n_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_dim, self.n_dim)
        )


        if decode_vit:
            self.dino_predictor = ViTDecoder(rssm_dim + latent_dim * latent_classes,
                                             norm_layer=nn.Identity if layer_norm else nn.GroupNorm)
            # self.dino_predictor = fc_nn_generator(rssm_dim + latent_dim*latent_classes,
            #                                        64*self.dino_vit.feat_dim,
            #                                        hidden_size=2048,
            #                                        num_layers=5,
            #                                        intermediate_activation=nn.ELU,
            #                                        layer_norm=layer_norm,
            #                                        final_activation=DistLayer('mse'))
        self.image_predictor = Decoder(rssm_dim + latent_dim * latent_classes,
                                       norm_layer=nn.Identity if layer_norm else nn.GroupNorm)

        self.reward_predictor = fc_nn_generator(slots_num*(rssm_dim + latent_dim * latent_classes),
                                                1,
                                                hidden_size=400,
                                                num_layers=5,
                                                intermediate_activation=nn.ELU,
                                                layer_norm=layer_norm,
                                                final_activation=DistLayer('mse'))
        self.discount_predictor = fc_nn_generator(slots_num*(rssm_dim + latent_dim * latent_classes),
                                                  1,
                                                  hidden_size=400,
                                                  num_layers=5,
                                                  intermediate_activation=nn.ELU,
                                                  layer_norm=layer_norm,
                                                  final_activation=DistLayer('binary'))
        self.reward_normalizer = Normalizer(momentum=1.00, scale=1.0, eps=1e-8)

    def get_initial_state(self, batch_size: int = 1, seq_size: int = 1):
        device = next(self.parameters()).device
        return State(torch.zeros(seq_size, batch_size, self.slots_num, self.rssm_dim, device=device),
                            torch.zeros(seq_size, batch_size, self.slots_num, self.latent_classes, self.latent_dim, device=device),
                            torch.zeros(seq_size, batch_size, self.slots_num, self.latent_classes * self.latent_dim, device=device))

    def predict_next(self, prev_state: State, action):
        prior, _ = self.recurrent_model.predict_next(prev_state, action)

        reward = self.reward_predictor(prior.combined).mode
        if self.predict_discount:
            discount_factors = self.discount_predictor(prior.combined).sample()
        else:
            discount_factors = torch.ones_like(reward)
        return prior, reward, discount_factors

    def get_latent(self, obs: torch.Tensor, action, state: t.Optional[State], prev_slots: t.Optional[torch.Tensor]) -> t.Tuple[State, torch.Tensor]:
        if state is None:
            state = self.get_initial_state()
        embed = self.encoder(obs.unsqueeze(0))
        embed_with_pos_enc = self.positional_augmenter_inp(embed)

        pre_slot_features_t = self.slot_mlp(embed_with_pos_enc.permute(0, 2, 3, 1).reshape(1, -1, self.n_dim))

        slots_t = self.slot_attention(pre_slot_features_t, prev_slots)

        _, posterior, _ = self.recurrent_model.forward(state, slots_t.unsqueeze(0), action)
        return posterior, slots_t

    def calculate_loss(self, obs: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                       discount: torch.Tensor, first: torch.Tensor):
        b, _, h, w = obs.shape  # s <- BxHxWx3

        embed = self.encoder(obs)
        embed_with_pos_enc = self.positional_augmenter_inp(embed)
        # embed_c = embed.reshape(b // self.cluster_size, self.cluster_size, -1)

        pre_slot_features = self.slot_mlp(embed_with_pos_enc.permute(0, 2, 3, 1).reshape(b, -1, self.n_dim))
        pre_slot_features_c = pre_slot_features.reshape(b // self.cluster_size, self.cluster_size, -1, self.n_dim)

        a_c = a.reshape(-1, self.cluster_size, self.actions_num)
        r_c = r.reshape(-1, self.cluster_size, 1)
        d_c = discount.reshape(-1, self.cluster_size, 1)
        first_c = first.reshape(-1, self.cluster_size, 1)

        losses = {}
        metrics = {}

        def KL(dist1, dist2):
            KL_ = torch.distributions.kl_divergence
            kl_lhs = KL_(td.OneHotCategoricalStraightThrough(logits=dist2.detach()), td.OneHotCategoricalStraightThrough(logits=dist1)).mean()
            kl_rhs = KL_(td.OneHotCategoricalStraightThrough(logits=dist2), td.OneHotCategoricalStraightThrough(logits=dist1.detach())).mean()
            kl_lhs = torch.maximum(kl_lhs, self.kl_free_nats)
            kl_rhs = torch.maximum(kl_rhs, self.kl_free_nats)
            return (self.kl_beta * (self.alpha * kl_lhs + (1 - self.alpha) * kl_rhs))

        priors = []
        posteriors = []

        if self.decode_vit:
            inp = obs
            if not self.encode_vit:
                ToTensor = tv.transforms.Compose([tv.transforms.Normalize((0.485, 0.456, 0.406),
                                                       (0.229, 0.224, 0.225)),
                                                  tv.transforms.Resize(224, antialias=True)])
                # ToTensor = tv.transforms.Normalize((0.485, 0.456, 0.406),
                #                                        (0.229, 0.224, 0.225))
                inp = ToTensor(obs + 0.5)
            d_features = self.dino_vit(inp)

        prev_state = self.get_initial_state(b // self.cluster_size)
        prev_slots = None
        for t in range(self.cluster_size):
            # s_t <- 1xB^xHxWx3
            pre_slot_feature_t, a_t, first_t = pre_slot_features_c[:, t], a_c[:, t].unsqueeze(0), first_c[:, t].unsqueeze(0)
            a_t = a_t * (1 - first_t)

            slots_t = self.slot_attention(pre_slot_feature_t, prev_slots)
            # prev_slots = None

            prior, posterior, diff = self.recurrent_model.forward(prev_state, slots_t.unsqueeze(0), a_t)
            prev_state = posterior

            priors.append(prior)
            posteriors.append(posterior)

            # losses['loss_determ_recons'] += diff

        posterior = State.stack(posteriors)
        prior = State.stack(priors)

        r_pred = self.reward_predictor(posterior.combined.transpose(0, 1))
        f_pred = self.discount_predictor(posterior.combined.transpose(0, 1))

        losses['loss_reconstruction_img'] = torch.Tensor([0]).to(obs.device)

        if not self.decode_vit:
            decoded_imgs, masks = self.image_predictor(posterior.combined_slots.transpose(0, 1).flatten(0, 1)).reshape(b, -1, 4, h, w).split([3, 1], dim=2)
            img_mask = F.softmax(masks, dim=1)
            decoded_imgs = decoded_imgs * img_mask
            x_r = td.Independent(td.Normal(torch.sum(decoded_imgs, dim=1), 1.0), 3)

            losses['loss_reconstruction'] = -x_r.log_prob(obs).float().mean()
        else:
            raise NotImplementedError("")
            # if self.vit_l2_ratio != 1.0:
            #     x_r = self.image_predictor(posterior.combined_slots.transpose(0, 1).flatten(0, 1))
            #     img_rec = -x_r.log_prob(obs).float().mean()
            # else:
            #     img_rec = 0
            #     x_r_detached = self.image_predictor(posterior.combined_slots.transpose(0, 1).flatten(0, 1).detach())
            #     losses['loss_reconstruction_img'] = -x_r_detached.log_prob(obs).float().mean()
            # d_pred = self.dino_predictor(posterior.combined_slots.transpose(0, 1).flatten(0, 1))
            # losses['loss_reconstruction'] = (self.vit_l2_ratio * -d_pred.log_prob(d_features.reshape(b, self.vit_feat_dim, 14, 14)).float().mean()/4 +
            #                                 (1-self.vit_l2_ratio) * img_rec)

        prior_logits = prior.stoch_logits
        posterior_logits = posterior.stoch_logits
        losses['loss_reward_pred'] = -r_pred.log_prob(r_c).float().mean()
        losses['loss_discount_pred'] = -f_pred.log_prob(d_c).float().mean()
        losses['loss_kl_reg'] = KL(prior_logits, posterior_logits)

        metrics['reward_mean'] = r.mean()
        metrics['reward_std'] = r.std()
        metrics['reward_sae'] = (torch.abs(r_pred.mode - r_c)).mean()
        metrics['prior_entropy'] = Dist(prior_logits).entropy().mean()
        metrics['posterior_entropy'] = Dist(posterior_logits).entropy().mean()

        return losses, posterior, metrics


class ImaginativeCritic(nn.Module):

    def __init__(self, discount_factor: float, update_interval: int,
                 soft_update_fraction: float, value_target_lambda: float, latent_dim: int, layer_norm: bool):
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
                (1 - self.lambda_) * vs[i+1] +
                self.lambda_ * v_lambdas[-1])
            v_lambdas.append(v_lambda)

        # FIXME: it copies array, so it is quite slow
        return torch.stack(v_lambdas).flip(dims=(0,))[:-1]

    def lambda_return(self, zs, rs, ds):
        vs = self.target_critic(zs).mode
        return self._lambda_return(vs, rs, ds)


class DreamerV2(RlAgent):

    def __init__(
            self,
            obs_space_num: list[int],  # NOTE: encoder/decoder will work only with 64x64 currently
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
            world_model_predict_discount: bool,
            actor_lr: float,
            critic_lr: float,
            discrete_rssm: bool,
            layer_norm: bool,
            encode_vit: bool,
            decode_vit: bool,
            vit_l2_ratio: float,
            slots_num: int,
            device_type: str = 'cpu',
            logger = None):

        self.logger = logger
        self.device = device_type
        self.imagination_horizon = imagination_horizon
        self.cluster_size = batch_cluster_size
        self.actions_num = actions_num
        self.rho = actor_reinforce_fraction
        self.eta = actor_entropy_scale
        self.is_discrete = (action_type != 'continuous')
        if self.rho is None:
            self.rho = self.is_discrete

        self.world_model = WorldModel(obs_space_num[0], batch_cluster_size, latent_dim, latent_classes,
                                      rssm_dim, actions_num, kl_loss_scale,
                                      kl_loss_balancing, kl_loss_free_nats,
                                      discrete_rssm,
                                      world_model_predict_discount, layer_norm,
                                      encode_vit, decode_vit, vit_l2_ratio, slots_num).to(device_type)

        self.actor = fc_nn_generator(slots_num*(rssm_dim + latent_dim * latent_classes),
                                     actions_num if self.is_discrete else actions_num * 2,
                                     400,
                                     5,
                                     layer_norm=layer_norm,
                                     intermediate_activation=nn.ELU,
                                     final_activation=DistLayer('onehot' if self.is_discrete else 'normal_trunc')).to(device_type)

        self.critic = ImaginativeCritic(discount_factor, critic_update_interval,
                                        critic_soft_update_fraction,
                                        critic_value_target_lambda,
                                        slots_num*(rssm_dim + latent_dim * latent_classes),
                                        layer_norm=layer_norm).to(device_type)

        self.scaler = torch.cuda.amp.GradScaler()
        self.image_predictor_optimizer = torch.optim.AdamW(self.world_model.image_predictor.parameters(),
                                                           lr=world_model_lr,
                                                           eps=1e-5,
                                                           weight_decay=1e-6)

        self.world_model_optimizer = torch.optim.AdamW(self.world_model.parameters(),
                                                       lr=world_model_lr,
                                                       eps=1e-5,
                                                       weight_decay=1e-6)

        warmup_steps = 1e3
        decay_rate = 0.5
        decay_steps = 5e5
        lr_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.world_model_optimizer, start_factor=1/warmup_steps, total_iters=int(warmup_steps))
        lr_decay_scheduler = torch.optim.lr_scheduler.LambdaLR(self.world_model_optimizer, lambda epoch: decay_rate**(epoch/decay_steps))
        # lr_decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate**(1/decay_steps))
        self.lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([lr_warmup_scheduler, lr_decay_scheduler])

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
            self, init_state: State, precomp_actions: t.Optional[list[Action]] = None, horizon: t.Optional[int] = None
    ) -> tuple[State, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        if horizon is None:
            horizon = self.imagination_horizon

        prev_state = init_state
        prev_action = torch.zeros_like(self.actor(prev_state.combined.detach()).mean)
        states, actions, rewards, ts = ([init_state],
                                       [prev_action],
                                       [self.world_model.reward_predictor(init_state.combined).mode],
                                       [torch.ones(prev_action.shape[:-1] + (1,), device=prev_action.device)])

        for i in range(horizon):
            if precomp_actions is not None:
                a = precomp_actions[i].unsqueeze(0)
            else:
                a_dist = self.actor(prev_state.combined.detach())
                a = a_dist.rsample()
            prior, reward, discount = self.world_model.predict_next(prev_state, a)
            prev_state = prior

            states.append(prior)
            rewards.append(reward)
            ts.append(discount)
            actions.append(a)

        return (State.stack(states), torch.cat(actions), torch.cat(rewards), torch.cat(ts))

    def reset(self):
        self._state = self.world_model.get_initial_state()
        self._prev_slots = None
        self._last_action = torch.zeros((1, 1, self.actions_num), device=self.device)
        self._latent_probs = torch.zeros((self.world_model.latent_classes, self.world_model.latent_dim), device=self.device)
        self._action_probs = torch.zeros((self.actions_num), device=self.device)
        self._stored_steps = 0

    def preprocess_obs(self, obs: torch.Tensor):
        # FIXME: move to dataloader in replay buffer
        order = list(range(len(obs.shape)))
        # Swap channel from last to 3 from last
        order = order[:-3] + [order[-1]] + order[-3:-1]
        if self.world_model.encode_vit:
            ToTensor = tv.transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225))
            return ToTensor(obs.type(torch.float32).permute(order))
        else:
            return ((obs.type(torch.float32) / 255.0) - 0.5).permute(order)
        # return obs.type(torch.float32).permute(order)

    def get_action(self, obs: Observation) -> Action:
        # NOTE: pytorch fails without .copy() only when get_action is called
        # FIXME: return back action selection
        obs = torch.from_numpy(obs.copy()).to(self.device)
        obs = self.preprocess_obs(obs)

        self._state, self._prev_slots = self.world_model.get_latent(obs, self._last_action, self._state, self._prev_slots)

        actor_dist = self.actor(self._state.combined)
        self._last_action = actor_dist.sample()

        if self.is_discrete:
            self._action_probs += actor_dist.probs.squeeze().mean(dim=0)
        self._latent_probs += self._state.stoch_dist.probs.squeeze().mean(dim=0)
        self._stored_steps += 1

        if self.is_discrete:
            return self._last_action.squeeze().detach().cpu().numpy().argmax()
        else:
            return self._last_action.squeeze().detach().cpu().numpy()

    def _generate_video(self, obs: list[Observation], actions: list[Action], update_num: int):
        obs = torch.from_numpy(obs.copy()).to(self.device)
        obs = self.preprocess_obs(obs)
        actions = self.from_np(actions)
        if self.is_discrete:
            actions = F.one_hot(actions.to(torch.int64), num_classes=self.actions_num).squeeze()
        video = []
        slots_video = []
        rews = []

        state = None
        prev_slots = None
        means = np.array([0.485, 0.456, 0.406])
        stds = np.array([0.229, 0.224, 0.225])
        UnNormalize = tv.transforms.Normalize(list(-means/stds),
                                           list(1/stds))
        for idx, (o, a) in enumerate(list(zip(obs, actions))):
            if idx > update_num:
                break
            state, prev_slots = self.world_model.get_latent(o, a.unsqueeze(0).unsqueeze(0), state, prev_slots)
            # video_r = self.world_model.image_predictor(state.combined_slots).mode.cpu().detach().numpy()

            decoded_imgs, masks = self.world_model.image_predictor(state.combined_slots.flatten(0, 1)).reshape(1, -1, 4, 64, 64).split([3, 1], dim=2)
            # TODO: try the scaling of softmax as in attention
            img_mask = F.softmax(masks, dim=1)
            decoded_imgs = decoded_imgs * img_mask
            video_r = torch.sum(decoded_imgs, dim=1).cpu().detach().numpy()

            rews.append(self.world_model.reward_predictor(state.combined).mode.item())
            if self.world_model.encode_vit:
                video_r = UnNormalize(torch.from_numpy(video_r)).numpy()
            else:
                video_r = (video_r + 0.5)
            video.append(video_r)
            slots_video.append(decoded_imgs.cpu().detach().numpy() + 0.5)

        rews = torch.Tensor(rews).to(obs.device)

        if update_num < len(obs):
            states, _, rews_2, _ = self.imagine_trajectory(state, actions[update_num+1:].unsqueeze(1), horizon=self.imagination_horizon - 1 - update_num)
            rews = torch.cat([rews, rews_2[1:].squeeze()])

            # video_r = self.world_model.image_predictor(states.combined_slots[1:]).mode.cpu().detach().numpy()
            decoded_imgs, masks = self.world_model.image_predictor(states.combined_slots[1:].flatten(0, 1)).reshape(-1, self.world_model.slots_num, 4, 64, 64).split([3, 1], dim=2)
            img_mask = F.softmax(masks, dim=1)
            decoded_imgs = decoded_imgs * img_mask
            video_r = torch.sum(decoded_imgs, dim=1).cpu().detach().numpy()

            if self.world_model.encode_vit:
                video_r = UnNormalize(torch.from_numpy(video_r)).numpy()
            else:
                video_r = (video_r + 0.5)
            video.append(video_r)
            slots_video.append(decoded_imgs.cpu().detach().numpy() + 0.5)

        return np.concatenate(video), rews, np.concatenate(slots_video)

    def viz_log(self, rollout, logger, epoch_num):
        init_indeces = np.random.choice(len(rollout.states) - self.imagination_horizon, 5)

        videos = np.concatenate([
            rollout.next_states[init_idx:init_idx + self.imagination_horizon].transpose(
                0, 3, 1, 2) for init_idx in init_indeces
        ], axis=3).astype(np.float32) / 255.0

        real_rewards = [rollout.rewards[idx:idx+ self.imagination_horizon] for idx in init_indeces]

        videos_r, imagined_rewards, slots_video = zip(*[self._generate_video(obs_0.copy(), a_0, update_num=self.imagination_horizon//3) for obs_0, a_0 in zip(
                [rollout.next_states[idx:idx+ self.imagination_horizon] for idx in init_indeces],
                [rollout.actions[idx:idx+ self.imagination_horizon] for idx in init_indeces])
        ])
        videos_r = np.concatenate(videos_r, axis=3)

        slots_video = np.concatenate(list(slots_video)[:3], axis=3)
        slots_video = slots_video.transpose((0, 2, 3, 1, 4))
        slots_video = np.expand_dims(slots_video.reshape(*slots_video.shape[:-2], -1), 0)

        videos_comparison = np.expand_dims(np.concatenate([videos, videos_r, np.abs(videos - videos_r + 1)/2], axis=2), 0)
        videos_comparison = (videos_comparison * 255.0).astype(np.uint8)
        latent_hist = (self._latent_probs / self._stored_steps).detach().cpu().numpy()
        latent_hist = ((latent_hist / latent_hist.max() * 255.0 )).astype(np.uint8)

        # if discrete action space
        if self.is_discrete:
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
        logger.add_video('val/dreamed_slots', slots_video, epoch_num)

        rewards_err = torch.Tensor([torch.abs(sum(imagined_rewards[i]) - real_rewards[i].sum()) for i in range(len(imagined_rewards))]).mean()
        logger.add_scalar('val/img_reward_err', rewards_err.item(), epoch_num)

        logger.add_scalar(f'val/reward', real_rewards[0].sum(), epoch_num)

    def from_np(self, arr: np.ndarray):
        arr = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
        return arr.to(self.device, non_blocking=True)

    def train(self, obs: Observations, a: Actions, r: Rewards, next_obs: Observations,
              is_finished: TerminationFlags, is_first: IsFirstFlags):

        obs = self.preprocess_obs(self.from_np(obs))
        a = self.from_np(a)
        if self.is_discrete:
            a = F.one_hot(a.to(torch.int64), num_classes=self.actions_num).squeeze()
        r = self.from_np(r)
        discount_factors = (1 - self.from_np(is_finished).type(torch.float32))
        first_flags = self.from_np(is_first).type(torch.float32)

        # take some latent embeddings as initial
        with torch.cuda.amp.autocast(enabled=False):
            losses, discovered_states, wm_metrics = self.world_model.calculate_loss(obs, a, r, discount_factors, first_flags)
            self.world_model.recurrent_model.discretizer_scheduler.step()

            # NOTE: 'aten::nonzero' inside KL divergence is not currently supported on M1 Pro MPS device
            image_predictor_loss = losses['loss_reconstruction_img']
            world_model_loss = (losses['loss_reconstruction'] +
                                losses['loss_reward_pred'] +
                                losses['loss_kl_reg'] +
                                losses['loss_discount_pred'])
        # for l in losses.values():
        #     world_model_loss += l
        if self.world_model.decode_vit and self.world_model.vit_l2_ratio == 1.0:
            self.image_predictor_optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(image_predictor_loss).backward()
            self.scaler.unscale_(self.image_predictor_optimizer)
            nn.utils.clip_grad_norm_(self.world_model.image_predictor.parameters(), 100)
            self.scaler.step(self.image_predictor_optimizer)

        self.world_model_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(world_model_loss).backward()

        # FIXME: clip gradient should be parametrized
        self.scaler.unscale_(self.world_model_optimizer)
        # for tag, value in self.world_model.named_parameters():
        #     wm_metrics[f"grad/{tag.replace('.', '/')}"] = value.detach()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 100)
        self.scaler.step(self.world_model_optimizer)
        self.lr_scheduler.step()

        metrics = wm_metrics

        with torch.cuda.amp.autocast(enabled=False):
            losses_ac = {}
            initial_states = State(discovered_states.determ.flatten(0, 1).unsqueeze(0).detach(),
                                   discovered_states.stoch_logits.flatten(0, 1).unsqueeze(0).detach(),
                                   discovered_states.stoch_.flatten(0, 1).unsqueeze(0).detach())

            states, actions, rewards, discount_factors = self.imagine_trajectory(initial_states)
            zs = states.combined
            rewards = self.world_model.reward_normalizer(rewards)

            # Discounted factors should be shifted as they predict whether next state cannot be used
            # First discount factor on contrary is always 1 as it cannot lead to trajectory finish
            discount_factors = torch.cat([torch.ones_like(discount_factors[:1]), discount_factors[:-1]], dim=0).detach()

            vs = self.critic.lambda_return(zs, rewards[:-1], discount_factors)

            # Ignore all factors after first is_finished state
            discount_factors = torch.cumprod(discount_factors, dim=0)

            predicted_vs_dist = self.critic.estimate_value(zs[:-1].detach())
            losses_ac['loss_critic'] = -(predicted_vs_dist.log_prob(vs.detach()).unsqueeze(2)*discount_factors[:-1]).mean()

            metrics['critic/avg_target_value'] = self.critic.target_critic(zs[1:]).mode.mean()
            metrics['critic/avg_lambda_value'] = vs.mean()
            metrics['critic/avg_predicted_value'] = predicted_vs_dist.mode.mean()

            # last action should be ignored as it is not used to predict next state, thus no feedback
            # first value should be ignored as it is comes from replay buffer
            action_dists = self.actor(zs[:-2].detach())
            baseline = self.critic.target_critic(zs[:-2]).mode
            advantage = (vs[1:] - baseline).detach()
            losses_ac['loss_actor_reinforce'] = -(self.rho * action_dists.log_prob(actions[1:-1].detach()).unsqueeze(2) * discount_factors[:-2] * advantage).mean()
            losses_ac['loss_actor_dynamics_backprop'] = -((1 - self.rho) * (vs[1:]*discount_factors[:-2])).mean()

            def calculate_entropy(dist):
                return dist.entropy().unsqueeze(2)
                # return dist.base_dist.base_dist.entropy().unsqueeze(2)

            losses_ac['loss_actor_entropy'] = -(self.eta * calculate_entropy(action_dists)*discount_factors[:-2]).mean()
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

        losses = {l: val.detach().cpu().numpy() for l, val in losses.items()}
        losses_ac = {l: val.detach().cpu().numpy() for l, val in losses_ac.items()}
        metrics = {l: val.detach().cpu().numpy() for l, val in metrics.items()}

        losses['total'] = sum(losses.values())
        return losses | losses_ac | metrics

    def save_ckpt(self, epoch_num: int, losses: dict[str, float]):
        torch.save(
            {
                'epoch': epoch_num,
                'world_model_state_dict': self.world_model.state_dict(),
                'world_model_optimizer_state_dict': self.world_model_optimizer.state_dict(),
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'losses': losses
            }, f'dreamerV2-{epoch_num}-{losses["total"]}.ckpt')

    def load_ckpt(self, ckpt_path: Path):
        ckpt = torch.load(ckpt_path)
        self.world_model.load_state_dict(ckpt['world_model_state_dict'])
        self.world_model_optimizer.load_state_dict(
            ckpt['world_model_optimizer_state_dict'])
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])
        return ckpt['epoch']
