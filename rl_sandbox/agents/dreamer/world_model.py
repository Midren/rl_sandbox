import typing as t

import torch
import torch.distributions as td
import torchvision as tv
from torch import nn
from torch.nn import functional as F

from rl_sandbox.agents.dreamer import Dist, Normalizer, View
from rl_sandbox.agents.dreamer.rssm import RSSM, State
from rl_sandbox.agents.dreamer.vision import Decoder, Encoder, SpatialBroadcastDecoder
from rl_sandbox.utils.dists import DistLayer
from rl_sandbox.utils.fc_nn import fc_nn_generator
from rl_sandbox.vision.dino import ViTFeat
from rl_sandbox.vision.slot_attention import PositionalEmbedding, SlotAttention

class WorldModel(nn.Module):

    def __init__(self, batch_cluster_size, latent_dim, latent_classes, rssm_dim,
                 actions_num, discount_loss_scale, kl_loss_scale, kl_loss_balancing, kl_free_nats, discrete_rssm,
                 predict_discount, layer_norm: bool, encode_vit: bool, decode_vit: bool,
                 vit_l2_ratio: float, vit_img_size: int):
        super().__init__()
        self.register_buffer('kl_free_nats', kl_free_nats * torch.ones(1))
        self.discount_scale = discount_loss_scale
        self.kl_beta = kl_loss_scale

        self.rssm_dim = rssm_dim
        self.latent_dim = latent_dim
        self.latent_classes = latent_classes
        self.state_size = (rssm_dim + latent_dim * latent_classes)

        self.cluster_size = batch_cluster_size
        self.actions_num = actions_num
        # kl loss balancing (prior/posterior)
        self.alpha = kl_loss_balancing
        self.predict_discount = predict_discount
        self.encode_vit = encode_vit
        self.decode_vit = decode_vit
        self.vit_l2_ratio = vit_l2_ratio
        self.vit_img_size = vit_img_size

        self.recurrent_model = RSSM(latent_dim,
                                    rssm_dim,
                                    actions_num,
                                    latent_classes,
                                    discrete_rssm,
                                    norm_layer=nn.LayerNorm if layer_norm else nn.Identity)
        if encode_vit or decode_vit:
            if self.vit_img_size == 224:
                self.dino_vit = ViTFeat("/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
                                        feat_dim=384, vit_arch='small', patch_size=16)
                self.decoder_kernels = [3, 3, 2]
                self.vit_size = 14
            elif self.vit_img_size == 64:
                self.dino_vit = ViTFeat("/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
                                       feat_dim=384, vit_arch='small', patch_size=8)
                self.decoder_kernels = [3, 4]
                self.vit_size = 8
            else:
                raise RuntimeError("Unknown vit img size")
            # self.dino_vit = ViTFeat("/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth", feat_dim=768, vit_arch='base', patch_size=8)
            self.vit_feat_dim = self.dino_vit.feat_dim
            self.dino_vit.requires_grad_(False)

        if encode_vit:
            self.post_vit = nn.Sequential(
                View((-1, self.vit_feat_dim, self.vit_size, self.vit_size)),
                Encoder(norm_layer=nn.GroupNorm if layer_norm else nn.Identity,
                        kernel_sizes=[2],
                        channel_step=384,
                        flatten_output=False,
                        in_channels=self.vit_feat_dim
                        )
            )
            self.encoder = nn.Sequential(
                self.dino_vit,
                self.post_vit
            )
        else:
            self.encoder = Encoder(norm_layer=nn.GroupNorm if layer_norm else nn.Identity,
                                   kernel_sizes=[4, 4, 4, 4],
                                   channel_step=48)

        if decode_vit:
            self.dino_predictor = SpatialBroadcastDecoder(self.state_size,
                                          norm_layer=nn.GroupNorm if layer_norm else nn.Identity,
                                          out_image=(14, 14),
                                          kernel_sizes = [5, 5, 5, 5],
                                          channel_step=self.vit_feat_dim,
                                          output_channels=self.vit_feat_dim,
                                          return_dist=True)

        self.image_predictor = Decoder(self.state_size,
                                       norm_layer=nn.GroupNorm if layer_norm else nn.Identity)

        self.reward_predictor = fc_nn_generator(self.state_size,
                                                1,
                                                hidden_size=400,
                                                num_layers=5,
                                                intermediate_activation=nn.ELU,
                                                layer_norm=layer_norm,
                                                final_activation=DistLayer('mse'))
        self.discount_predictor = fc_nn_generator(self.state_size,
                                                  1,
                                                  hidden_size=400,
                                                  num_layers=5,
                                                  intermediate_activation=nn.ELU,
                                                  layer_norm=layer_norm,
                                                  final_activation=DistLayer('binary'))
        self.reward_normalizer = Normalizer(momentum=1.00, scale=1.0, eps=1e-8)

    def precalc_data(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.decode_vit:
            return {}
        if not self.encode_vit:
            ToTensor = tv.transforms.Compose([tv.transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225)),
                                              tv.transforms.Resize(self.vit_img_size, antialias=True)])
            obs = ToTensor(obs + 0.5)
        with torch.no_grad():
            d_features = self.dino_vit(obs).cpu()
        return {'d_features': d_features}

    def get_initial_state(self, batch_size: int = 1, seq_size: int = 1):
        device = next(self.parameters()).device
        return State(torch.zeros(seq_size, batch_size, self.rssm_dim, device=device),
                     torch.zeros(seq_size, batch_size, self.latent_classes, self.latent_dim, device=device),
                     torch.zeros(seq_size, batch_size, self.latent_classes * self.latent_dim, device=device))

    def predict_next(self, prev_state: State, action):
        prior, _ = self.recurrent_model.predict_next(prev_state, action)

        # FIXME: rewrite to utilize batch processing
        reward = self.reward_predictor(prior.combined).mode
        if self.predict_discount:
            discount_factors = self.discount_predictor(prior.combined).mode
        else:
            discount_factors = torch.ones_like(reward)
        return prior, reward, discount_factors

    def get_latent(self, obs: torch.Tensor, action, state: t.Optional[State]) -> State:
        if state is None:
            state = self.get_initial_state()
        embed = self.encoder(obs.unsqueeze(0))
        _, posterior, _ = self.recurrent_model.forward(state, embed.unsqueeze(0),
                                                           action)
        return posterior

    def calculate_loss(self, obs: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                       discount: torch.Tensor, first: torch.Tensor, additional: dict[str, torch.Tensor]):
        self.recurrent_model.on_train_step()
        b, _, h, w = obs.shape  # s <- BxHxWx3

        if self.encode_vit:
            embed = self.post_vit(additional['d_features'])
        else:
            embed = self.encoder(obs)
        embed_c = embed.reshape(b // self.cluster_size, self.cluster_size, -1)

        a_c = a.reshape(-1, self.cluster_size, self.actions_num)
        r_c = r.reshape(-1, self.cluster_size, 1)
        d_c = discount.reshape(-1, self.cluster_size, 1)
        first_c = first.reshape(-1, self.cluster_size, 1)

        losses = {}
        metrics = {}

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
            return ((self.alpha * kl_lhs + (1 - self.alpha) * kl_rhs))

        priors = []
        posteriors = []

        if self.decode_vit:
            d_features = additional['d_features']

        prev_state = self.get_initial_state(b // self.cluster_size)
        for t in range(self.cluster_size):
            # s_t <- 1xB^xHxWx3
            embed_t, a_t, first_t = embed_c[:, t].unsqueeze(0), a_c[:, t].unsqueeze(0), first_c[:, t].unsqueeze(0)
            a_t = a_t * (1 - first_t)

            prior, posterior, diff = self.recurrent_model.forward(prev_state, embed_t, a_t)
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
            x_r = self.image_predictor(posterior.combined.transpose(0, 1).flatten(0, 1))
            losses['loss_reconstruction'] = -x_r.log_prob(obs).float().mean()
        else:
            if self.vit_l2_ratio != 1.0:
                x_r = self.image_predictor(posterior.combined.transpose(0, 1).flatten(0, 1))
                img_rec = -x_r.log_prob(obs).float().mean()
            else:
                img_rec = torch.tensor(0, device=obs.device)
                x_r_detached = self.image_predictor(posterior.combined.transpose(0, 1).flatten(0, 1).detach())
                losses['loss_reconstruction_img'] = -x_r_detached.log_prob(obs).float().mean()

            d_pred = self.dino_predictor(posterior.combined.transpose(0, 1).flatten(0, 1))
            d_obs = d_features.reshape(b, self.vit_feat_dim, self.vit_size, self.vit_size)
            d_rec = -d_pred.log_prob(d_obs).float().mean()
            d_rec = d_rec / torch.prod(torch.tensor(d_obs.shape[-3:])) * torch.prod(torch.tensor(obs.shape[-3:]))

            losses['loss_reconstruction'] = (self.vit_l2_ratio * d_rec + (1-self.vit_l2_ratio) * img_rec)
            metrics['loss_l2_rec'] = img_rec
            metrics['loss_dino_rec'] = d_rec

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

        losses['loss_wm'] = (losses['loss_reconstruction'] + losses['loss_reward_pred'] +
                             self.kl_beta * losses['loss_kl_reg'] + self.discount_scale*losses['loss_discount_pred'])

        return losses, posterior, metrics
