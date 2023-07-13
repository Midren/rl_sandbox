import typing as t

import torch
import torch.distributions as td
import torchvision as tv
from torch import nn
from torch.nn import functional as F

from rl_sandbox.agents.dreamer import Dist, Normalizer, View
from rl_sandbox.agents.dreamer.rssm_slots_attention import RSSM, State
from rl_sandbox.agents.dreamer.vision import Decoder, Encoder
from rl_sandbox.utils.dists import DistLayer
from rl_sandbox.utils.fc_nn import fc_nn_generator
from rl_sandbox.vision.dino import ViTFeat
from rl_sandbox.vision.slot_attention import PositionalEmbedding, SlotAttention


class WorldModel(nn.Module):

    def __init__(self, batch_cluster_size, latent_dim, latent_classes, rssm_dim,
                 actions_num, discount_loss_scale, kl_loss_scale, kl_loss_balancing, kl_free_nats,
                 discrete_rssm, predict_discount, layer_norm: bool, encode_vit: bool,
                 decode_vit: bool, vit_l2_ratio: float, vit_img_size: int, slots_num: int, slots_iter_num: int, use_prev_slots: bool = True,
                 full_qk_from: int = 1,
                 symmetric_qk: bool = False,
                 attention_block_num: int = 3,
                 mask_combination: str = 'soft',
                 per_slot_rec_loss: bool = False):
        super().__init__()
        self.use_prev_slots = use_prev_slots
        self.register_buffer('kl_free_nats', kl_free_nats * torch.ones(1))
        self.discount_scale = discount_loss_scale
        self.kl_beta = kl_loss_scale

        self.rssm_dim = rssm_dim
        self.latent_dim = latent_dim
        self.latent_classes = latent_classes
        self.slots_num = slots_num
        self.mask_combination = mask_combination
        self.state_size = slots_num * (rssm_dim + latent_dim * latent_classes)

        self.cluster_size = batch_cluster_size
        self.actions_num = actions_num
        # kl loss balancing (prior/posterior)
        self.alpha = kl_loss_balancing
        self.predict_discount = predict_discount
        self.encode_vit = encode_vit
        self.decode_vit = decode_vit
        self.vit_l2_ratio = vit_l2_ratio
        self.vit_img_size = vit_img_size
        self.per_slot_rec_loss = per_slot_rec_loss

        self.n_dim = 384

        self.recurrent_model = RSSM(
            latent_dim,
            rssm_dim,
            actions_num,
            latent_classes,
            discrete_rssm,
            norm_layer=nn.LayerNorm if layer_norm else nn.Identity,
            embed_size=self.n_dim,
            full_qk_from=full_qk_from,
            symmetric_qk=symmetric_qk,
            attention_block_num=attention_block_num)
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
                        double_conv=False,
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
                                   kernel_sizes=[4, 4, 4],
                                   channel_step=96,
                                   double_conv=True,
                                   flatten_output=False)

        self.slot_attention = SlotAttention(slots_num, self.n_dim, slots_iter_num)
        if self.encode_vit:
            self.positional_augmenter_inp = PositionalEmbedding(self.n_dim, (4, 4))
        else:
            self.positional_augmenter_inp = PositionalEmbedding(self.n_dim, (6, 6))
        # self.positional_augmenter_dec = PositionalEmbedding(self.n_dim, (8, 8))

        self.slot_mlp = nn.Sequential(nn.Linear(self.n_dim, self.n_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.n_dim, self.n_dim))

        if decode_vit:
            self.dino_predictor = Decoder(rssm_dim + latent_dim * latent_classes,
                                          norm_layer=nn.GroupNorm if layer_norm else nn.Identity,
                                          conv_kernel_sizes=[3],
                                          channel_step=2*self.vit_feat_dim,
                                          kernel_sizes=self.decoder_kernels,
                                          output_channels=self.vit_feat_dim+1,
                                          return_dist=False)
        self.image_predictor = Decoder(
            rssm_dim + latent_dim * latent_classes,
            norm_layer=nn.GroupNorm if layer_norm else nn.Identity,
            output_channels=3+1,
            return_dist=False)

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

    def slot_mask(self, masks: torch.Tensor) -> torch.Tensor:
        match self.mask_combination:
            case 'soft':
                img_mask = F.softmax(masks, dim=1)
            case 'hard':
                probs = F.softmax(masks - masks.logsumexp(dim=1,keepdim=True), dim=1)
                img_mask = F.one_hot(masks.argmax(dim=1), num_classes=masks.shape[1]).permute(0, 4, 1, 2, 3) + (probs - probs.detach())
            case 'qmix':
                raise NotImplementedError
            case _:
                raise NotImplementedError
        return img_mask

    def precalc_data(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.decode_vit:
            return {}
        if not self.encode_vit:
            ToTensor = tv.transforms.Compose([tv.transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225)),
                                              tv.transforms.Resize(self.vit_img_size, antialias=True)])
            obs = ToTensor(obs + 0.5)
        d_features = self.dino_vit(obs)
        return {'d_features': d_features}

    def get_initial_state(self, batch_size: int = 1, seq_size: int = 1):
        device = next(self.parameters()).device
        # Tuple of State-Space state and prev slots
        return State(
            torch.zeros(seq_size,
                        batch_size,
                        self.slots_num,
                        self.rssm_dim,
                        device=device),
            torch.zeros(seq_size,
                        batch_size,
                        self.slots_num,
                        self.latent_classes,
                        self.latent_dim,
                        device=device),
            torch.zeros(seq_size,
                        batch_size,
                        self.slots_num,
                        self.latent_classes * self.latent_dim,
                        device=device)), None

    def predict_next(self, prev_state: State, action):
        prior, _ = self.recurrent_model.predict_next(prev_state, action)

        reward = self.reward_predictor(prior.combined).mode
        if self.predict_discount:
            discount_factors = self.discount_predictor(prior.combined).sample()
        else:
            discount_factors = torch.ones_like(reward)
        return prior, reward, discount_factors

    def get_latent(self, obs: torch.Tensor, action, state: t.Optional[tuple[State, torch.Tensor]]) -> t.Tuple[State, torch.Tensor]:
        if state is None or state[0] is None:
            state, prev_slots = self.get_initial_state()
        else:
            if self.use_prev_slots:
                state, prev_slots = state
            else:
                state, prev_slots = state[0], None
        embed = self.encoder(obs.unsqueeze(0))
        embed_with_pos_enc = self.positional_augmenter_inp(embed)

        pre_slot_features_t = self.slot_mlp(
            embed_with_pos_enc.permute(0, 2, 3, 1).reshape(1, -1, self.n_dim))

        slots_t = self.slot_attention(pre_slot_features_t, prev_slots)

        _, posterior, _ = self.recurrent_model.forward(state, slots_t.unsqueeze(0),
                                                       action)
        return posterior, slots_t

    def calculate_loss(self, obs: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                       discount: torch.Tensor, first: torch.Tensor, additional: dict[str, torch.Tensor]):
        self.recurrent_model.on_train_step()
        b, _, h, w = obs.shape  # s <- BxHxWx3

        embed = self.encoder(obs)
        if self.encode_vit:
            embed = self.post_vit(additional['d_features'])
        else:
            embed = self.encoder(obs)
        embed_with_pos_enc = self.positional_augmenter_inp(embed)

        pre_slot_features = self.slot_mlp(
            embed_with_pos_enc.permute(0, 2, 3, 1).reshape(b, -1, self.n_dim))
        pre_slot_features_c = pre_slot_features.reshape(b // self.cluster_size,
                                                        self.cluster_size, -1, self.n_dim)

        a_c = a.reshape(-1, self.cluster_size, self.actions_num)
        r_c = r.reshape(-1, self.cluster_size, 1)
        d_c = discount.reshape(-1, self.cluster_size, 1)
        first_c = first.reshape(-1, self.cluster_size, 1)

        losses = {}
        metrics = {}

        def KL(dist1, dist2):
            KL_ = torch.distributions.kl_divergence
            kl_lhs = KL_(td.OneHotCategoricalStraightThrough(logits=dist2.detach()),
                         td.OneHotCategoricalStraightThrough(logits=dist1)).mean()
            kl_rhs = KL_(
                td.OneHotCategoricalStraightThrough(logits=dist2),
                td.OneHotCategoricalStraightThrough(logits=dist1.detach())).mean()
            kl_lhs = torch.maximum(kl_lhs, self.kl_free_nats)
            kl_rhs = torch.maximum(kl_rhs, self.kl_free_nats)
            return ((self.alpha * kl_lhs + (1 - self.alpha) * kl_rhs))

        priors = []
        posteriors = []

        if self.decode_vit:
            d_features = additional['d_features']

        prev_state, prev_slots = self.get_initial_state(b // self.cluster_size)

        self.last_attn = torch.zeros((self.slots_num, self.slots_num), device=a_c.device)

        for t in range(self.cluster_size):
            # s_t <- 1xB^xHxWx3
            pre_slot_feature_t, a_t, first_t = pre_slot_features_c[:,
                                                                   t], a_c[:, t].unsqueeze(
                                                                       0
                                                                   ), first_c[:,
                                                                              t].unsqueeze(
                                                                                  0)
            a_t = a_t * (1 - first_t)

            slots_t = self.slot_attention(pre_slot_feature_t, prev_slots)
            # FIXME: prev_slots was not used properly, need to rerun test
            if self.use_prev_slots:
                prev_slots = slots_t
            else:
                prev_slots = None

            prior, posterior, diff = self.recurrent_model.forward(
                prev_state, slots_t.unsqueeze(0), a_t)
            prev_state = posterior
            self.last_attn += self.recurrent_model.last_attention

            priors.append(prior)
            posteriors.append(posterior)

            # losses['loss_determ_recons'] += diff

        self.last_attn /= self.cluster_size

        posterior = State.stack(posteriors)
        prior = State.stack(priors)

        r_pred = self.reward_predictor(posterior.combined.transpose(0, 1))
        f_pred = self.discount_predictor(posterior.combined.transpose(0, 1))

        losses['loss_reconstruction_img'] = torch.tensor(0, device=obs.device)

        if not self.decode_vit:
            decoded_imgs, masks = self.image_predictor(posterior.combined_slots.transpose(0, 1).flatten(0, 2)).reshape(b, -1, 4, h, w).split([3, 1], dim=2)
            img_mask = self.slot_mask(masks)

            if self.per_slot_rec_loss:
                l2_loss = (img_mask * ((decoded_imgs - obs.unsqueeze(1))**2)).mean(dim=[2, 3, 4])
                normalizing_factor = (torch.prod(torch.tensor(obs.shape[1:]))) / img_mask.sum(dim=[2, 3, 4])
                # magic constant that describes the difference between log_prob and mse losses
                img_rec = (l2_loss * normalizing_factor).sum(dim=1).mean() * self.slots_num * 8
                decoded_imgs = decoded_imgs * img_mask
            else:
                decoded_imgs = decoded_imgs * img_mask
                x_r = td.Independent(td.Normal(torch.sum(decoded_imgs, dim=1), 1.0), 3)
                img_rec = -x_r.log_prob(obs).float().mean()

            losses['loss_reconstruction'] = img_rec
        else:
            if self.vit_l2_ratio != 1.0:
                decoded_imgs, masks = self.image_predictor(posterior.combined_slots.transpose(0, 1).flatten(0, 2)).reshape(b, -1, 4, h, w).split([3, 1], dim=2)
                img_mask = self.slot_mask(masks)

                if self.per_slot_rec_loss:
                    l2_loss = (img_mask*((decoded_imgs - obs.unsqueeze(1))**2)).mean(dim=[2, 3, 4])
                    normalizing_factor = (torch.prod(torch.tensor(obs.shape[1:])))/img_mask.sum(dim=[2, 3, 4])
                    # magic constant that describes the difference between log_prob and mse losses
                    img_rec = (l2_loss * normalizing_factor).sum(dim=1).mean() * self.slots_num * 8
                    decoded_imgs = decoded_imgs * img_mask
                else:
                    decoded_imgs = decoded_imgs * img_mask
                    x_r = td.Independent(td.Normal(torch.sum(decoded_imgs, dim=1), 1.0), 3)
                    img_rec = -x_r.log_prob(obs).float().mean()
            else:
                img_rec = 0
                decoded_imgs_detached, masks = self.image_predictor(posterior.combined_slots.transpose(0, 1).flatten(0, 2).detach()).reshape(b, -1, 4, h, w).split([3, 1], dim=2)
                img_mask = self.slot_mask(masks)

                if self.per_slot_rec_loss:
                    l2_loss = (img_mask*((decoded_imgs_detached - obs.unsqueeze(1))**2)).mean(dim=[2, 3, 4])
                    normalizing_factor = (torch.prod(torch.tensor(obs.shape[1:])))/img_mask.sum(dim=[2, 3, 4])
                    # magic constant that describes the difference between log_prob and mse losses
                    img_rec_detached = (l2_loss * normalizing_factor).sum(dim=1).mean() * self.slots_num * 8
                else:
                    decoded_imgs_detached = decoded_imgs_detached * img_mask
                    x_r_detached = td.Independent(td.Normal(torch.sum(decoded_imgs_detached, dim=1), 1.0), 3)
                    img_rec_detached = -x_r_detached.log_prob(obs).float().mean()

                losses['loss_reconstruction_img'] = img_rec_detached

            decoded_feats, masks = self.dino_predictor(posterior.combined_slots.transpose(0, 1).flatten(0, 1)).reshape(b, -1, self.vit_feat_dim+1, self.vit_size, self.vit_size).split([self.vit_feat_dim, 1], dim=2)
            feat_mask = self.slot_mask(masks)

            d_obs = d_features.reshape(b, self.vit_feat_dim, self.vit_size, self.vit_size)

            if self.per_slot_rec_loss:
                l2_loss = (feat_mask*((decoded_feats - d_obs.unsqueeze(1))**2)).mean(dim=[2, 3, 4])
                normalizing_factor = (torch.prod(torch.tensor(d_obs.shape[1:])))/feat_mask.sum(dim=[2, 3, 4]).clamp(min=1)
                # # magic constant that describes the difference between log_prob and mse losses
                d_rec = (l2_loss * normalizing_factor).sum(dim=1).mean()*self.slots_num * 4
                decoded_feats = decoded_feats * feat_mask
            else:
                decoded_feats = decoded_feats * feat_mask
                d_pred = td.Independent(td.Normal(torch.sum(decoded_feats, dim=1), 1.0), 3)
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

        metrics['attention_coeff'] = torch.tensor(self.recurrent_model.attention_scheduler.val)
        metrics['reward_mean'] = r.mean()
        metrics['reward_std'] = r.std()
        metrics['reward_sae'] = (torch.abs(r_pred.mode - r_c)).mean()
        metrics['prior_entropy'] = Dist(prior_logits).entropy().mean()
        metrics['posterior_entropy'] = Dist(posterior_logits).entropy().mean()

        losses['loss_wm'] = (losses['loss_reconstruction'] + losses['loss_reward_pred'] +
                             self.kl_beta * losses['loss_kl_reg'] + self.discount_scale*losses['loss_discount_pred'])

        return losses, posterior, metrics

