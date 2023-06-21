import typing as t
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import nn
from torch.nn import functional as F

from rl_sandbox.agents.dreamer import Dist, View, GRUCell
from rl_sandbox.utils.schedulers import LinearScheduler

@dataclass
class State:
    determ: Float[torch.Tensor, 'seq batch determ']
    stoch_logits: Float[torch.Tensor, 'seq batch latent_classes latent_dim']
    stoch_: t.Optional[Bool[torch.Tensor, 'seq batch stoch_dim']] = None

    @property
    def combined(self):
        return torch.concat([self.determ, self.stoch], dim=-1)

    @property
    def stoch(self):
        if self.stoch_ is None:
            self.stoch_ = Dist(self.stoch_logits).rsample().reshape(self.stoch_logits.shape[:2] + (-1,))
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


class Quantize(nn.Module):

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.inp_in = nn.Linear(1024, self.n_embed * self.dim)
        self.inp_out = nn.Linear(self.n_embed * self.dim, 1024)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, inp):
        # input = self.inp_in(inp).reshape(-1, 1, self.n_embed, self.dim)
        input = inp.reshape(-1, 1, self.n_embed, self.dim)
        inp = input
        flatten = input.reshape(-1, self.dim)
        dist = (flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed +
                self.embed.pow(2).sum(0, keepdim=True))
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum,
                                                         alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) /
                            (n + self.n_embed * self.eps) * n)
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # quantize_out = self.inp_out(quantize.reshape(-1, self.n_embed*self.dim))
        quantize_out = quantize
        diff = 0.25 * (quantize_out.detach() - inp).pow(2).mean() + (
            quantize_out - inp.detach()).pow(2).mean()
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
        img_sz = 4 * 384  # 384*2x2
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

    def on_train_step(self):
        pass

    def predict_next(self,
                     prev_state: State,
                     action) -> State:
        x = self.pre_determ_recurrent(torch.concat([prev_state.stoch, action], dim=-1))
        # NOTE: x and determ are actually the same value if sequence of 1 is inserted
        x, determ_prior = self.determ_recurrent(x, prev_state.determ)
        if self.discrete_rssm:
            determ_post, diff, embed_ind = self.determ_discretizer(determ_prior)
            determ_post = determ_post.reshape(determ_prior.shape)
            determ_post = self.determ_layer_norm(determ_post)
            alpha = self.discretizer_scheduler.val
            determ_post = alpha * determ_prior + (1-alpha) * determ_post
        else:
            determ_post, diff = determ_prior, 0

        # used for KL divergence
        predicted_stoch_logits = self.estimate_stochastic_latent(x)
        return State(determ_post, predicted_stoch_logits), diff

    def update_current(self, prior: State, embed) -> State: # Dreamer 'obs_out'
        return State(prior.determ, self.stoch_net(torch.concat([prior.determ, embed], dim=-1)))

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
