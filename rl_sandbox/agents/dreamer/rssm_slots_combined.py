import typing as t
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import nn

from rl_sandbox.agents.dreamer import Dist, View, GRUCell


@dataclass
class State:
    determ: Float[torch.Tensor, 'seq batch num_slots determ']
    stoch_logits: Float[torch.Tensor, 'seq batch num_slots latent_classes latent_dim']
    stoch_: t.Optional[Bool[torch.Tensor, 'seq batch num_slots stoch_dim']] = None
    pos_enc: t.Optional[Float[torch.Tensor, '1 1 num_slots stoch_dim+determ']] = None

    def flatten(self):
        return State(self.determ.flatten(0, 1).unsqueeze(0),
                     self.stoch_logits.flatten(0, 1).unsqueeze(0),
                     self.stoch_.flatten(0, 1).unsqueeze(0) if self.stoch_ is not None else None,
                     self.pos_enc.detach() if self.pos_enc is not None else None)

    def detach(self):
        return State(self.determ.detach(),
                     self.stoch_logits.detach(),
                     self.stoch_.detach() if self.stoch_ is not None else None,
                     self.pos_enc.detach() if self.pos_enc is not None else None)

    @property
    def combined(self):
        return self.combined_slots.flatten(2, 3)

    @property
    def combined_slots(self):
        state = torch.concat([self.determ, self.stoch], dim=-1)
        if self.pos_enc is not None:
            return state + self.pos_enc
        else:
            return state

    @property
    def stoch(self):
        if self.stoch_ is None:
            self.stoch_ = Dist(
                self.stoch_logits).rsample().reshape(self.stoch_logits.shape[:3] + (-1, ))
        return self.stoch_

    @property
    def stoch_dist(self):
        return Dist(self.stoch_logits)

    @classmethod
    def stack(cls, states: list['State'], dim=0):
        if states[0].stoch_ is not None:
            stochs = torch.cat([state.stoch for state in states], dim=dim)
        else:
            stochs = None
        return State(torch.cat([state.determ for state in states], dim=dim),
                     torch.cat([state.stoch_logits for state in states], dim=dim),
                     stochs,
                     states[0].pos_enc)


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, norm=False, update_bias=-1, **kwargs):
        super().__init__()
        self._size = hidden_size
        self._act = torch.tanh
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(input_size + hidden_size,
                                3 * hidden_size,
                                bias=norm is not None,
                                **kwargs)
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

    def __init__(self,
                 latent_dim,
                 hidden_size,
                 actions_num,
                 latent_classes,
                 discrete_rssm,
                 norm_layer: nn.LayerNorm | nn.Identity,
                 slots_num: int,
                 embed_size=2 * 2 * 384):
        super().__init__()
        self.slots_num = slots_num
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
            nn.ELU(inplace=True))
        self.determ_recurrent = GRUCell(input_size=hidden_size*slots_num,
                                        hidden_size=hidden_size*slots_num,
                                        norm=True)  # Dreamer gru '_cell'

        # Calculate stochastic state from prior embed
        # shared between all ensemble models
        self.ensemble_prior_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),  # Dreamer 'img_out_{k}'
                norm_layer(hidden_size),
                nn.ELU(inplace=True),
                nn.Linear(hidden_size,
                          latent_dim * self.latent_classes),  # Dreamer 'img_dist_{k}'
                View((1, -1, latent_dim, self.latent_classes)))
            for _ in range(self.ensemble_num)
        ])

        img_sz = embed_size
        self.stoch_net = nn.Sequential(
            # nn.LayerNorm(hidden_size + img_sz, hidden_size),
            nn.Linear(hidden_size + img_sz, hidden_size),  # Dreamer 'obs_out'
            norm_layer(hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size,
                      latent_dim * self.latent_classes),  # Dreamer 'obs_dist'
            View((1, -1, latent_dim, self.latent_classes)))

    def on_train_step(self):
        pass

    def estimate_stochastic_latent(self, prev_determ: torch.Tensor):
        dists_per_model = [model(prev_determ) for model in self.ensemble_prior_estimator]
        # NOTE: Maybe something smarter can be used instead of
        #       taking only one random between all ensembles
        # NOTE: in Dreamer ensemble_num is always 1
        idx = torch.randint(0, self.ensemble_num, ())
        return dists_per_model[0]

    def predict_next(self, prev_state: State, action) -> State:
        x = self.pre_determ_recurrent(
            torch.concat([
                prev_state.stoch + prev_state.pos_enc[:, :, :, -prev_state.stoch.shape[-1]:],
                action.unsqueeze(2).repeat((1, 1, prev_state.determ.shape[2], 1))
            ],
                         dim=-1))
        # NOTE: x and determ are actually the same value if sequence of 1 is inserted
        x, determ_prior = self.determ_recurrent(x.flatten(2, 3),
                                                prev_state.determ.flatten(2, 3))
        if self.discrete_rssm:
            raise NotImplementedError("discrete rssm was not adopted for slot attention")
        else:
            determ_post, diff = determ_prior, 0

        # used for KL divergence
        # TODO: Test both options (with slot in batch size and in feature dim)
        predicted_stoch_logits = self.estimate_stochastic_latent(x.reshape(prev_state.determ.shape) + prev_state.pos_enc[:, :, :, :-prev_state.stoch.shape[-1]])
        # Size is 1 x B x slots_num x ...
        return State(determ_post.reshape(prev_state.determ.shape),
                     predicted_stoch_logits.reshape(prev_state.stoch_logits.shape),
                     pos_enc=prev_state.pos_enc), diff

    def update_current(self, prior: State, embed) -> State:  # Dreamer 'obs_out'
        return State(
            prior.determ,
            self.stoch_net(torch.concat([prior.determ, embed], dim=-1)).flatten(
                1, 2).reshape(prior.stoch_logits.shape), pos_enc=prior.pos_enc)

    def forward(self, h_prev: State, embed, action) -> tuple[State, State]:
        """
            'h' <- internal state of the world
            'z' <- latent embedding of current observation
            'a' <- action taken on prev step
            Returns 'h_next' <- the next next of the world
        """
        prior, diff = self.predict_next(h_prev, action)
        posterior = self.update_current(prior, embed)

        return prior, posterior, diff


