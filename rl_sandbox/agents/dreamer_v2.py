import typing as t
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import torchvision as tv
from unpackable import unpack

from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.replay_buffer import (Action, Observation,
                                            RolloutChunks, EnvStep, Rollout)

from rl_sandbox.agents.dreamer.world_model import WorldModel, State
from rl_sandbox.agents.dreamer.ac import ImaginativeCritic, ImaginativeActor


class DreamerV2(RlAgent):

    def __init__(
            self,
            obs_space_num: list[int],  # NOTE: encoder/decoder will work only with 64x64 currently
            actions_num: int,
            world_model: t.Any,
            actor: t.Any,
            critic: t.Any,
            action_type: str,
            imagination_horizon: int,
            wm_optim: t.Any,
            actor_optim: t.Any,
            critic_optim: t.Any,
            layer_norm: bool,
            batch_cluster_size: int,
            f16_precision: bool,
            device_type: str = 'cpu',
            logger = None):

        self.logger = logger
        self.device = device_type
        self.imagination_horizon = imagination_horizon
        self.actions_num = actions_num
        self.is_discrete = (action_type != 'continuous')
        self.is_f16 = f16_precision

        self.world_model: WorldModel = world_model(actions_num=actions_num).to(device_type)
        self.actor: ImaginativeActor = actor(latent_dim=self.world_model.state_size,
                                                  actions_num=actions_num,
                                                  is_discrete=self.is_discrete).to(device_type)
        self.critic: ImaginativeCritic = critic(latent_dim=self.world_model.state_size).to(device_type)

        self.world_model_optimizer = wm_optim(model=self.world_model, scaler=self.is_f16)
        self.image_predictor_optimizer = wm_optim(model=self.world_model.image_predictor, scaler=self.is_f16)
        self.actor_optimizer = actor_optim(model=self.actor)
        self.critic_optimizer = critic_optim(model=self.critic)

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

        return (states[0].stack(states), torch.cat(actions), torch.cat(rewards), torch.cat(ts))

    def reset(self):
        self._state = self.world_model.get_initial_state()
        self._last_action = torch.zeros((1, 1, self.actions_num), device=self.device)
        self._action_probs = torch.zeros((self.actions_num), device=self.device)

    def preprocess(self, rollout: Rollout):
        obs = self.preprocess_obs(rollout.obs)
        additional = self.world_model.precalc_data(obs.to(self.device))
        return Rollout(obs=obs,
                       actions=rollout.actions,
                       rewards=rollout.rewards,
                       is_finished=rollout.is_finished,
                       is_first=rollout.is_first,
                       additional_data=rollout.additional_data | additional)

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
        obs = torch.from_numpy(obs).to(self.device)
        obs = self.preprocess_obs(obs)

        self._state = self.world_model.get_latent(obs, self._last_action, self._state)

        actor_dist = self.actor.get_action(self._state)
        self._last_action = actor_dist.sample()

        if self.is_discrete:
            self._action_probs += actor_dist.probs.squeeze()

        if self.is_discrete:
            return self._last_action.argmax()
        else:
            return self._last_action.squeeze().detach().cpu().numpy()

    def from_np(self, arr: np.ndarray):
        arr = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
        return arr.to(self.device, non_blocking=True)

    def train(self, rollout_chunks: RolloutChunks):
        obs, a, r, is_finished, is_first, additional = unpack(rollout_chunks)
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        # obs = self.preprocess_obs(self.from_np(obs))
        if self.is_discrete:
            a = F.one_hot(a.to(torch.int64), num_classes=self.actions_num).squeeze()
        discount_factors = (1 - is_finished).float()
        first_flags = is_first.float()

        # take some latent embeddings as initial
        with torch.cuda.amp.autocast(enabled=self.is_f16):
            losses_wm, discovered_states, metrics_wm = self.world_model.calculate_loss(obs, a, r, discount_factors, first_flags, additional)
            # FIXME: wholely remove discrete RSSM
            # self.world_model.recurrent_model.discretizer_scheduler.step()

        if self.world_model.decode_vit and self.world_model.vit_l2_ratio == 1.0:
            self.image_predictor_optimizer.step(losses_wm['loss_reconstruction_img'])


        metrics_wm |= self.world_model_optimizer.step(losses_wm['loss_wm'])

        with torch.cuda.amp.autocast(enabled=self.is_f16):
            initial_states = discovered_states.__class__(discovered_states.determ.flatten(0, 1).unsqueeze(0).detach(),
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

            losses_c, metrics_c = self.critic.calculate_loss(zs[:-1], vs, discount_factors[:-1])

            # last action should be ignored as it is not used to predict next state, thus no feedback
            # first value should be ignored as it is comes from replay buffer
            losses_a, metrics_a  = self.actor.calculate_loss(zs[:-2],
                                                             vs[1:],
                                                             self.critic.target_critic(zs[:-2]).mode,
                                                             discount_factors[:-2],
                                                             actions[1:-1])
        metrics_a |= self.actor_optimizer.step(losses_a['loss_actor'])
        metrics_c |= self.critic_optimizer.step(losses_c['loss_critic'])

        self.critic.update_target()

        losses = losses_wm | losses_a | losses_c
        metrics = metrics_wm | metrics_a | metrics_c

        losses = {l: val.detach().cpu().numpy() for l, val in losses.items()}
        metrics = {l: val.detach().cpu().numpy() for l, val in metrics.items()}

        losses['total'] = sum(losses.values())
        return losses | metrics

    def save_ckpt(self, epoch_num: int, losses: dict[str, float]):
        torch.save(
            {
                'epoch': epoch_num,
                'world_model_state_dict': self.world_model.state_dict(),
                'world_model_optimizer_state_dict': self.world_model_optimizer.optimizer.state_dict(),
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.optimizer.state_dict(),
                'losses': losses
            }, f'dreamerV2-{epoch_num}-{losses["total"]}.ckpt')

    def load_ckpt(self, ckpt_path: Path):
        ckpt = torch.load(ckpt_path)
        self.world_model.load_state_dict(ckpt['world_model_state_dict'])
        # FIXME: doesn't work for optimizers
        self.world_model_optimizer.load_state_dict(
            ckpt['world_model_optimizer_state_dict'])
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])
        return ckpt['epoch']
