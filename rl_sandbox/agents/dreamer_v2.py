import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision as tv

from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.fc_nn import fc_nn_generator
from rl_sandbox.utils.replay_buffer import (Action, Actions, Observation,
                                            Observations, Rewards,
                                            TerminationFlags, IsFirstFlags)
from rl_sandbox.utils.optimizer import Optimizer

from rl_sandbox.agents.dreamer import State
from rl_sandbox.agents.dreamer.world_model import WorldModel
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
            device_type: str = 'cpu',
            logger = None):

        self.logger = logger
        self.device = device_type
        self.imagination_horizon = imagination_horizon
        self.actions_num = actions_num
        self.is_discrete = (action_type != 'continuous')

        self.world_model: WorldModel = world_model(actions_num=actions_num).to(device_type)
        self.actor: ImaginativeActor = actor(latent_dim=self.world_model.state_size,
                                                  actions_num=actions_num,
                                                  is_discrete=self.is_discrete).to(device_type)
        self.critic: ImaginativeCritic = critic(latent_dim=self.world_model.state_size).to(device_type)

        self.world_model_optimizer = wm_optim(model=self.world_model)
        self.image_predictor_optimizer = wm_optim(model=self.world_model.image_predictor)
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
            losses_wm, discovered_states, metrics_wm = self.world_model.calculate_loss(obs, a, r, discount_factors, first_flags)
            self.world_model.recurrent_model.discretizer_scheduler.step()

        if self.world_model.decode_vit and self.world_model.vit_l2_ratio == 1.0:
            self.image_predictor_optimizer.step(losses_wm['loss_reconstruction_img'])


        metrics_wm |= self.world_model_optimizer.step(losses_wm['loss_wm'])

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
