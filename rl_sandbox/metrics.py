import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
from torch.nn import functional as F
import torch

from rl_sandbox.utils.replay_buffer import Rollout
from rl_sandbox.utils.replay_buffer import (Action, Actions, Observation,
                                            Observations, Rewards,
                                            TerminationFlags, IsFirstFlags)


class EpisodeMetricsEvaluator():
    def __init__(self, agent: 'DreamerV2', log_video: bool = False):
        self.agent = agent
        self.episode = 0
        self.log_video = log_video

    def on_step(self, logger):
        pass

    def on_episode(self, logger):
        pass

    def on_val(self, logger, rollouts: list[Rollout], global_step: int):
        metrics = self.calculate_metrics(rollouts)
        logger.log(metrics, global_step, mode='val')
        if self.log_video:
            video = rollouts[0].obs.unsqueeze(0)
            logger.add_video('val/visualization', self.agent.unprocess_obs(video), global_step)
        self.episode += 1

    def calculate_metrics(self, rollouts: list[Rollout]):
        return {
                'episode_len': self._episode_duration(rollouts),
                'episode_return': self._episode_return(rollouts)
                }

    def _episode_duration(self, rollouts: list[Rollout]):
        return np.mean(list(map(lambda x: len(x.obs), rollouts)))

    def _episode_return(self, rollouts: list[Rollout]):
        return np.mean(list(map(lambda x: sum(x.rewards), rollouts)))

class DreamerMetricsEvaluator():
    def __init__(self, agent: 'DreamerV2'):
        self.agent = agent
        self.stored_steps = 0
        self.episode = 0

        if agent.is_discrete:
            pass

        self.reset_ep()

    def reset_ep(self):
        self._latent_probs = torch.zeros((self.agent.world_model.latent_classes, self.agent.world_model.latent_dim), device=self.agent.device)
        self._action_probs = torch.zeros((self.agent.actions_num), device=self.agent.device)
        self.stored_steps = 0

    def on_step(self, logger):
        self.stored_steps += 1

        if self.agent.is_discrete:
            self._action_probs += self._action_probs
        self._latent_probs += self.agent._state.stoch_dist.base_dist.probs.squeeze().mean(dim=0)

    def on_episode(self, logger):
        latent_hist = (self._latent_probs / self.stored_steps).detach().cpu().numpy()
        self.latent_hist = ((latent_hist / latent_hist.max() * 255.0 )).astype(np.uint8)

        self.reset_ep()
        self.episode += 1

    def on_val(self, logger, rollouts: list[Rollout], global_step: int):
        self.viz_log(rollouts[0], logger, global_step)

        if self.episode == 0:
            return

        # if discrete action space
        if self.agent.is_discrete:
            action_hist = (self.agent._action_probs / self.stored_steps).detach().cpu().numpy()
            fig = plt.Figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.bar(np.arange(self.agent.actions_num), action_hist)
            logger.add_figure('val/action_probs', fig, self.episode)
        else:
            # log mean +- std
            pass
        logger.add_image('val/latent_probs', self.latent_hist, global_step, dataformats='HW')
        logger.add_image('val/latent_probs_sorted', np.sort(self.latent_hist, axis=1), global_step, dataformats='HW')

    def _generate_video(self, obs: list[Observation], actions: list[Action], update_num: int):
        # obs = self.agent.preprocess_obs(obs)
        if self.agent.is_discrete:
            actions = F.one_hot(actions.to(torch.int64), num_classes=self.agent.actions_num).squeeze()
        video = []
        rews = []

        state = None
        for idx, (o, a) in enumerate(list(zip(obs, actions))):
            if idx > update_num:
                break
            state = self.agent.world_model.get_latent(o, a.unsqueeze(0).unsqueeze(0), state)
            video_r = self.agent.world_model.image_predictor(state.combined).mode
            rews.append(self.agent.world_model.reward_predictor(state.combined).mode.item())

            video.append(self.agent.unprocess_obs(video_r))

        rews = torch.Tensor(rews).to(obs.device)

        if update_num < len(obs):
            states, _, rews_2, _ = self.agent.imagine_trajectory(state, actions[update_num+1:].unsqueeze(1), horizon=self.agent.imagination_horizon - 1 - update_num)
            rews = torch.cat([rews, rews_2[1:].squeeze()])
            video_r = self.agent.world_model.image_predictor(states.combined[1:]).mode.detach()

            video.append(self.agent.unprocess_obs(video_r))

        return torch.cat(video), rews

    def viz_log(self, rollout, logger, epoch_num):
        rollout = rollout.to(device=self.agent.device)
        init_indeces = np.random.choice(len(rollout.obs) - self.agent.imagination_horizon, 5)

        videos = torch.cat([
            rollout.obs[init_idx:init_idx + self.agent.imagination_horizon] for init_idx in init_indeces
         ], dim=3)
        videos = self.agent.unprocess_obs(videos)

        real_rewards = [rollout.rewards[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces]

        videos_r, imagined_rewards = zip(*[self._generate_video(obs_0, a_0, update_num=self.agent.imagination_horizon//3) for obs_0, a_0 in zip(
                [rollout.obs[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces],
                [rollout.actions[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces])
        ])
        videos_r = torch.cat(videos_r, dim=3)

        videos_comparison = torch.cat([videos, videos_r, (torch.abs(videos.float() - videos_r.float() + 1)/2).to(dtype=torch.uint8)], dim=2).unsqueeze(0)

        logger.add_video('val/dreamed_rollout', videos_comparison, epoch_num)

        rewards_err = torch.Tensor([torch.abs(sum(imagined_rewards[i]) - real_rewards[i].sum()) for i in range(len(imagined_rewards))]).mean()
        logger.add_scalar('val/img_reward_err', rewards_err.item(), epoch_num)

        logger.add_scalar(f'val/reward', real_rewards[0].sum(), epoch_num)


class SlottedDreamerMetricsEvaluator(DreamerMetricsEvaluator):
    def on_step(self, logger):
        self.stored_steps += 1

        if self.agent.is_discrete:
            self._action_probs += self._action_probs
        self._latent_probs += self.agent._state[0].stoch_dist.base_dist.probs.squeeze().mean(dim=0)

    def on_episode(self, logger):
        wm = self.agent.world_model

        mu = wm.slot_attention.slots_mu
        sigma = wm.slot_attention.slots_logsigma.exp()
        self.mu_hist = torch.mean((mu - mu.squeeze(0).unsqueeze(1)) ** 2, dim=-1)
        self.sigma_hist = torch.mean((sigma - sigma.squeeze(0).unsqueeze(1)) ** 2, dim=-1)


        super().on_episode(logger)

    def on_val(self, logger, rollouts: list[Rollout], global_step: int):
        super().on_val(logger, rollouts, global_step)

        if self.episode == 0:
            return

        wm = self.agent.world_model

        if hasattr(wm.recurrent_model, 'last_attention'):
            logger.add_image('val/mixer_attention', wm.recurrent_model.last_attention, self.episode, dataformats='HW')

        logger.add_image('val/slot_attention_mu', self.mu_hist/self.mu_hist.max(), self.episode, dataformats='HW')
        logger.add_image('val/slot_attention_sigma', self.sigma_hist/self.sigma_hist.max(), self.episode, dataformats='HW')

        logger.add_scalar('val/slot_attention_mu_diff_max', self.mu_hist.max(), self.episode)
        logger.add_scalar('val/slot_attention_sigma_diff_max', self.sigma_hist.max(), self.episode)

    def _generate_video(self, obs: list[Observation], actions: list[Action], update_num: int):
        # obs = torch.from_numpy(obs.copy()).to(self.agent.device)
        # obs = self.agent.preprocess_obs(obs)
        # actions = self.agent.from_np(actions)
        if self.agent.is_discrete:
            actions = F.one_hot(actions.to(torch.int64), num_classes=self.agent.actions_num).squeeze()
        video = []
        slots_video = []
        rews = []

        state = None
        prev_slots = None
        for idx, (o, a) in enumerate(list(zip(obs, actions))):
            if idx > update_num:
                break
            state, prev_slots = self.agent.world_model.get_latent(o, a.unsqueeze(0).unsqueeze(0), (state, prev_slots))
            # video_r = self.agent.world_model.image_predictor(state.combined_slots).mode

            decoded_imgs, masks = self.agent.world_model.image_predictor(state.combined_slots.flatten(0, 1)).reshape(1, -1, 4, 64, 64).split([3, 1], dim=2)
            # TODO: try the scaling of softmax as in attention
            img_mask = self.agent.world_model.slot_mask(masks)
            decoded_imgs = decoded_imgs * img_mask
            video_r = torch.sum(decoded_imgs, dim=1)

            rews.append(self.agent.world_model.reward_predictor(state.combined).mode.item())
            video.append(self.agent.unprocess_obs(video_r))
            slots_video.append(self.agent.unprocess_obs(decoded_imgs))

        rews = torch.Tensor(rews).to(obs.device)

        if update_num < len(obs):
            states, _, rews_2, _ = self.agent.imagine_trajectory(state, actions[update_num+1:].unsqueeze(1), horizon=self.agent.imagination_horizon - 1 - update_num)
            rews = torch.cat([rews, rews_2[1:].squeeze()])

            # video_r = self.agent.world_model.image_predictor(states.combined_slots[1:]).mode
            decoded_imgs, masks = self.agent.world_model.image_predictor(states.combined_slots[1:].flatten(0, 1)).reshape(-1, self.agent.world_model.slots_num, 4, 64, 64).split([3, 1], dim=2)
            img_mask = self.agent.world_model.slot_mask(masks)
            decoded_imgs = decoded_imgs * img_mask
            video_r = torch.sum(decoded_imgs, dim=1)

            video.append(self.agent.unprocess_obs(video_r))
            slots_video.append(self.agent.unprocess_obs(decoded_imgs))

        return torch.cat(video), rews, torch.cat(slots_video)

    def viz_log(self, rollout, logger, epoch_num):
        rollout = rollout.to(device=self.agent.device)
        init_indeces = np.random.choice(len(rollout.obs) - self.agent.imagination_horizon, 5)

        videos = torch.cat([
            rollout.obs[init_idx:init_idx + self.agent.imagination_horizon] for init_idx in init_indeces
        ], dim=3)
        videos = self.agent.unprocess_obs(videos)

        real_rewards = [rollout.rewards[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces]

        videos_r, imagined_rewards, slots_video = zip(*[self._generate_video(obs_0, a_0, update_num=self.agent.imagination_horizon//3) for obs_0, a_0 in zip(
                [rollout.obs[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces],
                [rollout.actions[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces])
        ])
        videos_r = torch.cat(videos_r, dim=3)

        videos_comparison = torch.cat([videos, videos_r, (torch.abs(videos.float() - videos_r.float() + 1)/2).to(dtype=torch.uint8)], dim=2).unsqueeze(0)
        slots_video = torch.cat(list(slots_video)[:3], dim=3)

        slots_video = slots_video.permute((0, 2, 3, 1, 4))
        slots_video = slots_video.reshape(*slots_video.shape[:-2], -1).unsqueeze(0)

        logger.add_video('val/dreamed_rollout', videos_comparison, epoch_num)
        logger.add_video('val/dreamed_slots', slots_video, epoch_num)

        rewards_err = torch.Tensor([torch.abs(sum(imagined_rewards[i]) - real_rewards[i].sum()) for i in range(len(imagined_rewards))]).mean()
        logger.add_scalar('val/img_reward_err', rewards_err.item(), epoch_num)

        logger.add_scalar(f'val/reward', real_rewards[0].sum(), epoch_num)

class SlottedDinoDreamerMetricsEvaluator(SlottedDreamerMetricsEvaluator):
    def _generate_video(self, obs: list[Observation], actions: list[Action], update_num: int):
        # obs = torch.from_numpy(obs.copy()).to(self.agent.device)
        # obs = self.agent.preprocess_obs(obs)
        # actions = self.agent.from_np(actions)
        if self.agent.is_discrete:
            actions = F.one_hot(actions.to(torch.int64), num_classes=self.agent.actions_num).squeeze()
        video = []
        slots_video = []
        vit_slots_video = []
        rews = []

        vit_size = self.agent.world_model.vit_size

        state = None
        prev_slots = None
        for idx, (o, a) in enumerate(list(zip(obs, actions))):
            if idx > update_num:
                break
            state, prev_slots = self.agent.world_model.get_latent(o, a.unsqueeze(0).unsqueeze(0), (state, prev_slots))
            # video_r = self.agent.world_model.image_predictor(state.combined_slots).mode

            decoded_imgs, masks = self.agent.world_model.image_predictor(state.combined_slots.flatten(0, 1)).reshape(1, -1, 4, 64, 64).split([3, 1], dim=2)
            # TODO: try the scaling of softmax as in attention
            img_mask = self.agent.world_model.slot_mask(masks)
            decoded_imgs = decoded_imgs * img_mask
            video_r = torch.sum(decoded_imgs, dim=1)

            _, vit_masks = self.agent.world_model.dino_predictor(state.combined_slots.flatten(0, 1)).reshape(-1, self.agent.world_model.slots_num, self.agent.world_model.vit_feat_dim+1, vit_size, vit_size).split([self.agent.world_model.vit_feat_dim, 1], dim=2)
            vit_mask = F.softmax(vit_masks, dim=1)
            upscale = tv.transforms.Resize(64, antialias=True)

            upscaled_mask = upscale(vit_mask.permute(0, 1, 4, 2, 3).squeeze())
            per_slot_vit = (upscaled_mask.unsqueeze(1) * o.to(self.agent.device).unsqueeze(0)).unsqueeze(0)

            rews.append(self.agent.world_model.reward_predictor(state.combined).mode.item())
            video.append(self.agent.unprocess_obs(video_r))
            slots_video.append(self.agent.unprocess_obs(decoded_imgs))
            vit_slots_video.append(self.agent.unprocess_obs(per_slot_vit/upscaled_mask.max()))

        rews = torch.Tensor(rews).to(obs.device)

        if update_num < len(obs):
            states, _, rews_2, _ = self.agent.imagine_trajectory(state, actions[update_num+1:].unsqueeze(1), horizon=self.agent.imagination_horizon - 1 - update_num)
            rews = torch.cat([rews, rews_2[1:].squeeze()])

            # video_r = self.agent.world_model.image_predictor(states.combined_slots[1:]).mode
            decoded_imgs, masks = self.agent.world_model.image_predictor(states.combined_slots[1:].flatten(0, 1)).reshape(-1, self.agent.world_model.slots_num, 4, 64, 64).split([3, 1], dim=2)
            img_mask = self.agent.world_model.slot_mask(masks)
            decoded_imgs = decoded_imgs * img_mask
            video_r = torch.sum(decoded_imgs, dim=1)

            _, vit_masks = self.agent.world_model.dino_predictor(states.combined_slots[1:].flatten(0, 1)).reshape(-1, self.agent.world_model.slots_num, self.agent.world_model.vit_feat_dim+1, vit_size, vit_size).split([self.agent.world_model.vit_feat_dim, 1], dim=2)
            vit_mask = F.softmax(vit_masks, dim=1)
            upscale = tv.transforms.Resize(64, antialias=True)
            upscaled_mask = upscale(vit_mask.permute(0, 1, 4, 2, 3).squeeze())
            per_slot_vit = (upscaled_mask.unsqueeze(2) * obs[update_num+1:].to(self.agent.device).unsqueeze(1))
            # per_slot_vit = (upscaled_mask.unsqueeze(1) * o.to(self.agent.device).unsqueeze(0)).unsqueeze(0)

            video.append(self.agent.unprocess_obs(video_r))
            slots_video.append(self.agent.unprocess_obs(decoded_imgs))
            vit_slots_video.append(self.agent.unprocess_obs(per_slot_vit/torch.amax(upscaled_mask, dim=(1,2,3)).view(-1, 1, 1, 1, 1)))

        return torch.cat(video), rews, torch.cat(slots_video), torch.cat(vit_slots_video)

    def viz_log(self, rollout, logger, epoch_num):
        rollout = rollout.to(device=self.agent.device)
        init_indeces = np.random.choice(len(rollout.obs) - self.agent.imagination_horizon, 5)

        videos = torch.cat([
            rollout.obs[init_idx:init_idx + self.agent.imagination_horizon] for init_idx in init_indeces
        ], dim=3)
        videos = self.agent.unprocess_obs(videos)

        real_rewards = [rollout.rewards[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces]

        videos_r, imagined_rewards, slots_video, vit_masks_video = zip(*[self._generate_video(obs_0, a_0, update_num=self.agent.imagination_horizon//3) for obs_0, a_0 in zip(
                [rollout.obs[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces],
                [rollout.actions[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces])
        ])
        videos_r = torch.cat(videos_r, dim=3)


        videos_comparison = torch.cat([videos, videos_r, (torch.abs(videos.float() - videos_r.float() + 1)/2).to(dtype=torch.uint8)], dim=2).unsqueeze(0)

        slots_video = torch.cat(list(slots_video)[:3], dim=3)
        slots_video = slots_video.permute((0, 2, 3, 1, 4))
        slots_video = slots_video.reshape(*slots_video.shape[:-2], -1).unsqueeze(0)

        vit_masks_video = torch.cat(list(vit_masks_video)[:3], dim=3)
        vit_masks_video = vit_masks_video.permute((0, 2, 3, 1, 4))
        vit_masks_video = vit_masks_video.reshape(*vit_masks_video.shape[:-2], -1).unsqueeze(0)

        logger.add_video('val/dreamed_rollout', videos_comparison, epoch_num)
        logger.add_video('val/dreamed_slots', slots_video, epoch_num)
        logger.add_video('val/dreamed_vit_masks', vit_masks_video, epoch_num)

        # FIXME: rewrite sum(...) as (...).sum()
        rewards_err = torch.Tensor([torch.abs(sum(imagined_rewards[i]) - real_rewards[i].sum()) for i in range(len(imagined_rewards))]).mean()
        logger.add_scalar('val/img_reward_err', rewards_err.item(), epoch_num)

        logger.add_scalar(f'val/reward', real_rewards[0].sum(), epoch_num)
