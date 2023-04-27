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

    def on_step(self):
        pass

    def on_episode(self, logger, rollout):
        pass

    def on_val(self, logger, rollouts: list[Rollout]):
        metrics = self.calculate_metrics(rollouts)
        logger.log(metrics, self.episode, mode='val')
        if self.log_video:
            video = np.expand_dims(rollouts[0].observations.transpose(0, 3, 1, 2), 0)
            logger.add_video('val/visualization', video, self.episode)
        self.episode += 1

    def calculate_metrics(self, rollouts: list[Rollout]):
        return {
                'episode_len': self._episode_duration(rollouts),
                'episode_return': self._episode_return(rollouts)
                }

    def _episode_duration(self, rollouts: list[Rollout]):
        return np.mean(list(map(lambda x: len(x.states), rollouts)))

    def _episode_return(self, rollouts: list[Rollout]):
        return np.mean(list(map(lambda x: sum(x.rewards), rollouts)))

class DreamerMetricsEvaluator():
    def __init__(self, agent: 'DreamerV2'):
        self.agent = agent
        self.stored_steps = 0
        self.episode = 0

        if agent.is_discrete:
            pass

    def reset_ep(self):
        self._latent_probs = torch.zeros((self.agent.world_model.latent_classes, self.agent.world_model.latent_dim), device=agent.device)
        self._action_probs = torch.zeros((self.agent.actions_num), device=self.agent.device)
        self.stored_steps = 0

    def on_step(self):
        self.stored_steps += 1

        if self.agent.is_discrete:
            self._action_probs += self._action_probs
        self._latent_probs += self.agent._state[0].stoch_dist.probs.squeeze().mean(dim=0)

    def on_episode(self, logger, rollout):
        latent_hist = (self.agent._latent_probs / self._stored_steps).detach().cpu().numpy()
        latent_hist = ((latent_hist / latent_hist.max() * 255.0 )).astype(np.uint8)

        # if discrete action space
        if self.agent.is_discrete:
            action_hist = (self.agent._action_probs / self._stored_steps).detach().cpu().numpy()
            fig = plt.Figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.bar(np.arange(self.agent.actions_num), action_hist)
            logger.add_figure('val/action_probs', fig, self.episode)
        else:
            # log mean +- std
            pass
        logger.add_image('val/latent_probs', latent_hist, self.episode, dataformats='HW')
        logger.add_image('val/latent_probs_sorted', np.sort(latent_hist, axis=1), self.episode, dataformats='HW')

        self.reset_ep()
        self.episode += 1

    def on_val(self, logger, rollouts: list[Rollout]):
        self.viz_log(rollouts[0], logger, self.episode)

    def _generate_video(self, obs: list[Observation], actions: list[Action], update_num: int):
        obs = torch.from_numpy(obs.copy()).to(self.agent.device)
        obs = self.agent.preprocess_obs(obs)
        actions = self.agent.from_np(actions)
        if self.agent.is_discrete:
            actions = F.one_hot(actions.to(torch.int64), num_classes=self.agent.actions_num).squeeze()
        video = []
        rews = []

        state = None
        means = np.array([0.485, 0.456, 0.406])
        stds = np.array([0.229, 0.224, 0.225])
        UnNormalize = tv.transforms.Normalize(list(-means/stds),
                                           list(1/stds))
        for idx, (o, a) in enumerate(list(zip(obs, actions))):
            if idx > update_num:
                break
            state = self.agent.world_model.get_latent(o, a.unsqueeze(0).unsqueeze(0), state)
            video_r = self.agent.world_model.image_predictor(state.combined).mode.cpu().detach().numpy()
            rews.append(self.agent.world_model.reward_predictor(state.combined).mode.item())
            if self.agent.world_model.encode_vit:
                video_r = UnNormalize(torch.from_numpy(video_r)).numpy()
            else:
                video_r = (video_r + 0.5)
            video.append(video_r)

        rews = torch.Tensor(rews).to(obs.device)

        if update_num < len(obs):
            states, _, rews_2, _ = self.agent.imagine_trajectory(state, actions[update_num+1:].unsqueeze(1), horizon=self.agent.imagination_horizon - 1 - update_num)
            rews = torch.cat([rews, rews_2[1:].squeeze()])
            video_r = self.agent.world_model.image_predictor(states.combined[1:]).mode.cpu().detach().numpy()
            if self.agent.world_model.encode_vit:
                video_r = UnNormalize(torch.from_numpy(video_r)).numpy()
            else:
                video_r = (video_r + 0.5)
            video.append(video_r)

        return np.concatenate(video), rews

    def viz_log(self, rollout, logger, epoch_num):
        init_indeces = np.random.choice(len(rollout.states) - self.agent.imagination_horizon, 5)

        videos = np.concatenate([
            rollout.next_states[init_idx:init_idx + self.agent.imagination_horizon].transpose(
                0, 3, 1, 2) for init_idx in init_indeces
        ], axis=3).astype(np.float32) / 255.0

        real_rewards = [rollout.rewards[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces]

        videos_r, imagined_rewards = zip(*[self._generate_video(obs_0.copy(), a_0, update_num=self.agent.imagination_horizon//3) for obs_0, a_0 in zip(
                [rollout.next_states[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces],
                [rollout.actions[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces])
        ])
        videos_r = np.concatenate(videos_r, axis=3)

        videos_comparison = np.expand_dims(np.concatenate([videos, videos_r, np.abs(videos - videos_r + 1)/2], axis=2), 0)
        videos_comparison = (videos_comparison * 255.0).astype(np.uint8)

        logger.add_video('val/dreamed_rollout', videos_comparison, epoch_num)

        rewards_err = torch.Tensor([torch.abs(sum(imagined_rewards[i]) - real_rewards[i].sum()) for i in range(len(imagined_rewards))]).mean()
        logger.add_scalar('val/img_reward_err', rewards_err.item(), epoch_num)

        logger.add_scalar(f'val/reward', real_rewards[0].sum(), epoch_num)


class SlottedDreamerMetricsEvaluator(DreamerMetricsEvaluator):
    def _generate_video(self, obs: list[Observation], actions: list[Action], update_num: int):
        obs = torch.from_numpy(obs.copy()).to(self.agent.device)
        obs = self.agent.preprocess_obs(obs)
        actions = self.agent.from_np(actions)
        if self.agent.is_discrete:
            actions = F.one_hot(actions.to(torch.int64), num_classes=self.agent.actions_num).squeeze()
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
            state, prev_slots = self.agent.world_model.get_latent(o, a.unsqueeze(0).unsqueeze(0), (state, prev_slots))
            # video_r = self.agent.world_model.image_predictor(state.combined_slots).mode.cpu().detach().numpy()

            decoded_imgs, masks = self.agent.world_model.image_predictor(state.combined_slots.flatten(0, 1)).reshape(1, -1, 4, 64, 64).split([3, 1], dim=2)
            # TODO: try the scaling of softmax as in attention
            img_mask = F.softmax(masks, dim=1)
            decoded_imgs = decoded_imgs * img_mask
            video_r = torch.sum(decoded_imgs, dim=1).cpu().detach().numpy()

            rews.append(self.agent.world_model.reward_predictor(state.combined).mode.item())
            if self.agent.world_model.encode_vit:
                video_r = UnNormalize(torch.from_numpy(video_r)).numpy()
            else:
                video_r = (video_r + 0.5)
            video.append(video_r)
            slots_video.append(decoded_imgs.cpu().detach().numpy() + 0.5)

        rews = torch.Tensor(rews).to(obs.device)

        if update_num < len(obs):
            states, _, rews_2, _ = self.agent.imagine_trajectory(state, actions[update_num+1:].unsqueeze(1), horizon=self.agent.imagination_horizon - 1 - update_num)
            rews = torch.cat([rews, rews_2[1:].squeeze()])

            # video_r = self.agent.world_model.image_predictor(states.combined_slots[1:]).mode.cpu().detach().numpy()
            decoded_imgs, masks = self.agent.world_model.image_predictor(states.combined_slots[1:].flatten(0, 1)).reshape(-1, self.agent.world_model.slots_num, 4, 64, 64).split([3, 1], dim=2)
            img_mask = F.softmax(masks, dim=1)
            decoded_imgs = decoded_imgs * img_mask
            video_r = torch.sum(decoded_imgs, dim=1).cpu().detach().numpy()

            if self.agent.world_model.encode_vit:
                video_r = UnNormalize(torch.from_numpy(video_r)).numpy()
            else:
                video_r = (video_r + 0.5)
            video.append(video_r)
            slots_video.append(decoded_imgs.cpu().detach().numpy() + 0.5)

        return np.concatenate(video), rews, np.concatenate(slots_video)

    def viz_log(self, rollout, logger, epoch_num):
        init_indeces = np.random.choice(len(rollout.states) - self.agent.imagination_horizon, 5)

        videos = np.concatenate([
            rollout.next_states[init_idx:init_idx + self.agent.imagination_horizon].transpose(
                0, 3, 1, 2) for init_idx in init_indeces
        ], axis=3).astype(np.float32) / 255.0

        real_rewards = [rollout.rewards[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces]

        videos_r, imagined_rewards, slots_video = zip(*[self._generate_video(obs_0.copy(), a_0, update_num=self.agent.imagination_horizon//3) for obs_0, a_0 in zip(
                [rollout.next_states[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces],
                [rollout.actions[idx:idx+ self.agent.imagination_horizon] for idx in init_indeces])
        ])
        videos_r = np.concatenate(videos_r, axis=3)

        slots_video = np.concatenate(list(slots_video)[:3], axis=3)
        slots_video = slots_video.transpose((0, 2, 3, 1, 4))
        slots_video = np.expand_dims(slots_video.reshape(*slots_video.shape[:-2], -1), 0)

        videos_comparison = np.expand_dims(np.concatenate([videos, videos_r, np.abs(videos - videos_r + 1)/2], axis=2), 0)
        videos_comparison = (videos_comparison * 255.0).astype(np.uint8)

        logger.add_video('val/dreamed_rollout', videos_comparison, epoch_num)
        logger.add_video('val/dreamed_slots', slots_video, epoch_num)

        rewards_err = torch.Tensor([torch.abs(sum(imagined_rewards[i]) - real_rewards[i].sum()) for i in range(len(imagined_rewards))]).mean()
        logger.add_scalar('val/img_reward_err', rewards_err.item(), epoch_num)

        logger.add_scalar(f'val/reward', real_rewards[0].sum(), epoch_num)
