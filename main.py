import gym
import hydra
import numpy as np
from dm_control import suite
from omegaconf import DictConfig, OmegaConf
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from rl_sandbox.agents.random_agent import RandomAgent
from rl_sandbox.metrics import MetricsEvaluator
from rl_sandbox.utils.dm_control import ActionDiscritizer, decode_dm_ts
from rl_sandbox.utils.replay_buffer import ReplayBuffer
from rl_sandbox.utils.rollout_generation import (collect_rollout_num,
                                                 fillup_replay_buffer)
from rl_sandbox.utils.schedulers import LinearScheduler


@hydra.main(version_base="1.2", config_path='config', config_name='config')
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))

    match cfg.env.type:
        case "dm_control":
            env = suite.load(domain_name=cfg.env.domain_name,
                             task_name=cfg.env.task_name)
            visualized_env = env
        case "gym":
            env = gym.make(cfg.env)
            visualized_env = gym.make(cfg.env, render_mode='rgb_array_list')
            if cfg.env.run_on_pixels:
                raise NotImplementedError("Run on pixels supported only for 'dm_control'")
        case _:
            raise RuntimeError("Invalid environment type")

    buff = ReplayBuffer()
    obs_res = cfg.env.obs_res if cfg.env.run_on_pixels else None
    fillup_replay_buffer(env, buff, cfg.training.batch_size, obs_res=obs_res, run_on_obs=cfg.env.run_on_pixels)

    action_disritizer = ActionDiscritizer(env.action_spec(), values_per_dim=10)
    metrics_evaluator = MetricsEvaluator()

    match cfg.env.type:
        case "dm_control":
            obs_space_num = sum([v.shape[0] for v in env.observation_spec().values()])
            if cfg.env.run_on_pixels:
                obs_space_num = (*cfg.env.obs_res, 3)
        case "gym":
            obs_space_num = env.observation_space.shape[0]

    exploration_agent = RandomAgent(env)
    # FIXME: currently action is 1 value, but not one-hot encoding
    agent = hydra.utils.instantiate(cfg.agent,
                            obs_space_num=obs_space_num,
                            actions_num=(1),
                            device_type=cfg.device_type)

    writer = SummaryWriter()

    scheduler = LinearScheduler(0.9, 0.01, 5_000)

    global_step = 0
    for epoch_num in tqdm(range(cfg.training.epochs)):
        ### Training and exploration

        match cfg.env.type:
            case "dm_control":
                state, _, _ = decode_dm_ts(env.reset())
                obs = env.physics.render(*cfg.env.obs_res, camera_id=0) if cfg.env.run_on_pixels else None
            case "gym":
                state, _ = env.reset()

        terminated = False
        while not terminated:
            if np.random.random() > scheduler.step():
                action = exploration_agent.get_action(state)
                action = action_disritizer.discretize(action)
            else:
                action = agent.get_action(state)

            match cfg.env.type:
                case "dm_control":
                    new_state, reward, terminated = decode_dm_ts(env.step(action_disritizer.undiscretize(action)))
                    new_obs = env.physics.render(*cfg.env.obs_res, camera_id=0) if cfg.env.run_on_pixels else None
                case "gym":
                    new_state, reward, terminated, _, _ = env.step(action)
                    action = action_disritizer.undiscretize(action)
                    obs = None

            if cfg.env.run_on_pixels:
                buff.add_sample(obs, action, reward, new_obs, terminated, obs)
            else:
                buff.add_sample(state, action, reward, new_state, terminated, obs)

            # NOTE: unintuitive that batch_size is now number of total
            #       samples, but not amount of sequences for recurrent model
            s, a, r, n, f = buff.sample(cfg.training.batch_size,
                                        return_observation=cfg.env.run_on_pixels,
                                        cluster_size=cfg.agent.get('batch_cluster_size', 1))

            losses = agent.train(s, a, r, n, f)
            if isinstance(losses, np.ndarray):
                writer.add_scalar('train/loss', loss, global_step)
            elif isinstance(losses, dict):
                for loss_name, loss in losses.items():
                    writer.add_scalar(f'train/{loss_name}', loss, global_step)
            else:
                raise RuntimeError("AAAA, very bad")
            global_step += 1

        ### Validation
        if epoch_num % 100 == 0:
            rollouts = collect_rollout_num(env, cfg.validation.rollout_num, agent)
            metrics = metrics_evaluator.calculate_metrics(rollouts)
            for metric_name, metric in metrics.items():
                writer.add_scalar(f'val/{metric_name}', metric, epoch_num)

            if cfg.validation.visualize:
                rollouts = collect_rollout_num(visualized_env, 1, agent, obs_res=cfg.obs_res)

                for rollout in rollouts:
                    video = np.expand_dims(rollout.observations.transpose(0, 3, 1, 2), 0)
                    writer.add_video('val/visualization', video, epoch_num)


if __name__ == "__main__":
    main()
