import gym
import hydra
import numpy as np
from dm_control import suite
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from rl_sandbox.agents.dqn_agent import DqnAgent
from rl_sandbox.agents.random_agent import RandomAgent
from rl_sandbox.metrics import MetricsEvaluator
from rl_sandbox.utils.dm_control import ActionDiscritizer, decode_dm_ts
from rl_sandbox.utils.replay_buffer import ReplayBuffer
from rl_sandbox.utils.rollout_generation import (collect_rollout,
                                                 collect_rollout_num,
                                                 fillup_replay_buffer)
from rl_sandbox.utils.schedulers import LinearScheduler


@hydra.main(version_base="1.2", config_path='config', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    match cfg.env.type:
        case "dm_control":
            env = suite.load(domain_name=cfg.env.domain_name,
                             task_name=cfg.env.task_name)
            visualized_env = env
        case "gym":
            env = gym.make(cfg.env)
            visualized_env = gym.make(cfg.env, render_mode='rgb_array_list')
        case _:
            raise RuntimeError("Invalid environment type")

    buff = ReplayBuffer()
    fillup_replay_buffer(env, buff, cfg.training.batch_size)

    agent_params = {**cfg.agent}
    agent_name = agent_params.pop('name')
    action_disritizer = ActionDiscritizer(env.action_spec(), values_per_dim=10)
    metrics_evaluator = MetricsEvaluator()

    match cfg.env.type:
        case "dm_control":
            obs_space_num = sum([v.shape[0] for v in env.observation_spec().values()])
        case "gym":
            obs_space_num = env.observation_space.shape[0]

    exploration_agent = RandomAgent(env)
    agent = DqnAgent(obs_space_num=obs_space_num,
                     actions_num=action_disritizer.shape,
                     # actions_num=env.action_space.n,
                     device_type=cfg.device_type,
                     **agent_params,
                     )

    writer = SummaryWriter()

    scheduler = LinearScheduler(0.9, 0.01, 5_000)

    global_step = 0
    for epoch_num in tqdm(range(cfg.training.epochs)):
        ### Training and exploration

        match cfg.env.type:
            case "dm_control":
                state, _, _ = decode_dm_ts(env.reset())
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
                case "gym":
                    new_state, reward, terminated, _, _ = env.step(action)
                    action = action_disritizer.undiscretize(action)

            buff.add_sample(state, action, reward, new_state, terminated)

            s, a, r, n, f = buff.sample(cfg.training.batch_size)
            a = np.stack([action_disritizer.discretize(a_) for a_ in a]).reshape(-1, 1)

            loss = agent.train(s, a, r, n, f)
            writer.add_scalar('train/loss', loss, global_step)
            global_step += 1

        ### Validation
        if epoch_num % 100 == 0:
            rollouts = collect_rollout_num(env, cfg.validation.rollout_num, agent)
            metrics = metrics_evaluator.calculate_metrics(rollouts)
            for metric_name, metric in metrics.items():
                writer.add_scalar(f'val/{metric_name}', metric, epoch_num)

            if cfg.validation.visualize:
                rollouts = collect_rollout_num(visualized_env, 1, agent, save_obs=True)

                for rollout in rollouts:
                    video = np.expand_dims(rollout.observations.transpose(0, 3, 1, 2), 0)
                    writer.add_video('val/visualization', video, epoch_num)


if __name__ == "__main__":
    main()
