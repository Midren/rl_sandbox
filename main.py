import hydra
from omegaconf import DictConfig, OmegaConf

from rl_sandbox.agents.dqn_agent import DqnAgent
from rl_sandbox.utils.replay_buffer import ReplayBuffer
from rl_sandbox.utils.rollout_generation import collect_rollout, fillup_replay_buffer, collect_rollout_num
from rl_sandbox.utils.schedulers import LinearScheduler

from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

import gym

@hydra.main(version_base="1.2", config_path='config', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    env = gym.make(cfg.env)
    visualized_env = gym.make(cfg.env, render_mode='rgb_array_list')

    buff = ReplayBuffer()
    fillup_replay_buffer(env, buff, cfg.training.batch_size)

    # INFO: currently supports only discrete action space
    agent_params = {**cfg.agent}
    agent_name = agent_params.pop('name')
    agent = DqnAgent(obs_space_num=env.observation_space.shape[0],
                     actions_num=env.action_space.n,
                     device_type=cfg.device_type,
                     **agent_params,
                     )

    writer = SummaryWriter()

    scheduler = LinearScheduler(0.9, 0.01, 5_000)

    global_step = 0
    for epoch_num in range(cfg.training.epochs):
        ### Training and exploration
        state, _ = env.reset()
        terminated = False
        while not terminated:
            if np.random.random() > scheduler.step():
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)
            new_state, reward, terminated, _, _ = env.step(action)
            buff.add_sample(state, action, reward, new_state, terminated)

            s, a, r, n, f = buff.sample(cfg.training.batch_size)

            loss = agent.train(s, a, r, n, f)
            writer.add_scalar('train/loss', loss, global_step)
            global_step += 1

        ### Validation
        if epoch_num % 100 == 0:
            rollouts = collect_rollout_num(env, cfg.validation.rollout_num, agent)
            average_len = np.mean(list(map(lambda x: len(x.states), rollouts)))
            writer.add_scalar('val/average_len', average_len, epoch_num)

            if cfg.validation.visualize:
                rollouts = collect_rollout_num(visualized_env, 1, agent, save_obs=True)

                for rollout in rollouts:
                    video = np.expand_dims(rollout.observations.transpose(0, 3, 1, 2), 0)
                    writer.add_video('val/visualization', video, epoch_num)


if __name__ == "__main__":
    main()
