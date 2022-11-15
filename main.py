import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from unpackable import unpack

from rl_sandbox.agents.random_agent import RandomAgent
from rl_sandbox.metrics import MetricsEvaluator
from rl_sandbox.utils.env import Env
from rl_sandbox.utils.replay_buffer import ReplayBuffer
from rl_sandbox.utils.rollout_generation import (collect_rollout_num,
                                                 fillup_replay_buffer)
from rl_sandbox.utils.schedulers import LinearScheduler


@hydra.main(version_base="1.2", config_path='config', config_name='config')
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))

    env: Env = hydra.utils.instantiate(cfg.env)

    # TODO: add replay buffer implementation, which stores rollouts
    #       on disk
    buff = ReplayBuffer()
    fillup_replay_buffer(env, buff, cfg.training.batch_size)

    metrics_evaluator = MetricsEvaluator()

    # TODO: Implement smarter techniques for exploration
    #       (Plan2Explore, etc)
    exploration_agent = RandomAgent(env)
    agent = hydra.utils.instantiate(cfg.agent,
                            obs_space_num=env.observation_space.shape[0],
                            # FIXME: feels bad
                            actions_num=(env.action_space.high - env.action_space.low + 1).item(),
                            device_type=cfg.device_type)

    writer = SummaryWriter()

    scheduler = LinearScheduler(0.9, 0.01, 5_000)

    global_step = 0
    for epoch_num in tqdm(range(cfg.training.epochs)):
        ### Training and exploration

        state, _, _ = unpack(env.reset())
        agent.reset()

        terminated = False
        while not terminated:
            if global_step % cfg.training.gradient_steps_per_step == 0:
                # TODO: For dreamer, add noise for sampling
                if np.random.random() > scheduler.step():
                    action = exploration_agent.get_action(state)
                else:
                    action = agent.get_action(state)

                new_state, reward, terminated = unpack(env.step(action))

                buff.add_sample(state, action, reward, new_state, terminated)

            # NOTE: unintuitive that batch_size is now number of total
            #       samples, but not amount of sequences for recurrent model
            s, a, r, n, f = buff.sample(cfg.training.batch_size,
                                        cluster_size=cfg.agent.get('batch_cluster_size', 1))

            losses = agent.train(s, a, r, n, f)
            for loss_name, loss in losses.items():
                writer.add_scalar(f'train/{loss_name}', loss, global_step)
            global_step += 1

        ### Validation
        if epoch_num % cfg.training.val_logs_every == 0:
            rollouts = collect_rollout_num(env, cfg.validation.rollout_num, agent)
            metrics = metrics_evaluator.calculate_metrics(rollouts)
            for metric_name, metric in metrics.items():
                writer.add_scalar(f'val/{metric_name}', metric, epoch_num)

            if cfg.validation.visualize:
                rollouts = collect_rollout_num(env, 1, agent, collect_obs=True)

                for rollout in rollouts:
                    video = np.expand_dims(rollout.observations.transpose(0, 3, 1, 2), 0)
                    writer.add_video('val/visualization', video, epoch_num)
                    # FIXME:Very bad from architecture point
                    agent.viz_log(rollout, writer, epoch_num)

        ### Checkpoint
        # if epoch_num % cfg.training.save_checkpoint_every == 0:
        #     agent.save_ckpt(epoch_num, losses)


if __name__ == "__main__":
    main()
