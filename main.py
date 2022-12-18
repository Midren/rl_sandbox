import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from rl_sandbox.agents.random_agent import RandomAgent
from rl_sandbox.agents.explorative_agent import ExplorativeAgent
from rl_sandbox.metrics import MetricsEvaluator
from rl_sandbox.utils.env import Env
from rl_sandbox.utils.replay_buffer import ReplayBuffer
from rl_sandbox.utils.persistent_replay_buffer import PersistentReplayBuffer
from rl_sandbox.utils.rollout_generation import (collect_rollout, collect_rollout_num, iter_rollout, iter_rollout_async,
                                                 fillup_replay_buffer)
from rl_sandbox.utils.schedulers import LinearScheduler


@hydra.main(version_base="1.2", config_path='config', config_name='config')
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    torch.distributions.Distribution.set_default_validate_args(False)
    torch.backends.cudnn.benchmark = True

    env: Env = hydra.utils.instantiate(cfg.env)

    # TODO: add replay buffer implementation, which stores rollouts
    #       on disk
    buff = ReplayBuffer()
    fillup_replay_buffer(env, buff, max(cfg.training.prefill, cfg.training.batch_size))

    metrics_evaluator = MetricsEvaluator()

    # TODO: Implement smarter techniques for exploration
    #       (Plan2Explore, etc)

    policy_agent = hydra.utils.instantiate(cfg.agent,
                            obs_space_num=env.observation_space.shape[0],
                            # FIXME: feels bad
                            actions_num=(env.action_space.high - env.action_space.low + 1).item(),
                            device_type=cfg.device_type)
    agent = ExplorativeAgent(
               policy_agent,
                # TODO: For dreamer, add noise for sampling instead
                # of just random actions
               RandomAgent(env),
               LinearScheduler(0.9, 0.01, 5_000))
    writer = SummaryWriter()

    prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('runs/profile_dreamer'),
                 schedule=torch.profiler.schedule(wait=10, warmup=10, active=5, repeat=5),
                 with_stack=True) if cfg.debug.profiler else None

    for i in tqdm(range(cfg.training.pretrain), desc='Pretraining'):
        s, a, r, n, f = buff.sample(cfg.training.batch_size,
                                    cluster_size=cfg.agent.get('batch_cluster_size', 1))
        losses = agent.train(s, a, r, n, f)
        for loss_name, loss in losses.items():
            writer.add_scalar(f'pre_train/{loss_name}', loss, i)

    global_step = 0
    pbar = tqdm(total=cfg.training.steps, desc='Training')
    while global_step < cfg.training.steps:
        ### Training and exploration

        # TODO: add buffer end prioritarization
        for s, a, r, n, f, _ in iter_rollout(env, agent):
            buff.add_sample(s, a, r, n, f)

            if global_step % cfg.training.gradient_steps_per_step == 0:
                # NOTE: unintuitive that batch_size is now number of total
                #       samples, but not amount of sequences for recurrent model
                s, a, r, n, f = buff.sample(cfg.training.batch_size,
                                            cluster_size=cfg.agent.get('batch_cluster_size', 1))

                losses = agent.train(s, a, r, n, f)
                if cfg.debug.profiler:
                    prof.step()
                for loss_name, loss in losses.items():
                    writer.add_scalar(f'train/{loss_name}', loss, global_step)
            global_step += cfg.env.repeat_action_num
            pbar.update(cfg.env.repeat_action_num)

        ### Validation
        if global_step % cfg.training.val_logs_every == 0:
            rollouts = collect_rollout_num(env, cfg.validation.rollout_num, agent)
            # TODO: make logs visualization in separate process
            metrics = metrics_evaluator.calculate_metrics(rollouts)
            for metric_name, metric in metrics.items():
                writer.add_scalar(f'val/{metric_name}', metric, global_step)

            if cfg.validation.visualize:
                rollouts = collect_rollout_num(env, 1, agent, collect_obs=True)

                for rollout in rollouts:
                    video = np.expand_dims(rollout.observations.transpose(0, 3, 1, 2), 0)
                    writer.add_video('val/visualization', video, global_step)
                    # FIXME: Very bad from architecture point
                    agent.policy_ag.viz_log(rollout, writer, global_step)

        ### Checkpoint
        if global_step % cfg.training.save_checkpoint_every == 0:
            agent.save_ckpt(global_step, losses)
    if cfg.debug.profiler:
        prof.stop()


if __name__ == "__main__":
    main()

