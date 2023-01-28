import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import random

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import lovely_tensors as lt

from rl_sandbox.agents.random_agent import RandomAgent
from rl_sandbox.agents.explorative_agent import ExplorativeAgent
from rl_sandbox.metrics import MetricsEvaluator
from rl_sandbox.utils.env import Env
from rl_sandbox.utils.replay_buffer import ReplayBuffer
from rl_sandbox.utils.persistent_replay_buffer import PersistentReplayBuffer
from rl_sandbox.utils.rollout_generation import (collect_rollout, collect_rollout_num, iter_rollout,
                                                 fillup_replay_buffer)
from rl_sandbox.utils.schedulers import LinearScheduler


class SummaryWriterMock():
    def add_scalar(*args, **kwargs):
        pass

    def add_video(*args, **kwargs):
        pass

    def add_image(*args, **kwargs):
        pass


@hydra.main(version_base="1.2", config_path='config', config_name='config')
def main(cfg: DictConfig):
    lt.monkey_patch()
    # print(OmegaConf.to_yaml(cfg))
    torch.distributions.Distribution.set_default_validate_args(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env: Env = hydra.utils.instantiate(cfg.env)

    buff = ReplayBuffer()
    fillup_replay_buffer(env, buff, max(cfg.training.prefill, cfg.training.batch_size))

    metrics_evaluator = MetricsEvaluator()

    # TODO: Implement smarter techniques for exploration
    #       (Plan2Explore, etc)
    agent = hydra.utils.instantiate(cfg.agent,
                            obs_space_num=env.observation_space.shape[0],
                            # FIXME: feels bad
                            # actions_num=(env.action_space.high - env.action_space.low + 1).item(),
                            # FIXME: currently only continuous tasks
                            actions_num=env.action_space.shape[0],
                            action_type='continuous',
                            device_type=cfg.device_type)

    writer = SummaryWriter()

    prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('runs/profile_dreamer'),
                 schedule=torch.profiler.schedule(wait=10, warmup=10, active=5, repeat=5),
                 with_stack=True) if cfg.debug.profiler else None

    for i in tqdm(range(int(cfg.training.pretrain)), desc='Pretraining'):
        s, a, r, n, f, first = buff.sample(cfg.training.batch_size,
                                    cluster_size=cfg.agent.get('batch_cluster_size', 1))
        losses = agent.train(s, a, r, n, f, first)
        for loss_name, loss in losses.items():
            writer.add_scalar(f'pre_train/{loss_name}', loss, i)

        log_every_n = 25
        st = int(cfg.training.pretrain) // log_every_n
        # FIXME: extract logging to seperate entity to omit
        # copy-paste
        if i % log_every_n == 0:
            rollouts = collect_rollout_num(env, cfg.validation.rollout_num, agent)
            # TODO: make logs visualization in separate process
            metrics = metrics_evaluator.calculate_metrics(rollouts)
            for metric_name, metric in metrics.items():
                writer.add_scalar(f'val/{metric_name}', metric, -st + i/log_every_n)

            if cfg.validation.visualize:
                rollouts = collect_rollout_num(env, 1, agent, collect_obs=True)

                for rollout in rollouts:
                    video = np.expand_dims(rollout.observations.transpose(0, 3, 1, 2), 0)
                    writer.add_video('val/visualization', video, -st + i/log_every_n)
                    # FIXME: Very bad from architecture point
                    agent.viz_log(rollout, writer, -st + i/log_every_n)

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
                s, a, r, n, f, first = buff.sample(cfg.training.batch_size,
                                            cluster_size=cfg.agent.get('batch_cluster_size', 1))

                losses = agent.train(s, a, r, n, f, first)
                if cfg.debug.profiler:
                    prof.step()
                # NOTE: Do not forget to run test with every step to check for outliers
                if global_step % 10 == 0:
                    for loss_name, loss in losses.items():
                        writer.add_scalar(f'train/{loss_name}', loss, global_step)
            global_step += cfg.env.repeat_action_num
            pbar.update(cfg.env.repeat_action_num)

        # FIXME: Currently works only val_logs_every is multiplier of amount of steps per rollout
        ### Validation
        if global_step % cfg.training.val_logs_every == 0:
            with torch.no_grad():
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
                    with torch.no_grad():
                        agent.viz_log(rollout, writer, global_step)

        ### Checkpoint
        if global_step % cfg.training.save_checkpoint_every == 0:
            agent.save_ckpt(global_step, losses)
    if cfg.debug.profiler:
        prof.stop()


if __name__ == "__main__":
    main()

