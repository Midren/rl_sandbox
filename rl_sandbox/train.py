import random
import os
os.environ['MUJOCO_GL'] = 'egl'

import crafter
import hydra
import lovely_tensors as lt
import numpy as np
import torch
from gym.spaces import Discrete
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

from rl_sandbox.utils.env import Env
from rl_sandbox.utils.logger import Logger
from rl_sandbox.utils.replay_buffer import ReplayBuffer
from rl_sandbox.utils.rollout_generation import (collect_rollout_num,
                                                 fillup_replay_buffer,
                                                 iter_rollout)


def val_logs(agent, val_cfg: DictConfig, metrics, env: Env, logger: Logger):
    with torch.no_grad():
        rollouts = collect_rollout_num(env, val_cfg.rollout_num, agent, collect_obs=True)
        rollouts = [agent.preprocess(r) for r in rollouts]

    for metric in metrics:
        metric.on_val(logger, rollouts)


@hydra.main(version_base="1.2", config_path='config', config_name='config')
def main(cfg: DictConfig):
    lt.monkey_patch()
    torch.distributions.Distribution.set_default_validate_args(False)
    eval('setattr(torch.backends.cudnn, "benchmark", True)') # need to be pickable for multirun
    torch.backends.cuda.matmul.allow_tf32 = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if HydraConfig.get()['mode'] == RunMode.MULTIRUN and cfg.device_type == 'cuda':
        num_gpus = torch.cuda.device_count()
        gpu_id = HydraConfig.get().job.num % num_gpus
        cfg.device_type = f'cuda:{gpu_id}'
        cfg.logger.message += "," + ",".join(HydraConfig.get()['overrides']['task'])

    # TODO: Implement smarter techniques for exploration
    #       (Plan2Explore, etc)
    print(f'Start run: {cfg.logger.message}')
    logger = Logger(**cfg.logger)

    env: Env = hydra.utils.instantiate(cfg.env)
    val_env: Env = hydra.utils.instantiate(cfg.env)
    # TOOD: Create maybe some additional validation env
    if cfg.env.task_name.startswith("Crafter"):
        val_env.env = crafter.Recorder(val_env.env,
                                       logger.log_dir(),
                                       save_stats=True,
                                       save_video=False,
                                       save_episode=False)

    is_discrete = isinstance(env.action_space, Discrete)
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_space_num=env.observation_space.shape,
        actions_num=env.action_space.n if is_discrete else env.action_space.shape[0],
        action_type='discrete' if is_discrete else 'continuous',
        device_type=cfg.device_type,
        f16_precision=cfg.training.f16_precision,
        logger=logger)

    buff = ReplayBuffer(prioritize_ends=cfg.training.prioritize_ends,
                        min_ep_len=cfg.agent.get('batch_cluster_size', 1) *
                        (cfg.training.prioritize_ends + 1),
                        preprocess_func=agent.preprocess,
                        device = cfg.device_type)

    fillup_replay_buffer(
        env, buff,
        max(cfg.training.prefill,
            cfg.training.batch_size * cfg.agent.get('batch_cluster_size', 1)),
        agent=agent)

    metrics = [metric(agent) for metric in hydra.utils.instantiate(cfg.validation.metrics)]

    prof = profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(logger.log_dir() + '/profiler'),
        schedule=torch.profiler.schedule(wait=10, warmup=10, active=5, repeat=5),
        with_stack=True) if cfg.debug.profiler else None

    for i in tqdm(range(int(cfg.training.pretrain)), desc='Pretraining'):
        if cfg.training.checkpoint_path is not None:
            break
        rollout_chunks = buff.sample(cfg.training.batch_size,
                                           cluster_size=cfg.agent.get(
                                               'batch_cluster_size', 1))
        losses = agent.train(rollout_chunks)
        logger.log(losses, i, mode='pre_train')

    val_logs(agent, cfg.validation, metrics, val_env, logger)

    if cfg.training.checkpoint_path is not None:
        prev_global_step = global_step = agent.load_ckpt(cfg.training.checkpoint_path)
    else:
        prev_global_step = global_step = 0

    pbar = tqdm(total=cfg.training.steps, desc='Training')
    while global_step < cfg.training.steps:
        ### Training and exploration

        for env_step in iter_rollout(env, agent):
            buff.add_sample(env_step)

            if global_step % cfg.training.train_every == 0:
                # NOTE: unintuitive that batch_size is now number of total
                #       samples, but not amount of sequences for recurrent model
                rollout_chunk = buff.sample(cfg.training.batch_size,
                                                   cluster_size=cfg.agent.get(
                                                       'batch_cluster_size', 1))

                losses = agent.train(rollout_chunk)
                if cfg.debug.profiler:
                    prof.step()
                if global_step % 100 == 0:
                    logger.log(losses, global_step, mode='train')

            for metric in metrics:
                metric.on_step(logger)

            global_step += cfg.env.repeat_action_num
            pbar.update(cfg.env.repeat_action_num)

        for metric in metrics:
            metric.on_episode(logger)

        # FIXME: find more appealing solution
        ### Validation
        if (global_step % cfg.training.val_logs_every) <= (prev_global_step %
                                                          cfg.training.val_logs_every):
            val_logs(agent, cfg.validation, metrics, val_env, logger)

        ### Checkpoint
        if (global_step % cfg.training.save_checkpoint_every) < (
                prev_global_step % cfg.training.save_checkpoint_every):
            agent.save_ckpt(global_step, losses)

        prev_global_step = global_step

    if cfg.debug.profiler:
        prof.stop()


if __name__ == "__main__":
    main()
