import random

import crafter
import hydra
import lovely_tensors as lt
import numpy as np
import torch
from gym.spaces import Discrete
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

from rl_sandbox.metrics import MetricsEvaluator
from rl_sandbox.utils.env import Env
from rl_sandbox.utils.logger import Logger
from rl_sandbox.utils.replay_buffer import ReplayBuffer
from rl_sandbox.utils.rollout_generation import (collect_rollout_num,
                                                 fillup_replay_buffer,
                                                 iter_rollout)


def val_logs(agent, val_cfg: DictConfig, env: Env, global_step: int, logger: Logger):
    with torch.no_grad():
        rollouts = collect_rollout_num(env, val_cfg.rollout_num, agent)
    # TODO: make logs visualization in separate process
    # Possibly make the data loader
    metrics = MetricsEvaluator().calculate_metrics(rollouts)
    logger.log(metrics, global_step, mode='val')

    if val_cfg.visualize:
        rollouts = collect_rollout_num(env, 1, agent, collect_obs=True)

        for rollout in rollouts:
            video = np.expand_dims(rollout.observations.transpose(0, 3, 1, 2), 0)
            logger.add_video('val/visualization', video, global_step)
            # FIXME: Very bad from architecture point
            with torch.no_grad():
                agent.viz_log(rollout, logger, global_step)


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

    # TODO: Implement smarter techniques for exploration
    #       (Plan2Explore, etc)
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

    buff = ReplayBuffer(prioritize_ends=cfg.training.prioritize_ends,
                        min_ep_len=cfg.agent.get('batch_cluster_size', 1) *
                        (cfg.training.prioritize_ends + 1))
    fillup_replay_buffer(
        env, buff,
        max(cfg.training.prefill,
            cfg.training.batch_size * cfg.agent.get('batch_cluster_size', 1)))

    is_discrete = isinstance(env.action_space, Discrete)
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_space_num=env.observation_space.shape,
        actions_num=env.action_space.n if is_discrete else env.action_space.shape[0],
        action_type='discrete' if is_discrete else 'continuous',
        device_type=cfg.device_type,
        logger=logger)

    prof = profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(logger.log_dir() + '/profiler'),
        schedule=torch.profiler.schedule(wait=10, warmup=10, active=5, repeat=5),
        with_stack=True) if cfg.debug.profiler else None

    for i in tqdm(range(int(cfg.training.pretrain)), desc='Pretraining'):
        if cfg.training.checkpoint_path is not None:
            break
        s, a, r, n, f, first = buff.sample(cfg.training.batch_size,
                                           cluster_size=cfg.agent.get(
                                               'batch_cluster_size', 1))
        losses = agent.train(s, a, r, n, f, first)
        logger.log(losses, i, mode='pre_train')

        # TODO: remove constants
        # log_every_n = 25
        # if i % log_every_n == 0:
        #     st = int(cfg.training.pretrain) // log_every_n
    val_logs(agent, cfg.validation, val_env, -1, logger)

    if cfg.training.checkpoint_path is not None:
        prev_global_step = global_step = agent.load_ckpt(cfg.training.checkpoint_path)
    else:
        prev_global_step = global_step = 0

    pbar = tqdm(total=cfg.training.steps, desc='Training')
    while global_step < cfg.training.steps:
        ### Training and exploration

        for s, a, r, n, f, _ in iter_rollout(env, agent):
            buff.add_sample(s, a, r, n, f)

            if global_step % cfg.training.train_every == 0:
                # NOTE: unintuitive that batch_size is now number of total
                #       samples, but not amount of sequences for recurrent model
                s, a, r, n, f, first = buff.sample(cfg.training.batch_size,
                                                   cluster_size=cfg.agent.get(
                                                       'batch_cluster_size', 1))

                losses = agent.train(s, a, r, n, f, first)
                if cfg.debug.profiler:
                    prof.step()
                if global_step % 100 == 0:
                    logger.log(losses, global_step, mode='train')

            global_step += cfg.env.repeat_action_num
            pbar.update(cfg.env.repeat_action_num)

        # FIXME: find more appealing solution
        ### Validation
        if (global_step % cfg.training.val_logs_every) < (prev_global_step %
                                                          cfg.training.val_logs_every):
            val_logs(agent, cfg.validation, val_env, global_step, logger)

        ### Checkpoint
        if (global_step % cfg.training.save_checkpoint_every) < (
                prev_global_step % cfg.training.save_checkpoint_every):
            agent.save_ckpt(global_step, losses)

        prev_global_step = global_step

    if cfg.debug.profiler:
        prof.stop()


if __name__ == "__main__":
    main()
