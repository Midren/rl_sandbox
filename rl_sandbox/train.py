import hydra
import numpy as np
from omegaconf import DictConfig
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import random

import torch
from torch.profiler import profile, ProfilerActivity
import lovely_tensors as lt
from gym.spaces import Discrete
import crafter

from rl_sandbox.metrics import MetricsEvaluator
from rl_sandbox.utils.env import Env
from rl_sandbox.utils.replay_buffer import ReplayBuffer
from rl_sandbox.utils.rollout_generation import (collect_rollout_num, iter_rollout,
                                                 fillup_replay_buffer)


class SummaryWriterMock():
    def add_scalar(*args, **kwargs):
        pass

    def add_video(*args, **kwargs):
        pass

    def add_image(*args, **kwargs):
        pass


def val_logs(agent, val_cfg: DictConfig, env, global_step, writer):
    with torch.no_grad():
        rollouts = collect_rollout_num(env, val_cfg.rollout_num, agent)
    # TODO: make logs visualization in separate process
    # Possibly make the data loader
    metrics = MetricsEvaluator().calculate_metrics(rollouts)
    for metric_name, metric in metrics.items():
        writer.add_scalar(f'val/{metric_name}', metric, global_step)

    if val_cfg.visualize:
        rollouts = collect_rollout_num(env, 1, agent, collect_obs=True)

        for rollout in rollouts:
            video = np.expand_dims(rollout.observations.transpose(0, 3, 1, 2), 0)
            writer.add_video('val/visualization', video, global_step, fps=20)
            # FIXME: Very bad from architecture point
            with torch.no_grad():
                agent.viz_log(rollout, writer, global_step)


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
    writer = SummaryWriter(comment=cfg.log_message or "")

    env: Env = hydra.utils.instantiate(cfg.env)
    val_env: Env = hydra.utils.instantiate(cfg.env)
    # TOOD: Create maybe some additional validation env
    if cfg.env.task_name.startswith("Crafter"):
        val_env.env = crafter.Recorder(val_env.env,
                    writer.log_dir,
                    save_stats=True,
                    save_video=False,
                    save_episode=False)

    buff = ReplayBuffer(prioritize_ends=cfg.training.prioritize_ends,
                        min_ep_len=cfg.agent.get('batch_cluster_size', 1)*(cfg.training.prioritize_ends + 1))
    fillup_replay_buffer(env, buff, max(cfg.training.prefill, cfg.training.batch_size))

    is_discrete = isinstance(env.action_space, Discrete)
    agent = hydra.utils.instantiate(cfg.agent,
                            obs_space_num=env.observation_space.shape,
                            actions_num = env.action_space.n if is_discrete else env.action_space.shape[0],
                            action_type='discrete' if is_discrete else 'continuous' ,
                            device_type=cfg.device_type,
                            logger=writer)

    prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('runs/profile_dreamer'),
                 schedule=torch.profiler.schedule(wait=10, warmup=10, active=5, repeat=5),
                 with_stack=True) if cfg.debug.profiler else None

    for i in tqdm(range(int(cfg.training.pretrain)), desc='Pretraining'):
        if cfg.training.checkpoint_path is not None:
            break
        s, a, r, n, f, first = buff.sample(cfg.training.batch_size,
                                    cluster_size=cfg.agent.get('batch_cluster_size', 1))
        losses = agent.train(s, a, r, n, f, first)
        for loss_name, loss in losses.items():
            if 'grad' in loss_name:
                writer.add_histogram(f'pre_train/{loss_name}', loss, i)
            else:
                writer.add_scalar(f'pre_train/{loss_name}', loss.item(), i)

        # TODO: remove constants
        log_every_n = 25
        st = int(cfg.training.pretrain) // log_every_n
        if i % log_every_n == 0:
            val_logs(agent, cfg.validation, val_env, -st + i/log_every_n, writer)

    if cfg.training.checkpoint_path is not None:
        prev_global_step = global_step = agent.load_ckpt(cfg.training.checkpoint_path)
    else:
        prev_global_step = global_step = 0

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
                if global_step % 100 == 0:
                    for loss_name, loss in losses.items():
                        if 'grad' in loss_name:
                            writer.add_histogram(f'train/{loss_name}', loss, global_step)
                        else:
                            writer.add_scalar(f'train/{loss_name}', loss.item(), global_step)
            global_step += cfg.env.repeat_action_num
            pbar.update(cfg.env.repeat_action_num)

        # FIXME: find more appealing solution
        ### Validation
        if (global_step % cfg.training.val_logs_every) < (prev_global_step % cfg.training.val_logs_every):
            val_logs(agent, cfg.validation, val_env, global_step, writer)

        ### Checkpoint
        if (global_step % cfg.training.save_checkpoint_every) < (prev_global_step % cfg.training.save_checkpoint_every):
            agent.save_ckpt(global_step, losses)

        prev_global_step = global_step

    if cfg.debug.profiler:
        prof.stop()


if __name__ == "__main__":
    main()

