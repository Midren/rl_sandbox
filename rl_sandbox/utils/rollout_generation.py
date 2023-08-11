import typing as t
from collections import defaultdict
from multiprocessing.synchronize import Lock

import numpy as np
import torch
import torch.multiprocessing as mp
from IPython.core.inputtransformer2 import warnings
from unpackable import unpack

from rl_sandbox.agents.random_agent import RandomAgent
from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.env import Env
from rl_sandbox.utils.replay_buffer import EnvStep, ReplayBuffer, Rollout

# (Action, Observation, ReplayBuffer, Rollout, State)

# FIXME: obsolete, need to be updated for new replay buffer
# def _async_env_worker(env: Env, obs_queue: mp.Queue, act_queue: mp.Queue):
#     state, _, terminated = unpack(env.reset())
#     obs_queue.put((state, 0, terminated), block=False)

#     while not terminated:
#         action = act_queue.get(block=True)

#         new_state, reward, terminated = unpack(env.step(action))
#         del action
#         obs_queue.put((state, reward, terminated), block=False)

#         state = new_state

# def iter_rollout_async(
#     env: Env,
#     agent: RlAgent
# ) -> t.Generator[tuple[State, Action, float, State, bool, t.Optional[Observation]], None,
#                  None]:
#     # NOTE: maybe use SharedMemory instead
#     obs_queue = mp.Queue(1)
#     a_queue = mp.Queue(1)
#     p = mp.Process(target=_async_env_worker, args=(env, obs_queue, a_queue))
#     p.start()
#     terminated = False

#     while not terminated:
#         state, reward, terminated = obs_queue.get(block=True)
#         action = agent.get_action(state)
#         a_queue.put(action)
#         yield state, action, reward, None, terminated, state


def iter_rollout(env: Env,
                 agent: RlAgent,
                 collect_obs: bool = False) -> t.Generator[EnvStep, None, None]:
    state, _, terminated = unpack(env.reset())
    agent.reset()

    reward = 0.0
    is_first = True
    with torch.no_grad():
        action = torch.zeros_like(agent.get_action(state))

    while not terminated:
        try:
            obs = env.render() if collect_obs else None
        except RuntimeError:
            # FIXME: hot-fix for Crafter env to work
            warnings.warn("Cannot render environment, using state instead")
            obs = state

        # FIXME: works only for crafter
        yield EnvStep(obs=torch.from_numpy(state),
                      action=torch.Tensor(action).squeeze(),
                      reward=reward,
                      is_finished=terminated,
                      is_first=is_first)
        is_first = False

        with torch.no_grad():
            action = agent.get_action(state)

        state, reward, terminated = unpack(env.step(action))


def collect_rollout(env: Env,
                    agent: t.Optional[RlAgent] = None,
                    collect_obs: bool = False) -> Rollout:
    s, a, r, t, f, additional = [], [], [], [], [], defaultdict(list)

    if agent is None:
        agent = RandomAgent(env)

    for step in iter_rollout(env, agent, collect_obs):
        obs, action, reward, terminated, first, add = unpack(step)
        s.append(obs)
        a.append(action)
        r.append(reward)
        t.append(terminated)
        f.append(first)
        for k, v in add.items():
            additional[k].append(v)

    return Rollout(torch.stack(s), torch.stack(a).reshape(len(a), -1),
                   torch.Tensor(r).float(), torch.Tensor(t), torch.Tensor(f),
                   {k: torch.stack(v)
                    for k, v in additional.items()})


def collect_rollout_num(env: Env,
                        num: int,
                        agent: t.Optional[t.Any] = None,
                        collect_obs: bool = False) -> t.List[Rollout]:
    # TODO: paralelyze
    rollouts = []
    for _ in range(num):
        rollouts.append(collect_rollout(env, agent, collect_obs))
    return rollouts


def fillup_replay_buffer(env: Env, rep_buffer: ReplayBuffer, num: int, agent: t.Optional[RlAgent] = None):
    # TODO: paralelyze
    while not rep_buffer.can_sample(num):
        rep_buffer.add_rollout(collect_rollout(env, agent=agent, collect_obs=False))
