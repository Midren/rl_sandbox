import typing as t
from multiprocessing.synchronize import Lock

import numpy as np
import torch.multiprocessing as mp
from unpackable import unpack

from rl_sandbox.agents.random_agent import RandomAgent
from rl_sandbox.agents.rl_agent import RlAgent
from rl_sandbox.utils.env import Env
from rl_sandbox.utils.replay_buffer import (Action, Observation, ReplayBuffer,
                                            Rollout, State)


def _async_env_worker(env: Env, obs_queue: mp.Queue, act_queue: mp.Queue):
    state, _, terminated = unpack(env.reset())
    obs_queue.put((state, 0, terminated), block=False)

    while not terminated:
        action = act_queue.get(block=True)

        new_state, reward, terminated = unpack(env.step(action))
        del action
        obs_queue.put((state, reward, terminated), block=False)

        state = new_state


def iter_rollout_async(
    env: Env,
    agent: RlAgent
) -> t.Generator[tuple[State, Action, float, State, bool, t.Optional[Observation]], None,
                 None]:
    # NOTE: maybe use SharedMemory instead
    obs_queue = mp.Queue(1)
    a_queue = mp.Queue(1)
    p = mp.Process(target=_async_env_worker, args=(env, obs_queue, a_queue))
    p.start()
    terminated = False

    while not terminated:
        state, reward, terminated = obs_queue.get(block=True)
        action = agent.get_action(state)
        a_queue.put(action)
        yield state, action, reward, None, terminated, state


def iter_rollout(
    env: Env,
    agent: RlAgent,
    collect_obs: bool = False
) -> t.Generator[tuple[State, Action, float, State, bool, t.Optional[Observation]], None,
                 None]:
    state, _, terminated = unpack(env.reset())
    agent.reset()

    prev_action = np.zeros_like(agent.get_action(state))
    while not terminated:
        action = agent.get_action(state)

        new_state, reward, terminated = unpack(env.step(action))

        # FIXME: will break for non-DM
        obs = env.render() if collect_obs else None
        # if collect_obs and isinstance(env, dmEnv):
        yield state, prev_action, reward, new_state, terminated, obs
        state = new_state
        prev_action = action


def collect_rollout(env: Env,
                    agent: t.Optional[RlAgent] = None,
                    collect_obs: bool = False) -> Rollout:
    s, a, r, n, f, o = [], [], [], [], [], []

    if agent is None:
        agent = RandomAgent(env)

    for state, action, reward, new_state, terminated, obs in iter_rollout(
            env, agent, collect_obs):
        s.append(state)
        a.append(action)
        r.append(reward)
        n.append(new_state)
        f.append(terminated)

        # FIXME: will break for non-DM
        if collect_obs:
            o.append(obs)

    # match env:
    #     case gym.Env():
    #         obs = np.stack(list(env.render())) if obs_res is not None else None
    #     case dmEnv():
    obs = np.array(o) if collect_obs is not None else None
    return Rollout(np.array(s),
                   np.array(a).reshape(len(s), -1), np.array(r, dtype=np.float32),
                   np.array(n), np.array(f), obs)


def collect_rollout_num(env: Env,
                        num: int,
                        agent: t.Optional[t.Any] = None,
                        collect_obs: bool = False) -> t.List[Rollout]:
    # TODO: paralelyze
    rollouts = []
    for _ in range(num):
        rollouts.append(collect_rollout(env, agent, collect_obs))
    return rollouts


def fillup_replay_buffer(env: Env, rep_buffer: ReplayBuffer, num: int):
    # TODO: paralelyze
    while not rep_buffer.can_sample(num):
        rep_buffer.add_rollout(collect_rollout(env, collect_obs=False))
