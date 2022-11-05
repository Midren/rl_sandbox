import typing as t

import gym
import numpy as np
from dm_env import Environment as dmEnv

from rl_sandbox.agents.random_agent import RandomAgent
from rl_sandbox.utils.dm_control import ActionDiscritizer, decode_dm_ts
from rl_sandbox.utils.replay_buffer import ReplayBuffer, Rollout


def collect_rollout(env: gym.Env | dmEnv, agent: t.Optional[t.Any] = None, obs_res: t.Optional[t.Tuple[int, int]] = None) -> Rollout:
    s, a, r, n, f, o = [], [], [], [], [], []

    match env:
        case gym.Env():
            state, _ = env.reset()
        case dmEnv():
            state, _, terminated = decode_dm_ts(env.reset())

    if agent is None:
        agent = RandomAgent(env)


    while not terminated:
        action = agent.get_action(state)

        match env:
            case gym.Env():
                new_state, reward, terminated, _, _ = env.step(action)
            case dmEnv():
                new_state, reward, terminated = decode_dm_ts(env.step(action))

        s.append(state)
        # FIXME: action discritezer should be defined once
        action_disritizer = ActionDiscritizer(env.action_spec(), values_per_dim=10)
        a.append(action_disritizer.discretize(action))
        r.append(reward)
        n.append(new_state)
        f.append(terminated)

        if obs_res is not None and isinstance(env, dmEnv):
            o.append(env.physics.render(*obs_res, camera_id=0))
        state = new_state

    match env:
        case gym.Env():
            obs = np.stack(list(env.render())) if obs_res is not None else None
        case dmEnv():
            obs = np.array(o) if obs_res is not None else None
    return Rollout(np.array(s), np.array(a).reshape(len(s), -1), np.array(r, dtype=np.float32), np.array(n), np.array(f), obs)

def collect_rollout_num(env: gym.Env, num: int, agent: t.Optional[t.Any] = None, obs_res: bool = False) -> t.List[Rollout]:
    # TODO: paralelyze
    rollouts = []
    for _ in range(num):
        rollouts.append(collect_rollout(env, agent, obs_res))
    return rollouts


def fillup_replay_buffer(env: gym.Env, rep_buffer: ReplayBuffer, num: int, obs_res: t.Optional[t.Tuple[int, int]] = None):
    # TODO: paralelyze
    while not rep_buffer.can_sample(num):
        rep_buffer.add_rollout(collect_rollout(env, obs_res=obs_res))
