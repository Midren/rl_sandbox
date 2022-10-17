import typing as t

import gym
import numpy as np

from rl_sandbox.utils.replay_buffer import (Actions, ReplayBuffer, Rewards,
                                            States, TerminationFlag)


def collect_rollout(env: gym.Env, agent: t.Optional[t.Any] = None) -> t.Tuple[States, Actions, Rewards, States, TerminationFlag]:
    s, a, r, n, f = [], [], [], [], []

    obs, _ = env.reset()
    terminated = False

    while not terminated:
        if agent is None:
            action = env.action_space.sample()
        else:
            # FIXME: you know
            action = agent.get_action(obs.reshape(1, -1))[0]
        new_obs, reward, terminated, _, _ = env.step(action)
        s.append(obs)
        a.append(action)
        r.append(reward)
        n.append(new_obs)
        f.append(terminated)
        obs = new_obs
    return np.array(s), np.array(a).reshape(len(s), -1), np.array(r, dtype=np.float32), np.array(n), np.array(f)

def collect_rollout_num(env: gym.Env, num: int, agent: t.Optional[t.Any] = None) -> t.List[t.Tuple[States, Actions, Rewards, States, TerminationFlag]]:
    rollouts = []
    for _ in range(num):
        rollouts.append(collect_rollout(env, agent))
    return rollouts


def fillup_replay_buffer(env: gym.Env, rep_buffer: ReplayBuffer, num: int):
    while not rep_buffer.can_sample(num):
        s, a, r, n, f = collect_rollout(env)
        rep_buffer.add_rollout(s, a, r, n, f)
