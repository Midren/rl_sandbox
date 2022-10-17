import typing as t

import gym
import numpy as np

from rl_sandbox.utils.replay_buffer import (ReplayBuffer, Rollout)

def collect_rollout(env: gym.Env, agent: t.Optional[t.Any] = None, save_obs: bool = False) -> Rollout:
    s, a, r, n, f = [], [], [], [], []

    state, _ = env.reset()
    terminated = False

    while not terminated:
        if agent is None:
            action = env.action_space.sample()
        else:
            # FIXME: move reshaping inside DqnAgent
            action = agent.get_action(state.reshape(1, -1))[0]
        new_state, reward, terminated, _, _ = env.step(action)
        s.append(state)
        a.append(action)
        r.append(reward)
        n.append(new_state)
        f.append(terminated)
        state = new_state

    obs = np.stack(list(env.render())) if save_obs else None
    return Rollout(np.array(s), np.array(a).reshape(len(s), -1), np.array(r, dtype=np.float32), np.array(n), np.array(f), obs)

def collect_rollout_num(env: gym.Env, num: int, agent: t.Optional[t.Any] = None, save_obs: bool = False) -> t.List[Rollout]:
    rollouts = []
    for _ in range(num):
        rollouts.append(collect_rollout(env, agent, save_obs))
    return rollouts


def fillup_replay_buffer(env: gym.Env, rep_buffer: ReplayBuffer, num: int):
    while not rep_buffer.can_sample(num):
        rep_buffer.add_rollout(collect_rollout(env))
