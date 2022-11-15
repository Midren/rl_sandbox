import typing as t

import numpy as np
from unpackable import unpack

from rl_sandbox.agents.random_agent import RandomAgent
from rl_sandbox.utils.env import Env
from rl_sandbox.utils.replay_buffer import ReplayBuffer, Rollout


# FIXME: whole function duplicates a lot of code from main.py
def collect_rollout(env: Env,
                    agent: t.Optional[t.Any] = None,
                    collect_obs: bool = False
                    ) -> Rollout:

    s, a, r, n, f, o = [], [], [], [], [], []

    if agent is None:
        agent = RandomAgent(env)

    state, _, terminated = unpack(env.reset())
    agent.reset()

    while not terminated:
        action = agent.get_action(state)

        new_state, reward, terminated = unpack(env.step(action))

        s.append(state)
        a.append(action)
        r.append(reward)
        n.append(new_state)
        f.append(terminated)

        # FIXME: will break for non-DM
        if collect_obs:
            o.append(env.render())
        # if collect_obs and isinstance(env, dmEnv):
        state = new_state

    obs = None
    # match env:
    #     case gym.Env():
    #         obs = np.stack(list(env.render())) if obs_res is not None else None
    #     case dmEnv():
    obs = np.array(o) if collect_obs is not None else None
    return Rollout(np.array(s), np.array(a).reshape(len(s), -1), np.array(r, dtype=np.float32), np.array(n), np.array(f), obs)

def collect_rollout_num(env: Env,
                        num: int,
                        agent: t.Optional[t.Any] = None,
                        collect_obs: bool = False) -> t.List[Rollout]:
    # TODO: paralelyze
    rollouts = []
    for _ in range(num):
        rollouts.append(collect_rollout(env, agent, collect_obs))
    return rollouts


def fillup_replay_buffer(env: Env,
                         rep_buffer: ReplayBuffer,
                         num: int):
    # TODO: paralelyze
    while not rep_buffer.can_sample(num):
        rep_buffer.add_rollout(collect_rollout(env, collect_obs=False))
