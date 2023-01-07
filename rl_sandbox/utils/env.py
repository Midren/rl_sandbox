import typing as t
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import gym
import numpy as np
from dm_control import suite
from dm_env import Environment as dmEnviron
from dm_env import TimeStep
from nptyping import Float, Int, NDArray, Shape

Observation = NDArray[Shape["*,*,3"], Int]
State = NDArray[Shape["*"], Float]
Action = NDArray[Shape["*"], Int]


@dataclass
class EnvStepResult:
    obs: Observation | State
    reward: float
    terminated: bool


class ActionTransformer(metaclass=ABCMeta):

    def set_env(self, env: 'Env'):
        self.low = env.action_space.low
        self.high = env.action_space.high

    @abstractmethod
    def transform_action(self, action):
        ...

    @abstractmethod
    def transform_space(self, space: gym.spaces.Box):
        ...


class ActionNormalizer(ActionTransformer):

    def set_env(self, env: 'Env'):
        super().set_env(env)
        if (~np.isfinite(self.low) | ~np.isfinite(self.high)).any():
            raise RuntimeError("Not bounded space cannot be normalized")

    def transform_action(self, action):
        return (self.high - self.low) * (action + 1) / 2 + self.low

    def transform_space(self, space: gym.spaces.Box):
        return gym.spaces.Box(-np.ones_like(self.low),
                              np.ones_like(self.high),
                              dtype=np.float32)


class ActionDisritezer(ActionTransformer):

    def __init__(self, actions_num: int):
        self.per_dim = actions_num

    def set_env(self, env: 'Env'):
        super().set_env(env)
        if (~np.isfinite(self.low) | ~np.isfinite(self.high)).any():
            raise RuntimeError("Not bounded space cannot be discritized")

        self.grid = np.stack([
            np.linspace(min, max, self.per_dim, endpoint=True)
            for min, max in zip(self.low, self.high)
        ])

    def transform_action(self, action: NDArray[Shape['*'],
                                               Int]) -> NDArray[Shape['*'], Float]:
        ks = []
        k = action
        for i in range(self.per_dim - 1, -1, -1):
            ks.append(k // self.per_dim**i)
            k -= ks[-1] * self.per_dim**i

        a = []
        for k, vals in zip(reversed(ks), self.grid):
            a.append(vals[k])
        return np.array(a)

    def transform_space(self, space: gym.spaces.Box):
        return gym.spaces.Box(0, self.per_dim**len(self.low)-1, dtype=np.int32)


class Env(metaclass=ABCMeta):

    def __init__(self, run_on_pixels: bool, obs_res: tuple[int, int],
                 repeat_action_num: int, transforms: list[ActionTransformer]):
        self.obs_res = obs_res
        self.run_on_pixels = run_on_pixels
        self.repeat_action_num = repeat_action_num
        assert self.repeat_action_num >= 1
        self.ac_trans = []
        for t in transforms:
            t.set_env(self)
            self.ac_trans.append(t)

    def step(self, action: Action) -> EnvStepResult:
        for t in reversed(self.ac_trans):
            action = t.transform_action(action)
        return self._step(action, self.repeat_action_num)

    @abstractmethod
    def _step(self, action: Action, repeat_num: int = 1) -> EnvStepResult:
        pass

    @abstractmethod
    def reset(self) -> EnvStepResult:
        pass

    @abstractmethod
    def _observation_space(self) -> gym.Space:
        pass

    @abstractmethod
    def _action_space(self) -> gym.Space:
        ...

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space()

    @property
    def action_space(self) -> gym.Space:
        space = self._action_space()
        for t in self.ac_trans:
            space = t.transform_space(t)
        return space


class GymEnv(Env):

    def __init__(self, task_name: str, run_on_pixels: bool, obs_res: tuple[int, int],
                 repeat_action_num: int, transforms: list[ActionTransformer]):
        super().__init__(run_on_pixels, obs_res, repeat_action_num, transforms)

        self.env: gym.Env = gym.make(task_name)
        self.visualized_env: gym.Env = gym.make(task_name, render_mode='rgb_array_list')

        if run_on_pixels:
            raise NotImplementedError("Run on pixels supported only for 'dm_control'")

    def _step(self, action: Action) -> EnvStepResult:
        new_state, reward, terminated, _, _ = self.env.step(action)
        return EnvStepResult(new_state, reward, terminated)

    def reset(self):
        state, _ = self.env.reset()
        return EnvStepResult(state, 0, False)

    @property
    def _observation_space(self):
        return self.env.observation_space

    def _action_space(self):
        return self.env.action_space

class MockEnv(Env):

    def __init__(self, run_on_pixels: bool,
                 obs_res: tuple[int, int], repeat_action_num: int,
                 transforms: list[ActionTransformer]):
        super().__init__(run_on_pixels, obs_res, repeat_action_num, transforms)
        self.max_steps = 255
        self.step_count = 0

    def _step(self, action: Action, repeat_num: int) -> EnvStepResult:
        self.step_count += repeat_num
        return EnvStepResult(self.render(), self.step_count, self.step_count >= self.max_steps)

    def reset(self):
        self.step_count = 0
        return EnvStepResult(self.render(), 0, False)

    def render(self):
        return np.ones(self.obs_res + (3, )) * self.step_count

    def _observation_space(self):
        return gym.spaces.Box(0, 255, self.obs_res + (3, ), dtype=np.uint8)

    def _action_space(self):
        return gym.spaces.Box(-1, 1, (1, ), dtype=np.float32)


class DmEnv(Env):

    def __init__(self, run_on_pixels: bool, obs_res: tuple[int,
                                                           int], repeat_action_num: int,
                 domain_name: str, task_name: str, transforms: list[ActionTransformer]):
        self.env: dmEnviron = suite.load(domain_name=domain_name, task_name=task_name)
        super().__init__(run_on_pixels, obs_res, repeat_action_num, transforms)

    def render(self):
        return self.env.physics.render(*self.obs_res)

    def _uncode_ts(self, ts: TimeStep) -> EnvStepResult:
        if self.run_on_pixels:
            state = self.render()
        else:
            state = ts.observation
            state = np.concatenate([state[s] for s in state], dtype=np.float32)
        return EnvStepResult(state, ts.reward, ts.last())

    def _step(self, action: Action, repeat_num: int) -> EnvStepResult:
        rew = 0
        for _ in range(repeat_num - 1):
            ts =  self.env.step(action)
            rew += ts.reward or 0.0
            if ts.last():
                break
        if repeat_num == 1 or not ts.last():
            env_res = self._uncode_ts(self.env.step(action))
        else:
            env_res = ts
        env_res.reward = np.tanh(rew + (env_res.reward or 0.0))
        return env_res

    def reset(self) -> EnvStepResult:
        return self._uncode_ts(self.env.reset())

    def _observation_space(self):
        if self.run_on_pixels:
            return gym.spaces.Box(0, 255, self.obs_res + (3, ), dtype=np.uint8)
        else:
            raise NotImplementedError(
                "Currently run on pixels is supported for 'dm_control'")
        # for space in self.env.observation_spec():
        # obs_space_num = sum([v.shape[0] for v in env.observation_space().values()])

    def _action_space(self):
        spec = self.env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
