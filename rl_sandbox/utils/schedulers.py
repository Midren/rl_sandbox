from abc import ABCMeta

import numpy as np

class Scheduler(metaclass=ABCMeta):
    def step(self) -> float:
        ...

class LinearScheduler(Scheduler):
    def __init__(self, initial_value, final_value, duration):
        self._init = initial_value
        self._final = final_value
        self._dur = duration - 1
        self._curr_t = 0

    @property
    def val(self) -> float:
        if self._curr_t >= self._dur:
            return self._final
        return np.interp([self._curr_t], [0, self._dur], [self._init, self._final]).item()

    def step(self) -> float:
        val = self.val
        self._curr_t += 1
        return val
