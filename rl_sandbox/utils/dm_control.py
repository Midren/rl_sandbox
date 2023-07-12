import numpy as np
from dm_env import specs
from nptyping import Float, Int, NDArray, Shape


# TODO: add tests
class ActionDiscritizer:
    def __init__(self, action_spec: specs.BoundedArray, values_per_dim: int):
        self.actions_dim = action_spec.shape[0]
        self.min = action_spec.minimum
        self.max = action_spec.maximum
        self.per_dim = values_per_dim
        self.shape = self.per_dim**self.actions_dim

        # actions_dim X per_dim
        self.grid = np.stack([np.linspace(min, max, self.per_dim, endpoint=True) for min, max in zip(self.min, self.max)])

    def discretize(self, action: NDArray[Shape['*'], Float]) -> NDArray[Shape['*'], Int]:
        ks = np.argmin((self.grid - np.ones((self.per_dim, 1)).dot(action).T)**2, axis=1)
        a = 0
        for i, k in enumerate(ks):
            a += k*self.per_dim**i
        # ret_a = np.zeros(self.shape, dtype=np.int64)
        # ret_a[a] = 1
        # return ret_a
        return a

    def undiscretize(self, action: NDArray[Shape['*'], Int]) -> NDArray[Shape['*'], Float]:
        ks = []
        # k = np.argmax(action)
        k = action
        for i in range(self.per_dim - 1, -1, -1):
            ks.append(k // self.per_dim**i)
            k -= ks[-1] * self.per_dim**i

        a = []
        for k, vals in zip(reversed(ks), self.grid):
            a.append(vals[k])
        return np.array(a)
