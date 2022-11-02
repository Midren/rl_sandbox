import numpy as np

from rl_sandbox.utils.replay_buffer import Rollout


class MetricsEvaluator():
    def calculate_metrics(self, rollouts: list[Rollout]):
        return {
                'episode_len': self._episode_duration(rollouts),
                'episode_return': self._episode_return(rollouts)
                }

    def _episode_duration(self, rollouts: list[Rollout]):
        return np.mean(list(map(lambda x: len(x.states), rollouts)))

    def _episode_return(self, rollouts: list[Rollout]):
        return np.mean(list(map(lambda x: sum(x.rewards), rollouts)))
