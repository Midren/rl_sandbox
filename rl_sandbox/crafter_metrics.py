import json
import pathlib
import warnings
import collections
from pathlib import Path

import numpy as np

from rl_sandbox.utils.replay_buffer import Rollout

def compute_scores(percents):
  # Geometric mean with an offset of 1%.
  assert (0 <= percents).all() and (percents <= 100).all()
  if (percents <= 1.0).all():
    print('Warning: The input may not be in the right range.')
  with warnings.catch_warnings():  # Empty seeds become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
  return scores


def load_stats(filename, budget):
  steps = 0
  rewards = []
  lengths = []
  achievements = collections.defaultdict(list)
  for line in filename.read_text().split('\n'):
    if not line.strip():
      continue
    episode = json.loads(line)
    steps += episode['length']
    if steps > budget:
      break
    lengths.append(episode['length'])
    for key, value in episode.items():
      if key.startswith('achievement_'):
        achievements[key].append(value)
    unlocks = int(np.sum([(v[-1] >= 1) for v in achievements.values()]))
    health = -0.9
    rewards.append(unlocks + health)
  return rewards, lengths, achievements


class CrafterMetricsEvaluator():
    def __init__(self, agent: 'DreamerV2'):
        self.agent = agent
        self.episode = 0

    def on_val(self, logger, rollouts: list[Rollout], global_step: int):
        if logger.log_dir() is None:
            return
        budget = 1e6
        stats_file = Path(logger.log_dir()) / "stats.jsonl"
        _, lengths, achievements = load_stats(stats_file, budget)

        tasks = list(achievements.keys())

        xs = np.cumsum(lengths).tolist()
        episodes = (np.array(xs) <= budget).sum()
        percents = np.empty((len(achievements)))
        percents[:] = np.nan
        for key, values in achievements.items():
            k = tasks.index(key)
            percent = 100 * (np.array(values[:episodes]) >= 1).mean()
            percents[k] = percent

        score = compute_scores(percents)

        logger.log({"score": score}, global_step, mode='val')

    def on_step(self, logger):
        pass

    def on_episode(self, logger, rollout, global_step: int):
        pass



