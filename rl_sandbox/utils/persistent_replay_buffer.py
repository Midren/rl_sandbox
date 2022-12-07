import typing as t
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import webdataset as wds

from rl_sandbox.utils.replay_buffer import (Action, Actions, Observation,
                                            Observations, Rewards, Rollout,
                                            State, States, TerminationFlags)


# TODO: add tagging of replay buffer meta-data (env config)
# to omit incompatible cache
class PersistentReplayBuffer:

    def __init__(self, directory: Path, max_len=1e6):
        self.max_len: int = int(max_len)
        self.directory = directory
        self.directory.mkdir(exist_ok=True)
        self.rollouts: list[str] = list(map(str, self.directory.glob('*.tar')))
        self.rollouts_num = len(self.rollouts)
        # FIXME: add correct length calculation, currently hardcoded
        self.rollouts_len: list[int] = [200] * self.rollouts_num
        self.total_num = sum(self.rollouts_len)
        self.rollout_idx = self.rollouts_num

        self.curr_rollout: t.Optional[Rollout] = None
        self.rollouts_changed: bool = True

    def add_rollout(self, rollout: Rollout):
        name = str(self.directory / f'rollout-{self.rollout_idx % self.max_len}.tar')
        sink = wds.TarWriter(name)

        for idx in range(len(rollout)):
            s, a, r, t = rollout.states[idx], rollout.actions[idx], rollout.rewards[
                idx], rollout.is_finished[idx]
            sink.write({
                "__key__": "sample%06d" % idx,
                "state.pyd": s,
                "action.pyd": a,
                "reward.pyd": np.array(r, dtype=np.float32),
                "is_finished.pyd": np.array(t, dtype=np.bool_)
            })

        if self.rollout_idx < self.max_len:
            self.total_num += len(rollout)
            self.rollouts_num += 1
            self.rollouts.append(name)
            self.rollouts_len.append(len(rollout))
        else:
            self.total_num += len(rollout) - self.rollouts_len[self.rollout_idx %
                                                               self.max_len]
            self.rollouts[self.rollout_idx % self.max_len] = name
            self.rollouts_len[self.rollout_idx % self.max_len] = len(rollout)
        self.rollout_idx += 1
        self.rollouts_changed = True

    # Add sample expects that each subsequent sample
    # will be continuation of last rollout util termination flag true
    # is encountered
    def add_sample(self, s: State, a: Action, r: float, n: State, f: bool):
        if self.curr_rollout is None:
            self.curr_rollout = Rollout([s], [a], [r], None, [f])
        else:
            self.curr_rollout.states.append(s)
            self.curr_rollout.actions.append(a)
            self.curr_rollout.rewards.append(r)
            self.curr_rollout.is_finished.append(f)

            if f:
                self.add_rollout(
                    Rollout(np.array(self.curr_rollout.states),
                            np.array(self.curr_rollout.actions),
                            np.array(self.curr_rollout.rewards, dtype=np.float32),
                            np.array([n]), np.array(self.curr_rollout.is_finished)))
                self.curr_rollout = None

    def can_sample(self, num: int):
        return self.total_num >= num

    @staticmethod
    def add_next(src):
        s, a, r, t = src
        return s[:-1], a[:-1], r[:-1], s[1:], t[:-1]

    def sample(
        self,
        batch_size: int,
        cluster_size: int = 1
    ) -> tuple[States, Actions, Rewards, States, TerminationFlags]:
        seq_num = batch_size // cluster_size
        # TODO: Could be done in async before
        # NOTE: maybe use WDS_REWRITE

        if self.rollouts_changed:
            # NOTE: shardshuffle will specify amount of urls that will be taken
            # into account. Sorting not everything doesn't make sense
            self.dataset = wds.WebDataset(self.rollouts
                ).decode().to_tuple("state.pyd", "action.pyd", "reward.pyd", "is_finished.pyd"
                # NOTE: does not take into account is_finished
                ).batched(cluster_size + 1, partial=False
                ).map(self.add_next).batched(seq_num)
            # NOTE: in WebDataset github, it is recommended to use such batching by ourselves
            # https://github.com/webdataset/webdataset#dataloader
            self.loader = iter(
                wds.WebLoader(self.dataset, batch_size=None,
                              num_workers=4, pin_memory=True).unbatched().shuffle(1000).unbatched().batched(batch_size))
        return next(self.loader)
