import numpy as np
from pytest import fixture

from rl_sandbox.utils.replay_buffer import ReplayBuffer, Rollout


@fixture
def rep_buf():
    return ReplayBuffer()


def test_creation(rep_buf: ReplayBuffer):
    assert len(rep_buf) == 0


def test_adding(rep_buf: ReplayBuffer):
    s = np.ones((3, 8))
    a = np.ones((3, 3), dtype=np.int32)
    r = np.ones((3))
    n = np.ones((3, 8))
    f = np.zeros((3), dtype=np.bool8)
    rep_buf.add_rollout(Rollout(s, a, r, n, f))

    assert len(rep_buf) == 3

    s = np.zeros((3, 8))
    a = np.zeros((3, 3), dtype=np.int32)
    r = np.zeros((3))
    n = np.zeros((3, 8))
    f = np.zeros((3), dtype=np.bool8)
    rep_buf.add_rollout(Rollout(s, a, r, n, f))

    assert len(rep_buf) == 6


def test_can_sample(rep_buf: ReplayBuffer):
    assert rep_buf.can_sample(1) == False

    s = np.ones((3, 8))
    a = np.zeros((3, 3), dtype=np.int32)
    r = np.ones((3))
    n = np.zeros((3, 8))
    f = np.zeros((3), dtype=np.bool8)
    rep_buf.add_rollout(Rollout(s, a, r, n, f))

    assert rep_buf.can_sample(5) == False
    assert rep_buf.can_sample(1) == True

    rep_buf.add_rollout(Rollout(s, a, r, n, f))

    assert rep_buf.can_sample(5) == True


def test_sampling(rep_buf: ReplayBuffer):
    for i in range(5):
        rep_buf.add_rollout(
            Rollout(np.ones((1, 3)), np.ones((1, 2), dtype=np.int32), i * np.ones((1)),
                    np.ones((3, 8)), np.zeros((3), dtype=np.bool8)))

    np.random.seed(42)
    _, _, r, _, _ = rep_buf.sample(3)
    assert (r == [1, 4, 3]).all()


def test_cluster_sampling(rep_buf: ReplayBuffer):
    for i in range(5):
        rep_buf.add_rollout(
            Rollout(np.stack([np.arange(3, dtype=np.float32) for _ in range(3)]).T,
                    np.ones((3, 2), dtype=np.int32), i * np.ones((3)),
                    np.stack([np.arange(1, 4, dtype=np.float32) for _ in range(3)]).T,
                    np.zeros((3), dtype=np.bool8)))

    np.random.seed(42)
    s, _, r, n, _ = rep_buf.sample(4, cluster_size=2)
    assert (r == [1, 1, 4, 4]).all()
    assert (s[:, 0] == [0, 1, 1, 2]).all()
    assert (n[:, 0] == [1, 2, 2, 3]).all()

    s, _, r, n, _ = rep_buf.sample(4, cluster_size=2)
    assert (r == [2, 2, 0, 0]).all()
    assert (s[:, 0] == [0, 1, 0, 1]).all()
    assert (n[:, 0] == [1, 2, 1, 2]).all()

    s, _, r, n, _ = rep_buf.sample(4, cluster_size=2)
    assert (r == [0, 0, 4, 4]).all()
    assert (s[:, 0] == [1, 2, 1, 2]).all()
    assert (n[:, 0] == [2, 3, 2, 3]).all()
