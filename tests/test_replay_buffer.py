import numpy as np
import random
from pytest import fixture

from rl_sandbox.utils.replay_buffer import ReplayBuffer

@fixture
def rep_buf():
    return ReplayBuffer()

def test_creation(rep_buf):
    assert len(rep_buf.states) == 0

def test_adding(rep_buf):
    s = np.ones((3, 8))
    a = np.ones((3, 3))
    r = np.ones((3))
    rep_buf.add_rollout(s, a, r)

    assert len(rep_buf.states) == 3
    assert len(rep_buf.actions) == 3
    assert len(rep_buf.rewards) == 3

    s = np.zeros((3, 8))
    a = np.zeros((3, 3))
    r = np.zeros((3))
    rep_buf.add_rollout(s, a, r)

    assert len(rep_buf.states) == 6
    assert len(rep_buf.actions) == 6
    assert len(rep_buf.rewards) == 6

def test_can_sample(rep_buf):
    assert rep_buf.can_sample(1) == False

    s = np.ones((3, 8))
    a = np.ones((3, 3))
    r = np.ones((3))
    rep_buf.add_rollout(s, a, r)

    assert rep_buf.can_sample(5) == False
    assert rep_buf.can_sample(1) == True

    rep_buf.add_rollout(s, a, r)

    assert rep_buf.can_sample(5) == True

def test_sampling(rep_buf):
    for i in range(1, 5):
        rep_buf.add_rollout(np.ones((1,3)), np.ones((1,2)), i*np.ones((1)))

    random.seed(42)
    _, _, r = rep_buf.sample(3)
    assert (r == [3, 2, 4]).all()
