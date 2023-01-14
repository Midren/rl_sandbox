from rl_sandbox.utils.schedulers import LinearScheduler

def test_linear_schedule():
    s = LinearScheduler(0, 10, 5)
    assert s.step() == 0
    assert s.step() == 2.5
    assert s.step() == 5
    assert s.step() == 7.5
    assert s.step() == 10.0

def test_linear_schedule_after():
    s = LinearScheduler(0, 10, 5)
    for _ in range(5):
        s.step()
    assert s.step() == 10.0
