import pytest
import torch

from rl_sandbox.agents.dreamer_v2 import ImaginativeCritic

@pytest.fixture
def imaginative_critic():
    return ImaginativeCritic(discount_factor=1,
                             update_interval=100,
                             soft_update_fraction=1,
                             value_target_lambda=0.95,
                             latent_dim=10)

def test_lambda_return_discount_0(imaginative_critic):
    # Should just return rewards if discount_factor is 0
    imaginative_critic.lambda_ = 0
    imaginative_critic.gamma = 0
    rs = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    vs = torch.ones_like(rs)
    ts = torch.ones_like(rs)
    lambda_ret = imaginative_critic._lambda_return(vs, rs, ts)
    assert torch.all(lambda_ret == rs)

def test_lambda_return_lambda_0(imaginative_critic):
    # Should return 1-step return if lambda is 0
    imaginative_critic.lambda_ = 0
    imaginative_critic.gamma = 1
    vs = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rs = torch.ones_like(vs)
    ts = torch.ones_like(vs)
    lambda_ret = imaginative_critic._lambda_return(vs, rs, ts)
    assert torch.all(lambda_ret == torch.Tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))

def test_lambda_return_lambda_0_gamma_0_5(imaginative_critic):
    # Should return 1-step return if lambda is 0
    imaginative_critic.lambda_ = 0
    imaginative_critic.gamma = 0.5
    vs = torch.Tensor([2, 2, 4, 4, 6, 6, 8, 8, 10, 10])
    rs = torch.ones_like(vs)
    ts = torch.ones_like(vs)
    lambda_ret = imaginative_critic._lambda_return(vs, rs, ts)
    assert torch.all(lambda_ret == torch.Tensor([2, 2, 3, 3, 4, 4, 5, 5, 6, 6]))

def test_lambda_return_lambda_1(imaginative_critic):
    # Should return Monte-Carlo return
    imaginative_critic.lambda_ = 1
    imaginative_critic.gamma = 1
    vs = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rs = torch.ones_like(vs)
    ts = torch.ones_like(vs)
    lambda_ret = imaginative_critic._lambda_return(vs, rs, ts)
    assert torch.all(lambda_ret == torch.Tensor([20, 19, 18, 17, 16, 15, 14, 13, 12, 11]))

def test_lambda_return_lambda_1_gamma_0_5(imaginative_critic):
    # Should return Monte-Carlo return
    imaginative_critic.lambda_ = 1
    imaginative_critic.gamma = 0.5
    vs = torch.Tensor([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    rs = torch.zeros_like(vs)
    ts = torch.ones_like(vs)
    lambda_ret = imaginative_critic._lambda_return(vs, rs, ts)
    assert torch.all(lambda_ret == torch.Tensor([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))
