import torch
import pytest
import numpy as np
from types import SimpleNamespace
from algs.PPO.ppo import PPO

def test_ppo_loss_clipped_objective():
    # Use PPO directly since conftest.py mocks @ray.remote to be a no-op
    ppo_logic = PPO
    dummy_self = SimpleNamespace(
        clip_low=0.2, 
        clip_high=0.2, 
        ent_coeff=0.0, 
        kl_coeff=0.0,
        alg_name="Mock"
    )
    
    logprobs = torch.tensor([[-0.1, -0.2]])
    old_logprobs = torch.tensor([[-0.1, -0.2]])
    # Ratio = exp(0) = 1.0
    advantages = torch.tensor([[1.0, 2.0]])
    mask = torch.tensor([[1.0, 1.0]])
    
    # codebase logic
    loss, metrics = ppo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)
    
    # Loss = - (1.0 * 1.0 + 1.0 * 2.0) / 2 = -1.5
    assert np.isclose(metrics['pi_loss'], -1.5)
    assert np.isclose(metrics['clipfrac'], 0.0)
    assert np.isclose(metrics['approx_kl'], 0.0)

def test_ppo_loss_clipping():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(
        clip_low=0.2, 
        clip_high=0.2, 
        ent_coeff=0.0, 
        kl_coeff=0.0,
        alg_name="Mock"
    )
    
    logprobs = torch.tensor([[10.0]]) # Ratio will be exp(10)
    old_logprobs = torch.tensor([[0.0]])
    advantages = torch.tensor([[1.0]])
    mask = torch.tensor([[1.0]])
    
    loss, metrics = ppo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)
    
    # Ratio = exp(10) >> 1.2
    # Clipped = 1.2 * 1.0 = 1.2
    # Unclipped = exp(10) * 1.0 = VERY LARGE
    # Loss = -min(VERY LARGE, 1.2) = -1.2
    assert np.isclose(metrics['pi_loss'], -1.2)
    assert np.isclose(metrics['clipfrac'], 1.0)

def test_ppo_loss_entropy():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(
        clip_low=0.2, 
        clip_high=0.2, 
        ent_coeff=0.1, 
        kl_coeff=0.0,
        alg_name="Mock"
    )
    
    logprobs = torch.tensor([[-0.5]])
    old_logprobs = torch.tensor([[-0.5]])
    advantages = torch.tensor([[0.0]])
    mask = torch.tensor([[1.0]])
    entropies = torch.tensor([[0.7]])
    
    loss, metrics = ppo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, entropies, None)
    
    # Loss_pi = 0 (adv = 0)
    # Loss_ent = 0.7
    # Loss_total = 0 - 0.1 * 0.7 = -0.07
    assert np.isclose(metrics['pi_loss_total'], -0.07)

def test_ppo_loss_gradient_flow():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(
        clip_low=0.2, 
        clip_high=0.2, 
        ent_coeff=0.0, 
        kl_coeff=0.0,
        alg_name="Mock"
    )
    
    logprobs = torch.tensor([[-0.5]], requires_grad=True)
    old_logprobs = torch.tensor([[-0.5]])
    advantages = torch.tensor([[1.0]])
    mask = torch.tensor([[1.0]])
    
    loss, metrics = ppo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)
    loss.backward()
    
    assert logprobs.grad is not None
    assert not torch.isnan(logprobs.grad)
    assert logprobs.grad.item() != 0.0
