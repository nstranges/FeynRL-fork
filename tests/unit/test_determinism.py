import torch
import torch.nn as nn
import numpy as np
import pytest
from tests.models import TinyModel, TinyValueModel
from algs.PPO.ppo import PPO
from types import SimpleNamespace

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test_sampling_determinism():
    vocab_size = 50
    hidden_dim = 16
    set_seed(42)
    model = TinyModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    model.eval()
    input_ids = torch.randint(0, vocab_size, (2, 8))
    
    # Forward pass outputs
    out1 = model(input_ids).logits
    
    set_seed(42)
    model = TinyModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    model.eval()
    out2 = model(input_ids).logits
    
    assert torch.equal(out1, out2)

def test_loss_determinism():
    def compute_mock_loss():
        set_seed(42)
        vocab_size = 50
        hidden_dim = 16
        model = TinyModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
        input_ids = torch.randint(0, vocab_size, (2, 8))
        target_ids = torch.randint(0, vocab_size, (2, 8))
        
        logits = model(input_ids).logits
        loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
        return loss
    
    loss1 = compute_mock_loss()
    loss2 = compute_mock_loss()
    
    assert torch.equal(loss1, loss2)

def test_gae_determinism():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95)
    
    def run_gae():
        torch.manual_seed(42)
        B, T = 4, 8
        rewards = torch.randn(B, T)
        values = torch.randn(B, T)
        done = torch.randint(0, 2, (B, T)).float()
        mask = torch.ones(B, T)
        return ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)
    
    rets1, advs1 = run_gae()
    rets2, advs2 = run_gae()
    
    assert torch.equal(rets1, rets2)
    assert torch.equal(advs1, advs2)
