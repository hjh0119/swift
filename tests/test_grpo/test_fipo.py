"""Tests for swift.grpo.fipo — FIPO influence weight computation."""
import pytest
import torch

from swift.grpo.fipo import compute_fipo_influence

B, T = 4, 16
DEVICE = 'cpu'
GAMMA = 2**(-1 / 32.0)


def _make_tensors(seed=42):
    torch.manual_seed(seed)
    log_ratio = torch.randn(B, T, device=DEVICE) * 0.1
    coef_1 = torch.exp(log_ratio)
    advantages = torch.randn(B, device=DEVICE)
    mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
    mask[:, -3:] = False
    return log_ratio, coef_1, advantages, mask


class TestComputeFipoInfluence:

    def test_shape(self):
        log_ratio, coef_1, adv, mask = _make_tensors()
        weight, metrics = compute_fipo_influence(log_ratio, coef_1, adv, mask, GAMMA)
        assert weight.shape == (B, T)
        assert 'future_kl' in metrics
        assert 'influence_weight' in metrics
        assert 'safety_mask' in metrics
        assert metrics['future_kl'].shape == (B, T)

    def test_weight_is_detached(self):
        log_ratio, coef_1, adv, mask = _make_tensors()
        weight, _ = compute_fipo_influence(log_ratio, coef_1, adv, mask, GAMMA)
        assert not weight.requires_grad

    def test_clipping(self):
        log_ratio, coef_1, adv, mask = _make_tensors()
        weight, _ = compute_fipo_influence(log_ratio, coef_1, adv, mask, GAMMA, clip_range=0.2, clip_high_only=False)
        assert (weight >= 0.8 - 1e-6).all()
        assert (weight <= 1.2 + 1e-6).all()

    def test_clip_high_only(self):
        log_ratio, coef_1, adv, mask = _make_tensors()
        weight, _ = compute_fipo_influence(log_ratio, coef_1, adv, mask, GAMMA, clip_range=0.2, clip_high_only=True)
        assert (weight >= 1.0 - 1e-6).all()
        assert (weight <= 1.2 + 1e-6).all()

    def test_no_clipping(self):
        log_ratio, coef_1, adv, mask = _make_tensors()
        weight, _ = compute_fipo_influence(log_ratio, coef_1, adv, mask, GAMMA, clip_range=None)
        assert weight.shape == (B, T)

    def test_delta(self):
        torch.manual_seed(0)
        log_ratio = torch.randn(B, T, device=DEVICE)
        coef_1 = torch.exp(log_ratio)
        adv = torch.randn(B, device=DEVICE)
        mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
        weight_no_delta, _ = compute_fipo_influence(log_ratio, coef_1, adv, mask, GAMMA, delta=None, clip_range=None)
        weight_delta, _ = compute_fipo_influence(log_ratio, coef_1, adv, mask, GAMMA, delta=1.2, clip_range=None)
        assert not torch.allclose(weight_no_delta, weight_delta)

    def test_safety_threshold(self):
        log_ratio, coef_1, adv, mask = _make_tensors()
        _, metrics = compute_fipo_influence(log_ratio, coef_1, adv, mask, GAMMA, safety_threshold=0.5)
        safety_mask = metrics['safety_mask']
        assert safety_mask.shape == (B, T)
        assert safety_mask.dtype == torch.bool

    def test_no_safety(self):
        log_ratio, coef_1, adv, mask = _make_tensors()
        _, metrics = compute_fipo_influence(log_ratio, coef_1, adv, mask, GAMMA, safety_threshold=None)
        assert metrics['safety_mask'].all()

    def test_masked_positions_zero(self):
        log_ratio, coef_1, adv, mask = _make_tensors()
        _, metrics = compute_fipo_influence(log_ratio, coef_1, adv, mask, GAMMA)
        future_kl = metrics['future_kl']
        assert (future_kl[:, -3:] == 0).all()

    def test_zero_log_ratio_gives_unit_weight(self):
        log_ratio = torch.zeros(B, T, device=DEVICE)
        coef_1 = torch.ones(B, T, device=DEVICE)
        adv = torch.randn(B, device=DEVICE)
        mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
        weight, metrics = compute_fipo_influence(log_ratio, coef_1, adv, mask, GAMMA, clip_range=None)
        torch.testing.assert_close(weight, torch.ones(B, T, device=DEVICE), atol=1e-5, rtol=0)
