"""Tests for swift.grpo.rollout_is — rollout importance sampling functions."""
import pytest
import torch

from swift.grpo.rollout_is import (
    apply_rollout_importance_sampling,
    compute_is_metrics,
    compute_off_policy_sequence_mask,
    compute_offpolicy_metrics,
    compute_sequence_level_ratios,
)

B, T = 4, 8
DEVICE = 'cpu'


def _make_tensors(seed=42):
    torch.manual_seed(seed)
    logps = torch.randn(B, T, device=DEVICE)
    old_logps = torch.randn(B, T, device=DEVICE)
    mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
    mask[:, -2:] = False
    return logps, old_logps, mask


class TestComputeSequenceLevelRatios:

    def test_shape(self):
        _, _, mask = _make_tensors()
        is_ratio = torch.ones(B, T, device=DEVICE) * 1.5
        result = compute_sequence_level_ratios(is_ratio, mask)
        assert result.shape == (B,)

    def test_uniform_ratio(self):
        mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
        is_ratio = torch.ones(B, T, device=DEVICE) * 2.0
        result = compute_sequence_level_ratios(is_ratio, mask)
        torch.testing.assert_close(result, torch.ones(B, device=DEVICE) * 2.0, atol=1e-5, rtol=0)


class TestApplyRolloutImportanceSampling:

    @pytest.mark.parametrize('mode', ['token_truncate', 'token_mask', 'sequence_truncate', 'sequence_mask'])
    def test_shape(self, mode):
        _, _, mask = _make_tensors()
        log_ratio = torch.randn(B, T, device=DEVICE)
        weights = apply_rollout_importance_sampling(log_ratio, mask, mode, threshold=2.0)
        assert weights.shape == (B, T)

    def test_token_truncate_clamps(self):
        mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
        log_ratio = torch.ones(B, T, device=DEVICE) * 5.0
        weights = apply_rollout_importance_sampling(log_ratio, mask, 'token_truncate', threshold=2.0)
        assert (weights <= 2.0 + 1e-6).all()

    def test_token_mask_zeros(self):
        mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
        log_ratio = torch.ones(B, T, device=DEVICE) * 5.0
        weights = apply_rollout_importance_sampling(log_ratio, mask, 'token_mask', threshold=2.0)
        assert (weights == 0).all()

    def test_small_ratio_passes_through(self):
        mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
        log_ratio = torch.zeros(B, T, device=DEVICE)
        weights = apply_rollout_importance_sampling(log_ratio, mask, 'token_truncate', threshold=2.0)
        torch.testing.assert_close(weights, torch.ones(B, T, device=DEVICE), atol=1e-5, rtol=0)


class TestComputeOffPolicySequenceMask:

    def test_shape(self):
        logps, old_logps, mask = _make_tensors()
        adv = torch.randn(B, device=DEVICE)
        result = compute_off_policy_sequence_mask(logps, old_logps, mask, adv, delta=0.5)
        assert result.shape == (B,)
        assert result.dtype == torch.bool

    def test_keeps_positive_advantage(self):
        mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
        logps = torch.zeros(B, T, device=DEVICE)
        old_logps = torch.ones(B, T, device=DEVICE) * 10
        adv = torch.ones(B, device=DEVICE)
        result = compute_off_policy_sequence_mask(logps, old_logps, mask, adv, delta=0.5)
        assert result.all()

    def test_masks_negative_high_delta(self):
        mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
        logps = torch.zeros(B, T, device=DEVICE)
        old_logps = torch.ones(B, T, device=DEVICE) * 10
        adv = -torch.ones(B, device=DEVICE)
        result = compute_off_policy_sequence_mask(logps, old_logps, mask, adv, delta=0.5)
        assert not result.any()


class TestComputeOffpolicyMetrics:

    def test_keys(self):
        logps, old_logps, mask = _make_tensors()
        metrics = compute_offpolicy_metrics(logps, old_logps, mask)
        expected_keys = {
            'training_ppl', 'training_log_ppl', 'kl', 'k3_kl', 'rollout_ppl', 'rollout_log_ppl', 'log_ppl_diff',
            'log_ppl_abs_diff', 'log_ppl_diff_max', 'log_ppl_diff_min', 'ppl_ratio', 'chi2_token', 'chi2_seq'
        }
        assert set(metrics.keys()) == expected_keys

    def test_values_finite(self):
        logps, old_logps, mask = _make_tensors()
        metrics = compute_offpolicy_metrics(logps, old_logps, mask)
        for k, v in metrics.items():
            assert torch.isfinite(v), f'{k} is not finite: {v}'

    def test_kl_zero_when_equal(self):
        logps = torch.randn(B, T, device=DEVICE)
        mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
        metrics = compute_offpolicy_metrics(logps, logps, mask)
        torch.testing.assert_close(metrics['kl'], torch.tensor(0.0), atol=1e-5, rtol=0)
        torch.testing.assert_close(metrics['k3_kl'], torch.tensor(0.0), atol=1e-5, rtol=0)


class TestComputeIsMetrics:

    def test_keys(self):
        _, _, mask = _make_tensors()
        log_ratio = torch.randn(B, T, device=DEVICE)
        is_weights = torch.ones(B, T, device=DEVICE)
        metrics = compute_is_metrics(log_ratio, is_weights, mask, 'token_truncate', 2.0)
        assert 'is_weight_mean' in metrics
        assert 'ess' in metrics
        assert 'clipped_frac' in metrics

    @pytest.mark.parametrize('mode', ['token_truncate', 'token_mask', 'sequence_truncate', 'sequence_mask'])
    def test_all_modes(self, mode):
        _, _, mask = _make_tensors()
        log_ratio = torch.randn(B, T, device=DEVICE)
        is_weights = apply_rollout_importance_sampling(log_ratio, mask, mode, 2.0)
        metrics = compute_is_metrics(log_ratio, is_weights, mask, mode, 2.0)
        for k, v in metrics.items():
            assert torch.isfinite(v), f'{k} not finite for mode {mode}'
