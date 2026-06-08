"""Tests for swift.grpo.loss — GRPO loss pure functions."""
import pytest
import torch

from swift.grpo.loss import (
    GRPOLossConfig,
    compute_clipping_metrics,
    compute_entropy_mask,
    compute_importance_weights,
    compute_kl_divergence,
    compute_per_token_loss,
    compute_real_loss,
    reduce_loss,
)

B, T = 4, 8
DEVICE = 'cpu'


def _make_tensors(seed=42):
    torch.manual_seed(seed)
    logps = torch.randn(B, T, device=DEVICE)
    old_logps = torch.randn(B, T, device=DEVICE)
    ref_logps = torch.randn(B, T, device=DEVICE)
    mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
    mask[:, -2:] = False
    advantages = torch.randn(B, device=DEVICE)
    return logps, old_logps, ref_logps, mask, advantages


class TestComputeImportanceWeights:

    def test_token_level(self):
        logps, old_logps, _, mask, _ = _make_tensors()
        coef, log_r = compute_importance_weights(logps, old_logps, mask, 'token')
        assert coef.shape == (B, T)
        assert log_r.shape == (B, T)
        torch.testing.assert_close(log_r, logps - old_logps)
        torch.testing.assert_close(coef, torch.exp(log_r))

    def test_sequence_level(self):
        logps, old_logps, _, mask, _ = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'sequence')
        assert coef.shape == (B, 1)

    def test_sequence_token_level(self):
        logps, old_logps, _, mask, _ = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'sequence_token')
        assert coef.shape == (B, T)

    def test_invalid_level_raises(self):
        logps, old_logps, _, mask, _ = _make_tensors()
        with pytest.raises(ValueError, match='Unknown importance sampling level'):
            compute_importance_weights(logps, old_logps, mask, 'invalid')


class TestComputePerTokenLoss:

    @pytest.mark.parametrize('loss_type', ['grpo', 'bnpo', 'dr_grpo', 'dapo'])
    def test_ppo_family(self, loss_type):
        logps, old_logps, _, mask, adv = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'token')
        ptl = compute_per_token_loss(loss_type, coef, adv, logps, epsilon_low=0.2, epsilon_high=0.2)
        assert ptl.shape == (B, T)
        assert torch.isfinite(ptl).all()

    def test_cispo(self):
        logps, old_logps, _, mask, adv = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'token')
        ptl = compute_per_token_loss('cispo', coef, adv, logps, epsilon_high=0.3)
        assert ptl.shape == (B, T)

    def test_sapo(self):
        logps, old_logps, _, mask, adv = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'token')
        ptl = compute_per_token_loss('sapo', coef, adv, logps, tau_pos=1.0, tau_neg=1.05)
        assert ptl.shape == (B, T)

    def test_fipo_with_weight(self):
        logps, old_logps, _, mask, adv = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'token')
        fipo_w = torch.ones(B, T, device=DEVICE)
        ptl = compute_per_token_loss('fipo', coef, adv, logps, fipo_weight=fipo_w)
        assert ptl.shape == (B, T)

    def test_delta_clipping(self):
        logps, old_logps, _, mask, adv = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'token')
        ptl = compute_per_token_loss('grpo', coef, adv, logps, delta=1.5)
        assert ptl.shape == (B, T)

    def test_unknown_loss_raises(self):
        logps, old_logps, _, mask, adv = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'token')
        with pytest.raises(ValueError, match='Unknown loss type'):
            compute_per_token_loss('nonexistent', coef, adv, logps)


class TestReduceLoss:

    def _make_ptl(self):
        logps, old_logps, _, mask, adv = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'token')
        ptl = compute_per_token_loss('grpo', coef, adv, logps)
        return ptl, mask

    @pytest.mark.parametrize('loss_type', ['grpo', 'sapo'])
    def test_per_sample_mean(self, loss_type):
        ptl, mask = self._make_ptl()
        loss = reduce_loss(loss_type, ptl, mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_bnpo(self):
        ptl, mask = self._make_ptl()
        loss = reduce_loss('bnpo', ptl, mask)
        assert loss.dim() == 0

    def test_dr_grpo(self):
        ptl, mask = self._make_ptl()
        loss = reduce_loss('dr_grpo', ptl, mask, batch_size=B, max_completion_length=T)
        assert loss.dim() == 0

    def test_dapo_with_num_items(self):
        ptl, mask = self._make_ptl()
        num_items = torch.tensor(24.0)
        loss = reduce_loss('dapo', ptl, mask, num_items_in_batch=num_items, dp_size=1)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_unknown_loss_raises(self):
        ptl, mask = self._make_ptl()
        with pytest.raises(ValueError):
            reduce_loss('nonexistent', ptl, mask)


class TestComputeRealLoss:

    def test_basic(self):
        logps, old_logps, _, mask, adv = _make_tensors()
        log_ratio = logps - old_logps
        loss = compute_real_loss(log_ratio, adv, mask, num_generations=2, real_tau=0.5)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_with_kl(self):
        logps, old_logps, ref_logps, mask, adv = _make_tensors()
        log_ratio = logps - old_logps
        kl = compute_kl_divergence(ref_logps, logps)
        loss = compute_real_loss(log_ratio, adv, mask, num_generations=2, real_tau=0.5, per_token_kl=kl, beta=0.04)
        assert loss.dim() == 0

    def test_all_same_sign_advantages(self):
        logps, old_logps, _, mask, _ = _make_tensors()
        adv_positive = torch.ones(B, device=DEVICE)
        log_ratio = logps - old_logps
        loss = compute_real_loss(log_ratio, adv_positive, mask, num_generations=2, real_tau=0.5)
        assert loss.dim() == 0


class TestComputeKLDivergence:

    def test_shape(self):
        _, _, ref_logps, _, _ = _make_tensors()
        logps = torch.randn(B, T, device=DEVICE)
        kl = compute_kl_divergence(ref_logps, logps)
        assert kl.shape == (B, T)

    def test_non_negative(self):
        """KL via exp(r)-r-1 is >= 0 for all r (minimum at r=0)."""
        logps, _, ref_logps, _, _ = _make_tensors()
        kl = compute_kl_divergence(ref_logps, logps)
        assert (kl >= -1e-5).all(), f'KL has negative values: {kl.min()}'

    def test_zero_when_equal(self):
        logps = torch.randn(B, T, device=DEVICE)
        kl = compute_kl_divergence(logps, logps)
        torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-6, rtol=0)


class TestComputeEntropyMask:

    def test_no_filter(self):
        entropies = torch.rand(B, T, device=DEVICE)
        mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
        entropy_mask, threshold = compute_entropy_mask(entropies, mask, 1.0)
        assert entropy_mask is None
        assert threshold is None

    def test_with_filter(self):
        torch.manual_seed(42)
        entropies = torch.rand(B, T, device=DEVICE)
        mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
        entropy_mask, threshold = compute_entropy_mask(entropies, mask, 0.5)
        assert entropy_mask is not None
        assert entropy_mask.shape == (B, T)
        assert threshold is not None
        assert 0 < threshold < 1
        kept_frac = entropy_mask.float().mean().item()
        assert 0.3 < kept_frac < 0.7


class TestComputeClippingMetrics:

    def test_grpo(self):
        logps, old_logps, _, mask, adv = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'token')
        result = compute_clipping_metrics('grpo', coef, adv, mask, 0.2, 0.2)
        assert 'low_clip' in result
        assert 'high_clip' in result
        assert 'region_clip' in result

    def test_cispo(self):
        logps, old_logps, _, mask, adv = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'token')
        result = compute_clipping_metrics('cispo', coef, adv, mask, 0.2, 0.2)
        assert 'cispo_clip_ratio' in result

    def test_sapo_empty(self):
        logps, old_logps, _, mask, adv = _make_tensors()
        coef, _ = compute_importance_weights(logps, old_logps, mask, 'token')
        result = compute_clipping_metrics('sapo', coef, adv, mask, 0.2, 0.2)
        assert result == {}


class TestGRPOLossConfig:

    def test_defaults(self):
        cfg = GRPOLossConfig(loss_type='grpo', beta=0.04, epsilon_low=0.2, epsilon_high=0.2)
        assert cfg.loss_type == 'grpo'
        assert cfg.importance_sampling_level == 'token'
        assert cfg.delta is None
