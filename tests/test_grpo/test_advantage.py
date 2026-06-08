"""Tests for swift.grpo.advantage — advantage computation pure functions."""
import pytest
import torch

from swift.grpo.advantage import (
    RewardMetrics,
    compute_advantages,
    compute_advantages_dynamic,
    compute_reward_metrics,
)

DEVICE = 'cpu'


class TestComputeAdvantages:

    def _make_rewards(self, N=8, n_funcs=3, seed=42):
        torch.manual_seed(seed)
        rpf = torch.randn(N, n_funcs, device=DEVICE)
        rw = torch.ones(n_funcs, device=DEVICE)
        return rpf, rw

    @pytest.mark.parametrize('estimator', ['grpo', 'rloo', 'reinforce_plus_plus'])
    def test_estimator_shapes(self, estimator):
        rpf, rw = self._make_rewards()
        adv, rew = compute_advantages(rpf, rw, num_generations=4, advantage_estimator=estimator)
        assert adv.shape == (8,)
        assert rew.shape == (8,)

    @pytest.mark.parametrize('scale', ['group', 'batch', 'none', 'gdpo'])
    def test_scale_rewards(self, scale):
        rpf, rw = self._make_rewards()
        adv, rew = compute_advantages(rpf, rw, num_generations=4, scale_rewards=scale)
        assert adv.shape == (8,)
        assert torch.isfinite(adv).all()

    def test_grpo_group_advantages_sum_zero_per_group(self):
        """GRPO advantages within each group should sum to ~0 (before normalization)."""
        rpf, rw = self._make_rewards()
        adv, _ = compute_advantages(rpf, rw, num_generations=4, scale_rewards='none')
        grouped = adv.view(-1, 4)
        group_sums = grouped.sum(dim=1)
        torch.testing.assert_close(group_sums, torch.zeros_like(group_sums), atol=1e-5, rtol=0)

    def test_rloo_leave_one_out(self):
        rpf, rw = self._make_rewards(N=4, n_funcs=1)
        adv, rew = compute_advantages(rpf, rw, num_generations=4, advantage_estimator='rloo', scale_rewards='none')
        K = 4
        mean_all = rew.mean()
        expected_0 = rew[0] * K / (K - 1) - mean_all * K / (K - 1)
        torch.testing.assert_close(adv[0], expected_0, atol=1e-5, rtol=0)

    def test_kl_in_reward(self):
        rpf, rw = self._make_rewards()
        kl = torch.ones(8, device=DEVICE) * 0.5
        adv_no_kl, rew_no_kl = compute_advantages(rpf, rw, num_generations=4, kl_in_reward=False, beta=0.04)
        adv_kl, rew_kl = compute_advantages(rpf, rw, num_generations=4, kl_in_reward=True, beta=0.04, kl_values=kl)
        assert not torch.allclose(rew_no_kl, rew_kl)

    def test_single_generation(self):
        rpf, rw = self._make_rewards(N=4)
        adv, _ = compute_advantages(rpf, rw, num_generations=1, scale_rewards='none')
        torch.testing.assert_close(adv, torch.zeros(4, device=DEVICE), atol=1e-5, rtol=0)


class TestComputeAdvantagesDynamic:

    def test_basic(self):
        torch.manual_seed(42)
        rpf = torch.randn(6, 2, device=DEVICE)
        rw = torch.ones(2, device=DEVICE)
        prompt_ids = ['p1', 'p1', 'p1', 'p2', 'p2', 'p2']
        request_ids = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
        adv, rew = compute_advantages_dynamic(rpf, rw, prompt_ids, request_ids)
        assert adv.shape == (6,)

    def test_duplicate_request_ids(self):
        torch.manual_seed(42)
        rpf = torch.randn(4, 1, device=DEVICE)
        rw = torch.ones(1, device=DEVICE)
        prompt_ids = ['p1', 'p1', 'p1', 'p1']
        request_ids = ['r1', 'r1', 'r2', 'r2']
        adv, rew = compute_advantages_dynamic(rpf, rw, prompt_ids, request_ids)
        assert adv.shape == (4,)
        assert adv[0] == adv[1]
        assert adv[2] == adv[3]


class TestComputeRewardMetrics:

    def test_basic(self):
        torch.manual_seed(42)
        N, K, n_funcs = 8, 4, 3
        rpf = torch.randn(N, n_funcs, device=DEVICE)
        rw = torch.ones(n_funcs, device=DEVICE)
        rewards = (rpf * rw.unsqueeze(0)).sum(dim=1)
        rm = compute_reward_metrics(rewards, rpf, ['r1', 'r2', 'r3'], K, 'group')
        assert isinstance(rm, RewardMetrics)
        assert isinstance(rm.reward_mean, float)
        assert isinstance(rm.reward_std, float)
        assert len(rm.per_func_mean) == 3
        assert len(rm.per_func_std) == 3

    def test_single_generation(self):
        rpf = torch.randn(4, 2, device=DEVICE)
        rw = torch.ones(2, device=DEVICE)
        rewards = (rpf * rw.unsqueeze(0)).sum(dim=1)
        rm = compute_reward_metrics(rewards, rpf, ['a', 'b'], 1, 'group')
        assert rm.reward_std == 0.0
