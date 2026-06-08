"""Tests for swift.grpo.reward — reward scoring functions."""
import pytest
import torch

from swift.grpo.reward import compute_rewards_per_func, score_with_gym

DEVICE = 'cpu'


def _dummy_reward_func(completions, **kwargs):
    return [len(c) / 100.0 for c in completions]


def _dummy_reward_none(completions, **kwargs):
    return [None for _ in completions]


class TestComputeRewardsPerFunc:

    def test_single_func(self):
        inputs = [{'messages': [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello world'}]}]
        completions = ['hello world']
        result = compute_rewards_per_func(
            reward_funcs=[_dummy_reward_func],
            reward_func_names=['length'],
            inputs=inputs,
            completions=completions,
            device=torch.device(DEVICE),
        )
        assert result.shape == (1, 1)
        assert result[0, 0].item() == pytest.approx(0.11, abs=0.01)

    def test_multiple_funcs(self):
        inputs = [{'messages': [{'role': 'assistant', 'content': 'a'}]},
                  {'messages': [{'role': 'assistant', 'content': 'bb'}]}]
        completions = ['a', 'bb']
        result = compute_rewards_per_func(
            reward_funcs=[_dummy_reward_func, _dummy_reward_func],
            reward_func_names=['len1', 'len2'],
            inputs=inputs,
            completions=completions,
            device=torch.device(DEVICE),
        )
        assert result.shape == (2, 2)

    def test_none_rewards_become_nan(self):
        inputs = [{'messages': [{'role': 'assistant', 'content': 'x'}]}]
        completions = ['x']
        result = compute_rewards_per_func(
            reward_funcs=[_dummy_reward_none],
            reward_func_names=['none_func'],
            inputs=inputs,
            completions=completions,
            device=torch.device(DEVICE),
        )
        assert torch.isnan(result[0, 0])


class TestScoreWithGym:

    def test_gym_only(self):
        inputs = [
            {'messages': [{'role': 'assistant', 'content': 'a'}], 'rollout_infos': {'total_reward': 1.0}},
            {'messages': [{'role': 'assistant', 'content': 'b'}], 'rollout_infos': {'total_reward': 2.0}},
        ]
        result = score_with_gym(inputs, reward_funcs=[], reward_func_names=[], device=torch.device(DEVICE),
                                use_gym_env=True)
        assert result.shape == (2, 1)
        assert result[0, 0].item() == 1.0
        assert result[1, 0].item() == 2.0

    def test_gym_plus_funcs(self):
        inputs = [
            {'messages': [{'role': 'assistant', 'content': 'hello'}], 'rollout_infos': {'total_reward': 5.0}},
        ]
        result = score_with_gym(
            inputs,
            reward_funcs=[_dummy_reward_func],
            reward_func_names=['len'],
            device=torch.device(DEVICE),
            use_gym_env=True,
            completions=['hello'],
        )
        assert result.shape == (1, 2)
        assert result[0, 1].item() == 5.0

    def test_no_gym(self):
        inputs = [{'messages': [{'role': 'assistant', 'content': 'hi'}]}]
        result = score_with_gym(
            inputs,
            reward_funcs=[_dummy_reward_func],
            reward_func_names=['len'],
            device=torch.device(DEVICE),
            use_gym_env=False,
            completions=['hi'],
        )
        assert result.shape == (1, 1)
