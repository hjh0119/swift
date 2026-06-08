# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reward scoring — unified across HF / Megatron / Ray backends.

Pure functions for computing rewards from reward functions. The only
external dependency beyond ``torch`` is ``asyncio`` for async reward
functions.
"""
import asyncio
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


def compute_rewards_per_func(
    reward_funcs: List[Callable],
    reward_func_names: List[str],
    inputs: Sequence[Dict[str, Any]],
    completions: List[str],
    device: torch.device,
    *,
    reward_model_plugins: Optional[List] = None,
    async_indices: Optional[List[int]] = None,
    async_loop=None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Compute rewards for all reward functions.

    Unified logic extracted from HF ``GRPOTrainer._compute_rewards_per_func``,
    Megatron ``MegatronGRPOTrainer._compute_rewards_per_func``, and
    Ray ``GRPOTrainer._compute_reward_funcs``.

    Args:
        reward_funcs: List of reward functions (callables or ``nn.Module``).
        reward_func_names: Display names for each function.
        inputs: Input data list (each element is a dict with messages, etc.).
        completions: Completion text list, length ``N``.
        device: Device for the output tensor.
        reward_model_plugins: Optional plugin callbacks for ``nn.Module`` reward
            models. When provided, ``reward_model_plugins[i](inputs=..., **kwargs)``
            is called instead of ``reward_funcs[i]``. Length must match ``reward_funcs``.
        async_indices: Indices of async reward functions. If ``None``, auto-detected.
        async_loop: ``asyncio`` event loop for running async reward functions.
            Required when async reward functions are present and the caller uses
            a dedicated thread event loop (HF/Megatron pattern). When ``None`` and
            async functions are detected, falls back to ``asyncio.run``.
        extra_kwargs: Additional keyword arguments passed to every reward function
            (e.g., ``{'trainer_state': state}``).

    Returns:
        Reward tensor of shape ``[N, n_funcs]``.
    """
    n = len(inputs)
    n_funcs = len(reward_funcs)
    rewards = torch.zeros((n, n_funcs), device=device)

    if reward_model_plugins is None:
        reward_model_plugins = [None] * n_funcs
    if extra_kwargs is None:
        extra_kwargs = {}

    if async_indices is None:
        async_indices = _detect_async_indices(reward_funcs)
    async_set = set(async_indices)

    reward_kwargs = dict(extra_kwargs)

    for i, (func, plugin, name) in enumerate(zip(reward_funcs, reward_model_plugins, reward_func_names)):
        if i in async_set:
            continue
        if isinstance(func, nn.Module) and plugin is not None:
            output = plugin(inputs=list(inputs), **reward_kwargs)
        else:
            output = func(completions, **reward_kwargs)
        output = [r if r is not None else torch.nan for r in output]
        rewards[:, i] = torch.tensor(output, dtype=torch.float32, device=device)

    if async_indices:
        _run_async_rewards(reward_funcs, reward_func_names, async_indices, completions, reward_kwargs, rewards, device,
                           async_loop)

    if torch.isnan(rewards).all(dim=1).any():
        from swift.utils import get_logger
        logger = get_logger()
        nan_idx = torch.isnan(rewards).all(dim=1).nonzero(as_tuple=True)[0][0]
        logger.warning(f'All reward functions returned None for sample index {nan_idx}. '
                       'Ensure at least one reward function returns a valid reward.')

    return rewards


def score_with_gym(
    inputs: Sequence[Dict[str, Any]],
    reward_funcs: List[Callable],
    reward_func_names: List[str],
    device: torch.device,
    use_gym_env: bool = False,
    **compute_kwargs,
) -> torch.Tensor:
    """Score completions, optionally appending gym total_reward as an extra column.

    Args:
        inputs: Input data list.
        reward_funcs: Reward function list.
        reward_func_names: Function names.
        device: Device.
        use_gym_env: If ``True``, extract ``rollout_infos['total_reward']``
            from each input and append as the last column.
        **compute_kwargs: Forwarded to :func:`compute_rewards_per_func`.

    Returns:
        ``[N, n_funcs + (1 if use_gym_env)]``.
    """
    completions = [inp['messages'][-1]['content'] for inp in inputs]

    if use_gym_env:
        gym_reward = torch.tensor(
            [inp['rollout_infos']['total_reward'] for inp in inputs], dtype=torch.float32, device=device).unsqueeze(1)
        if not reward_funcs:
            return gym_reward
        func_rewards = compute_rewards_per_func(
            reward_funcs, reward_func_names, inputs, completions, device, **compute_kwargs)
        return torch.cat([func_rewards, gym_reward], dim=1)

    return compute_rewards_per_func(reward_funcs, reward_func_names, inputs, completions, device, **compute_kwargs)


def _detect_async_indices(reward_funcs: List[Callable]) -> List[int]:
    indices = []
    for i, func in enumerate(reward_funcs):
        if not isinstance(func, nn.Module):
            if asyncio.iscoroutinefunction(func) or asyncio.iscoroutinefunction(getattr(func, '__call__', None)):
                indices.append(i)
    return indices


def _run_async_rewards(reward_funcs, reward_func_names, async_indices, completions, reward_kwargs, rewards, device,
                       async_loop):

    async def _invoke(index):
        func = reward_funcs[index]
        output = await func(completions, **reward_kwargs)
        output = [r if r is not None else torch.nan for r in output]
        return index, output

    async def _run_all():
        return await asyncio.gather(*[_invoke(idx) for idx in async_indices])

    if async_loop is not None:
        results = asyncio.run_coroutine_threadsafe(_run_all(), async_loop).result()
    else:
        results = asyncio.run(_run_all())

    for idx, output in results:
        rewards[:, idx] = torch.tensor(output, dtype=torch.float32, device=device)
