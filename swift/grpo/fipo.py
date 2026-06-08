# Copyright (c) ModelScope Contributors. All rights reserved.
"""FIPO token-level influence weight computation.

Implements the Future-KL based influence weighting from the FIPO paper
(https://arxiv.org/abs/2603.19835). This is a pure tensor function with
no distributed communication or Trainer dependencies.
"""
from typing import Dict, Optional, Tuple

import torch


def compute_fipo_influence(
    log_ratio: torch.Tensor,
    coef_1: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    gamma: float,
    *,
    delta: Optional[float] = None,
    clip_range: Optional[float] = 0.2,
    clip_high_only: bool = True,
    safety_threshold: Optional[float] = 4.0,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute FIPO token-level influence weight from discounted Future-KL.

    Args:
        log_ratio: per_token_logps - old_per_token_logps, shape ``[B, T]``.
        coef_1: Importance weight ratios ``exp(log_importance_weights)``, shape ``[B, T]``.
        advantages: Per-sample advantages, shape ``[B]``.
        completion_mask: Boolean mask for completion tokens, shape ``[B, T]``.
        gamma: Decay factor, typically ``2 ** (-1 / fipo_decay_rate)``.
        delta: Dual-clip threshold (INTELLECT-2). Tokens with ``coef_1 > delta``
            do not contribute to Future-KL. ``None`` to disable.
        clip_range: Clip range for influence weight. ``None`` to disable clipping.
        clip_high_only: If ``True``, clip to ``[1, 1 + clip_range]``; otherwise
            ``[1 - clip_range, 1 + clip_range]``.
        safety_threshold: Safety threshold for negative-advantage tokens. Tokens
            with ``advantage < 0`` and ``coef_1 > safety_threshold`` have their
            influence weight capped to ``[0.8, 1.0]``. ``None`` to disable.
        chunk_size: Chunk size for the Future-KL matmul (memory optimisation).

    Returns:
        ``(influence_weight, metrics)`` where:

        - ``influence_weight``: Detached weight tensor, shape ``[B, T]``.
        - ``metrics``: Dict with ``'future_kl'``, ``'influence_weight'``, and
          ``'safety_mask'`` tensors (all ``[B, T]``).
    """
    future_kl_delta = log_ratio.masked_fill(~completion_mask, 0.0)

    if delta is not None:
        delta_t = torch.as_tensor(delta, dtype=log_ratio.dtype, device=log_ratio.device)
        high_ratio_mask = coef_1 > delta_t
        future_kl_delta = torch.where(high_ratio_mask, torch.zeros_like(future_kl_delta), future_kl_delta)

    seq_len = future_kl_delta.shape[1]
    future_kl = torch.zeros_like(future_kl_delta)
    positions = torch.arange(seq_len, device=log_ratio.device).unsqueeze(1)
    gamma_t = torch.as_tensor(gamma, dtype=log_ratio.dtype, device=log_ratio.device)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(seq_len, chunk_start + chunk_size)
        chunk_positions = torch.arange(chunk_start, chunk_end, device=log_ratio.device).unsqueeze(0)
        distance = chunk_positions - positions
        future_mask = distance >= 0
        decay_block = torch.pow(gamma_t, distance.clamp(min=0)) * future_mask.to(log_ratio.dtype)
        future_kl += torch.matmul(future_kl_delta[:, chunk_start:chunk_end], decay_block.t())

    future_kl = future_kl.masked_fill(~completion_mask, 0.0)
    influence_weight = torch.exp(future_kl)

    if clip_range:
        high = 1 + clip_range
        low = 1.0 if clip_high_only else 1 - clip_range
        influence_weight = torch.clamp(influence_weight, min=low, max=high)
    influence_weight = influence_weight.detach()

    safety_mask = torch.ones_like(completion_mask, dtype=torch.bool)
    if safety_threshold is not None:
        negative_advantage = advantages.unsqueeze(1) < 0
        high_is_ratio = coef_1 > safety_threshold
        safety_mask = ~(negative_advantage & high_is_ratio)
        influence_weight = torch.where(
            safety_mask, influence_weight, torch.clamp(influence_weight, min=0.8, max=1.0))

    metrics = {
        'future_kl': future_kl,
        'influence_weight': influence_weight,
        'safety_mask': safety_mask,
    }
    return influence_weight, metrics
