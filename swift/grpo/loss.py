# Copyright (c) ModelScope Contributors. All rights reserved.
"""GRPO loss computation — pure tensor functions.

Extracted from HF ``GRPOTrainer._compute_loss_and_metrics`` and Megatron
``MegatronGRPOTrainer.loss_func``. Both implementations share identical
per-token loss and reduction logic; only the I/O (how logps are obtained)
and metric gathering (accelerate vs mpu) differ.

All functions here operate on local tensors and perform **no** distributed
communication. The caller (Trainer) is responsible for gather / all-reduce.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .fipo import compute_fipo_influence


@dataclass
class GRPOLossConfig:
    """All configuration for GRPO loss computation.

    Constructed once during Trainer init and passed to the compute functions.
    """
    loss_type: str
    beta: float
    epsilon_low: float
    epsilon_high: float
    delta: Optional[float] = None
    importance_sampling_level: str = 'token'
    kl_in_reward: bool = False
    overlong_filter: bool = False
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    real_tau: float = 0.5
    num_generations: int = 8
    max_completion_length: int = 512
    fipo_gamma: float = 0.0
    fipo_clip_range: Optional[float] = 0.2
    fipo_clip_high_only: bool = True
    fipo_safety_threshold: Optional[float] = 4.0


# ---------------------------------------------------------------------------
# Importance weights
# ---------------------------------------------------------------------------

def compute_importance_weights(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    level: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute importance sampling weights for the policy ratio.

    Args:
        per_token_logps: Current policy log probs, ``[B, T]``.
        old_per_token_logps: Old policy log probs, ``[B, T]``.
        completion_mask: Completion token mask, ``[B, T]``.
        level: ``'token'``, ``'sequence'``, or ``'sequence_token'``.

    Returns:
        ``(coef_1, log_ratio)`` where ``coef_1 = exp(log_importance_weights)``.
    """
    log_ratio = per_token_logps - old_per_token_logps

    if level == 'token':
        log_importance_weights = log_ratio
    elif level in ('sequence', 'sequence_token'):
        seq_level = ((log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
        if level == 'sequence':
            log_importance_weights = seq_level
        else:
            log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level.detach()
    else:
        raise ValueError(f"Unknown importance sampling level: {level}. "
                         "Expected 'token', 'sequence', or 'sequence_token'.")

    coef_1 = torch.exp(log_importance_weights)
    return coef_1, log_ratio


# ---------------------------------------------------------------------------
# Per-token loss
# ---------------------------------------------------------------------------

def compute_per_token_loss(
    loss_type: str,
    coef_1: torch.Tensor,
    advantages: torch.Tensor,
    per_token_logps: torch.Tensor,
    *,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    delta: Optional[float] = None,
    tau_pos: float = 1.0,
    tau_neg: float = 1.05,
    fipo_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-token loss (before KL / entropy / rollout-IS modifiers).

    Supports: ``grpo``, ``bnpo``, ``dr_grpo``, ``dapo``, ``fipo``,
    ``cispo``, ``sapo``. REAL uses :func:`compute_real_loss` instead.

    Args:
        loss_type: One of the supported loss types.
        coef_1: Importance weight ratios ``exp(log_importance_weights)``, ``[B, T]`` or ``[B, 1]``.
        advantages: Per-sample advantages, ``[B]``.
        per_token_logps: Current policy log probs (needed by ``cispo``), ``[B, T]``.
        epsilon_low: Lower clipping bound.
        epsilon_high: Upper clipping bound.
        delta: Dual-clip upper bound (INTELLECT-2). ``None`` to disable.
        tau_pos: SAPO positive temperature.
        tau_neg: SAPO negative temperature.
        fipo_weight: Pre-computed FIPO influence weight, ``[B, T]``. Required when ``loss_type='fipo'``.

    Returns:
        ``per_token_loss``, shape ``[B, T]``.
    """
    if loss_type == 'cispo':
        clamped_ratios = torch.clamp(coef_1, max=epsilon_high).detach()
        return -clamped_ratios * advantages.unsqueeze(1) * per_token_logps

    if loss_type == 'sapo':
        adv = advantages.unsqueeze(1)
        gate_pos = torch.sigmoid(tau_pos * (coef_1 - 1)) * (4.0 / tau_pos)
        gate_neg = torch.sigmoid(tau_neg * (coef_1 - 1)) * (4.0 / tau_neg)
        soft_gate = torch.where(adv > 0, gate_pos, gate_neg)
        return -soft_gate * adv

    if loss_type in ('grpo', 'bnpo', 'dr_grpo', 'dapo', 'fipo'):
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
        c1 = coef_1
        if delta is not None:
            c1 = torch.clamp(c1, max=delta)
        per_token_loss1 = c1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if loss_type == 'fipo' and fipo_weight is not None:
            per_token_loss = per_token_loss * fipo_weight
        return per_token_loss

    raise ValueError(f'Unknown loss type for compute_per_token_loss: {loss_type}')


# ---------------------------------------------------------------------------
# Loss reduction
# ---------------------------------------------------------------------------

def reduce_loss(
    loss_type: str,
    per_token_loss: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    batch_size: Optional[int] = None,
    max_completion_length: Optional[int] = None,
    num_items_in_batch: Optional[torch.Tensor] = None,
    dp_size: int = 1,
) -> torch.Tensor:
    """Reduce per-token loss to a scalar.

    Args:
        loss_type: Loss type (determines reduction strategy).
        per_token_loss: ``[B, T]``.
        completion_mask: ``[B, T]``.
        batch_size: Micro batch size (``dr_grpo``).
        max_completion_length: Max completion length (``dr_grpo``).
        num_items_in_batch: Total completion tokens across all processes (``cispo/dapo/fipo``).
        dp_size: Data parallel world size (``cispo/dapo/fipo``).

    Returns:
        Scalar loss tensor.
    """
    if loss_type in ('grpo', 'sapo'):
        return ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

    if loss_type == 'bnpo':
        return (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    if loss_type == 'dr_grpo':
        bs = batch_size if batch_size is not None else completion_mask.shape[0]
        mcl = max_completion_length if max_completion_length is not None else completion_mask.shape[1]
        return (per_token_loss * completion_mask).sum() / (bs * mcl)

    if loss_type in ('cispo', 'dapo', 'fipo'):
        normalizer = num_items_in_batch / dp_size
        return (per_token_loss * completion_mask).sum() / normalizer.clamp(min=1.0)

    raise ValueError(f'Unknown loss type for reduce_loss: {loss_type}')


# ---------------------------------------------------------------------------
# REAL loss
# ---------------------------------------------------------------------------

def compute_real_loss(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    num_generations: int,
    real_tau: float,
    *,
    per_token_kl: Optional[torch.Tensor] = None,
    beta: float = 0.0,
) -> torch.Tensor:
    """Compute REAL loss (https://arxiv.org/abs/2602.05630).

    REAL has a fundamentally different reduction: it operates on group-level
    positive/negative score partitions rather than per-token clipping.

    Args:
        log_ratio: ``per_token_logps - old_per_token_logps``, ``[B, T]``.
        advantages: Per-sample advantages, ``[B]``.
        completion_mask: ``[B, T]``.
        num_generations: ``K`` — completions per prompt.
        real_tau: Temperature parameter.
        per_token_kl: Optional KL penalty, ``[B, T]``.
        beta: KL coefficient.

    Returns:
        Scalar loss tensor.
    """
    global_scores = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
    group_scores = global_scores.view(-1, num_generations)
    group_rewards = advantages.view(-1, num_generations)

    pos_mask = group_rewards > 0
    neg_mask = group_rewards <= 0
    valid_mask = (pos_mask.sum(dim=1) != 0) & (neg_mask.sum(dim=1) != 0)

    if not valid_mask.any():
        loss = torch.tensor(0.0, device=global_scores.device) * global_scores.mean()
    else:
        batch_scores = group_scores[valid_mask]
        batch_pos_mask = pos_mask[valid_mask]
        batch_neg_mask = neg_mask[valid_mask]

        scaled_scores = batch_scores / real_tau
        zeros = torch.zeros(batch_scores.size(0), 1, device=batch_scores.device, dtype=batch_scores.dtype)

        neg_input = scaled_scores.masked_fill(~batch_neg_mask, float('-inf'))
        neg_loss = torch.logsumexp(torch.cat([neg_input, zeros], dim=1), dim=1)

        pos_input = (-scaled_scores).masked_fill(~batch_pos_mask, float('-inf'))
        pos_loss = torch.logsumexp(torch.cat([pos_input, zeros], dim=1), dim=1)

        loss = (neg_loss + pos_loss).sum() / group_rewards.size(0)

    if beta != 0.0 and per_token_kl is not None:
        kl_loss = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        loss = loss + kl_loss * beta

    return loss


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def compute_kl_divergence(
    ref_logps: torch.Tensor,
    cur_logps: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token KL divergence using the ``exp(r) - r - 1`` estimator.

    This is numerically more stable than the naive ``p * log(p/q)`` form.

    Args:
        ref_logps: Reference model log probs, ``[B, T]``.
        cur_logps: Current model log probs, ``[B, T]``.

    Returns:
        Per-token KL, ``[B, T]``.
    """
    safe_ratio = torch.clamp(ref_logps - cur_logps, min=-20, max=20)
    return torch.clamp(torch.exp(safe_ratio) - safe_ratio - 1, min=-10, max=10)


# ---------------------------------------------------------------------------
# Entropy mask
# ---------------------------------------------------------------------------

def compute_entropy_mask(
    entropies: torch.Tensor,
    completion_mask: torch.Tensor,
    top_entropy_quantile: float,
) -> Tuple[Optional[torch.Tensor], Optional[float]]:
    """Compute entropy threshold and mask for top-quantile token selection.

    Args:
        entropies: Per-token entropy, ``[B, T]``. Padded tokens should be NaN.
        completion_mask: ``[B, T]``.
        top_entropy_quantile: Quantile in ``(0, 1]``. ``1.0`` means no filtering.

    Returns:
        ``(entropy_mask, threshold)`` — mask is ``None`` when quantile is 1.0.
    """
    if top_entropy_quantile >= 1.0:
        return None, None
    threshold = torch.nanquantile(entropies.flatten().float(), 1 - top_entropy_quantile).item()
    mask = entropies >= threshold
    return mask, threshold


# ---------------------------------------------------------------------------
# Clipping metrics
# ---------------------------------------------------------------------------

def compute_clipping_metrics(
    loss_type: str,
    coef_1: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
) -> Dict[str, torch.Tensor]:
    """Compute clipping ratio metrics (local tensors, not gathered).

    Args:
        loss_type: Loss type.
        coef_1: Importance weight ratios, ``[B, T]`` or ``[B, 1]``.
        advantages: ``[B]``.
        completion_mask: ``[B, T]``.
        epsilon_low: Lower clip bound.
        epsilon_high: Upper clip bound.

    Returns:
        Dict of boolean/float tensors. Trainer gathers and logs them.
    """
    token_count = completion_mask.sum().clamp(min=1.0)

    def masked_mean(x):
        if x.shape[1] == 1:
            return x.mean()
        return (x * completion_mask).sum() / token_count

    if loss_type == 'cispo':
        is_clipped = (coef_1 > epsilon_high) & (advantages.unsqueeze(1) > 0)
        return {'cispo_clip_ratio': masked_mean(is_clipped.float())}

    if loss_type in ('sapo', 'real'):
        return {}

    is_low = (coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)
    is_high = (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
    is_region = is_low | is_high

    return {
        'low_clip': masked_mean(is_low.float()),
        'high_clip': masked_mean(is_high.float()),
        'region_clip': masked_mean(is_region.float()),
    }
