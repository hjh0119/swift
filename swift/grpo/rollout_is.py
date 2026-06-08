# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rollout importance sampling correction functions.

Pure tensor functions for IS correction, off-policy diagnostics, and
sequence masking. No distributed communication or Trainer dependencies.
"""
from typing import Dict, Literal, Optional

import torch


def compute_sequence_level_ratios(
    is_ratio: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sequence-level IS ratios as geometric mean of token-level ratios.

    Args:
        is_ratio: Token-level IS ratios, shape ``[B, T]``.
        completion_mask: Boolean mask for completion tokens, shape ``[B, T]``.

    Returns:
        Sequence-level ratios, shape ``[B]``.
    """
    log_ratio = torch.log(is_ratio.clamp(min=1e-10))
    seq_log_ratios = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
    return torch.exp(seq_log_ratios)


def apply_rollout_importance_sampling(
    rollout_log_ratio: torch.Tensor,
    completion_mask: torch.Tensor,
    mode: Literal['token_truncate', 'token_mask', 'sequence_truncate', 'sequence_mask'],
    threshold: float,
) -> torch.Tensor:
    """Apply rollout importance sampling correction.

    Args:
        rollout_log_ratio: ``log(pi_theta / pi_rollout)`` per token, shape ``[B, T]``.
        completion_mask: Boolean mask for completion tokens, shape ``[B, T]``.
        mode: IS correction mode.
        threshold: Truncation/masking threshold.

    Returns:
        IS weights, shape ``[B, T]``.
    """
    SAFETY_BOUND = 20.0
    rollout_log_ratio_safe = torch.clamp(rollout_log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    is_ratio = torch.exp(rollout_log_ratio_safe)

    if mode == 'token_truncate':
        return torch.clamp(is_ratio, max=threshold)

    if mode == 'token_mask':
        return torch.where(is_ratio <= threshold, is_ratio, torch.zeros_like(is_ratio))

    if mode == 'sequence_truncate':
        seq_ratios = compute_sequence_level_ratios(is_ratio, completion_mask)
        clipped = torch.clamp(seq_ratios, max=threshold)
        return clipped.unsqueeze(-1).expand_as(is_ratio)

    if mode == 'sequence_mask':
        seq_ratios = compute_sequence_level_ratios(is_ratio, completion_mask)
        seq_mask = (seq_ratios <= threshold).float()
        return is_ratio * seq_mask.unsqueeze(-1)

    return is_ratio


def compute_off_policy_sequence_mask(
    per_token_logps: torch.Tensor,
    old_policy_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Compute off-policy sequence mask (DeepSeek-V3.2 method).

    Masks out sequences where:
    1. ``mean(old_logps - current_logps) > delta``
    2. AND ``advantage < 0``

    Args:
        per_token_logps: Current policy log probs, shape ``[B, T]``.
        old_policy_logps: Old/rollout policy log probs, shape ``[B, T]``.
        completion_mask: Boolean mask for completion tokens, shape ``[B, T]``.
        advantages: Advantage values per sample, shape ``[B]``.
        delta: Threshold for sequence masking.

    Returns:
        Mask, shape ``[B]``. ``True`` = keep, ``False`` = mask out.
    """
    log_ratio = old_policy_logps - per_token_logps
    seq_mean = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
    exceeds = seq_mean > delta
    negative_adv = advantages < 0
    return ~(exceeds & negative_adv)


def compute_offpolicy_metrics(
    per_token_logps: torch.Tensor,
    rollout_logps: torch.Tensor,
    completion_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute off-policy diagnostic metrics (local tensors, no gather).

    Computes KL divergence estimators, perplexity metrics, and chi-squared
    divergence between training and rollout policies.

    Args:
        per_token_logps: Log probs from training policy, shape ``[B, T]``.
        rollout_logps: Log probs from rollout policy, shape ``[B, T]``.
        completion_mask: Boolean mask for completion tokens, shape ``[B, T]``.

    Returns:
        Dict of scalar tensors. Trainer is responsible for gathering.
    """
    SAFETY_BOUND = 20.0

    def masked_mean(x, mask, axis=None):
        if axis is None:
            return (x * mask).sum() / mask.sum().clamp(min=1.0)
        return (x * mask).sum(axis) / mask.sum(axis).clamp(min=1.0)

    metrics: Dict[str, torch.Tensor] = {}

    mean_log_prob_training = masked_mean(per_token_logps, completion_mask, axis=-1)
    training_ppl = torch.exp(-mean_log_prob_training).mean()
    metrics['training_ppl'] = training_ppl
    metrics['training_log_ppl'] = (-mean_log_prob_training).mean()

    log_ratio = per_token_logps - rollout_logps
    log_ratio = log_ratio * completion_mask

    kl = masked_mean(-log_ratio, completion_mask)
    metrics['kl'] = kl

    log_ratio_safe = torch.clamp(log_ratio, min=-20, max=20)
    k3_kl_matrix = torch.clamp(torch.exp(log_ratio_safe) - log_ratio_safe - 1, min=-10, max=10)
    metrics['k3_kl'] = masked_mean(k3_kl_matrix, completion_mask)

    mean_log_prob_rollout = masked_mean(rollout_logps, completion_mask, axis=-1)
    rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()
    metrics['rollout_ppl'] = rollout_ppl
    metrics['rollout_log_ppl'] = (-mean_log_prob_rollout).mean()

    log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
    metrics['log_ppl_diff'] = log_ppl_diff.mean()
    metrics['log_ppl_abs_diff'] = log_ppl_diff.abs().mean()
    metrics['log_ppl_diff_max'] = log_ppl_diff.max()
    metrics['log_ppl_diff_min'] = log_ppl_diff.min()

    ppl_ratio = torch.exp(log_ppl_diff).mean()
    metrics['ppl_ratio'] = ppl_ratio

    log_ratio_safe2 = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    rho_token = torch.exp(log_ratio_safe2)
    chi2_token = masked_mean(rho_token.square(), completion_mask) - 1.0
    metrics['chi2_token'] = chi2_token

    log_ratio_mean = masked_mean(log_ratio, completion_mask, axis=-1)
    log_ratio_mean_safe = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    rho_geo = torch.exp(log_ratio_mean_safe)
    chi2_seq = rho_geo.square().mean() - 1.0
    metrics['chi2_seq'] = chi2_seq

    return metrics


def compute_is_metrics(
    rollout_log_ratio: torch.Tensor,
    is_weights: torch.Tensor,
    completion_mask: torch.Tensor,
    mode: str,
    threshold: float,
) -> Dict[str, torch.Tensor]:
    """Compute IS correction metrics (local tensors, no gather).

    Args:
        rollout_log_ratio: ``log(pi_policy / pi_rollout)``, shape ``[B, T]``.
        is_weights: IS weights after correction, shape ``[B, T]``.
        completion_mask: Boolean mask for completion tokens, shape ``[B, T]``.
        mode: IS correction mode.
        threshold: Truncation/masking threshold.

    Returns:
        Dict with ``is_weight_mean``, ``ess``, ``clipped_frac`` scalar tensors.
    """
    SAFETY_BOUND = 20.0
    threshold_lower = 1.0 / threshold

    def masked_mean(x, mask):
        return (x * mask).sum() / mask.sum().clamp(min=1.0)

    log_ratio_safe = torch.clamp(rollout_log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    is_ratio = torch.exp(log_ratio_safe)

    metrics: Dict[str, torch.Tensor] = {}
    metrics['is_weight_mean'] = masked_mean(is_weights, completion_mask)

    weights_for_ess = is_weights.clamp(min=threshold_lower, max=threshold)
    mean_for_ess = masked_mean(weights_for_ess, completion_mask)
    is_weights_normalized = weights_for_ess / (mean_for_ess + 1e-8)
    ess = 1.0 / masked_mean(is_weights_normalized.square(), completion_mask).clamp(min=1e-10)
    metrics['ess'] = ess

    if mode in ['token_truncate', 'token_mask']:
        if mode == 'token_truncate':
            clipped_frac = masked_mean((is_ratio > threshold).float(), completion_mask)
        else:
            clipped_frac = masked_mean((is_weights == 0).float(), completion_mask)
        metrics['clipped_frac'] = clipped_frac
    else:
        seq_ratios = compute_sequence_level_ratios(is_ratio, completion_mask)
        metrics['clipped_frac'] = (seq_ratios > threshold).float().mean()

    return metrics
