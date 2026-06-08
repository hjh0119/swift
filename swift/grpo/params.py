# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared GRPO algorithm parameters.

``GRPOAlgorithmParams`` is a dataclass mixin that holds all pure-algorithm
parameters shared across HF, Megatron, and Ray backends. Backend-specific
parameters (vLLM config, DeepSpeed config, etc.) remain in their respective
argument classes.

Usage::

    # HF side (swift/rlhf_trainers/args_mixin.py)
    @dataclass
    class GRPOArgumentsMixin(GRPOAlgorithmParams, RolloutTrainerArgumentsMixin):
        # HF-specific fields ...

    # Megatron side (swift/megatron/arguments/megatron_args.py)
    @dataclass
    class RLHFMegatronArgumentsMixin(GRPOAlgorithmParams, ...):
        # Megatron-specific fields ...
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class GRPOAlgorithmParams:
    """Pure GRPO algorithm parameters — no backend-specific config."""

    # clipping
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None
    delta: Optional[float] = None

    # advantage estimation
    advantage_estimator: Literal['grpo', 'rloo', 'reinforce_plus_plus'] = 'grpo'
    scale_rewards: Optional[Literal['group', 'batch', 'none', 'gdpo']] = None
    kl_in_reward: Optional[bool] = None

    # importance sampling level (GSPO)
    importance_sampling_level: Literal['token', 'sequence', 'sequence_token'] = 'token'

    # SAPO
    tau_pos: float = 1.0
    tau_neg: float = 1.05

    # REAL
    real_tau: float = 0.5

    # FIPO
    fipo_decay_rate: float = 32.0
    fipo_clip_range: Optional[float] = 0.2
    fipo_clip_high_only: bool = True
    fipo_safety_threshold: Optional[float] = 4.0

    # DAPO
    dynamic_sample: bool = False
    max_resample_times: int = 3
    overlong_filter: bool = False

    # entropy
    log_entropy: bool = False
    top_entropy_quantile: float = 1.0

    # rollout importance sampling correction
    rollout_importance_sampling_mode: Optional[Literal['token_truncate', 'token_mask', 'sequence_truncate',
                                                       'sequence_mask']] = None
    rollout_importance_sampling_threshold: float = 2.0
    log_rollout_offpolicy_metrics: bool = False
    off_policy_sequence_mask_delta: Optional[float] = None

    # reward function parameters
    reward_funcs: List[str] = field(default_factory=list)
    reward_weights: Optional[List[float]] = None
    cosine_min_len_value_wrong: float = -0.5
    cosine_max_len_value_wrong: float = 0.0
    cosine_min_len_value_correct: float = 1.0
    cosine_max_len_value_correct: float = 0.5
    cosine_max_len: Optional[int] = None
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0
    soft_max_length: Optional[int] = None
    soft_cache_length: Optional[int] = None
