# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron Trainer subclasses for Ray remote-model scenarios.

These run inside Ray workers as "inner trainers" that only handle
single-model forward/backward + algorithm-specific loss computation.
Cross-model data flow (ref logps, teacher logits, rollout rewards)
is managed by the outer RayTrainer (driver layer).
"""
import torch
from collections import namedtuple
from functools import partial
from megatron.core import mpu
from torch.distributed.nn import all_reduce

from swift.megatron.trainers.base import BaseMegatronTrainer
from swift.megatron.trainers.rlhf_mixin import MegatronRLHFTrainer
from swift.utils import get_current_device


class RayMegatronDPOTrainer(MegatronRLHFTrainer):
    """Policy-only DPO trainer for Ray remote-ref mode.

    Unlike ``MegatronDPOTrainer`` which owns ref models locally,
    this trainer receives pre-computed ``ref_logps`` from the driver.
    It only runs policy model forward and computes DPO loss.
    """

    def __init__(self, args, template):
        super().__init__(args, template)
        self._ref_logps = None

        from trl.trainer import FDivergenceConstants

        from swift.rlhf_trainers import DPOTrainer
        self._dummy = type('_D', (DPOTrainer, ), {'__init__': lambda s: None})()
        self._dummy.accelerator = namedtuple('A', ['device'])(device=get_current_device())
        self._dummy.f_alpha_divergence_coef = 1.
        self._dummy.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: 1.}
        self._dummy.reference_free = args.reference_free
        self._dummy.label_smoothing = args.label_smoothing
        self._dummy.f_divergence_type = args.f_divergence_type
        self._dummy.loss_type = args.loss_type
        self._dummy.beta = args.beta

    def prepare_model(self):
        """Policy model only — skip ref model creation."""
        BaseMegatronTrainer.prepare_model(self)
        self.ref_models = []

    def _load_checkpoint(self):
        BaseMegatronTrainer._load_checkpoint(self)

    def set_ref_logps(self, ref_logps):
        """Called by driver via ``worker.call_trainer('set_ref_logps', ...)``."""
        if ref_logps is not None:
            self._ref_logps = ref_logps.to('cuda')

    def forward_step(self, data_iterator, model):
        unwrapped_model = model.module.module
        # input_tensor = unwrapped_model.get_input_tensor()
        vp_stage = unwrapped_model.vp_stage
        data = self.get_batch(data_iterator, vp_stage)
        data.pop('loss_scale', None)

        output_tensor = model(**data)
        return output_tensor, partial(
            self.loss_func, labels=data.get('labels'), packed_seq_params=data.get('packed_seq_params'))

    def loss_func(self, output_tensor, *, labels, packed_seq_params):
        args = self.args
        num_samples = labels.shape[0] // 2 if packed_seq_params is None else packed_seq_params.num_samples

        logps = self.get_logps(output_tensor, labels, packed_seq_params, num_samples * 2)

        ref_logps = self._ref_logps
        if ref_logps is None:
            raise RuntimeError('ref_logps not set before train_step. '
                               'Ensure DPORayTrainer calls set_ref_logps.')

        loss, chosen_rewards, rejected_rewards = self._dummy.dpo_loss(
            logps[:num_samples],
            logps[num_samples:],
            ref_logps[:num_samples],
            ref_logps[num_samples:],
        )
        if args.rpo_alpha:
            loss_mask = labels != -100
            if args.padding_free:
                num_tokens = packed_seq_params.cu_seqlens_q[num_samples] // args.context_parallel_size
                loss_mask[:, num_tokens:] = 0
            else:
                loss_mask[num_samples:] = 0
            nll_loss = torch.concat([torch.sum(output_tensor * loss_mask)[None], loss_mask.sum()[None]])
            if args.context_parallel_size > 1:
                nll_loss = all_reduce(nll_loss, group=mpu.get_context_parallel_group())
            nll_loss = nll_loss[0] / nll_loss[1]
            loss = loss + args.rpo_alpha * nll_loss
        loss = loss.mean()
        metric = {
            'loss': loss.detach().clone(),
            'logps/chosen': logps[:num_samples].mean(),
            'logps/rejected': logps[num_samples:].mean(),
            'rewards/chosen': chosen_rewards.mean(),
            'rewards/rejected': rejected_rewards.mean(),
            'rewards/accuracies': (chosen_rewards > rejected_rewards).float().mean(),
            'rewards/margins': (chosen_rewards - rejected_rewards).mean(),
        }
        if args.rpo_alpha:
            metric['nll_loss'] = nll_loss.detach()
        metric = self._all_reduce_metric(metric)
        loss = loss / mpu.get_context_parallel_world_size()
        return loss, metric
