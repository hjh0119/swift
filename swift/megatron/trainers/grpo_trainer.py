# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.inference.communication_utils import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank
from megatron.training import get_args, get_model, training
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import unwrap_model
from torch.distributed.nn import all_reduce

from swift.utils import get_current_device, get_logger, is_vllm_available
from ..argument import MegatronRLHFArguments
from .rlhf_base import MegatronRLHFTrainer
from .trainer import MegatronTrainer
from .utils import get_batch

logger = get_logger()


class MegatronGRPOTrainer(MegatronRLHFTrainer):

    def __init__(self, args: MegatronRLHFArguments):
        super().__init__(args)
        # TODO: init vllm
        self.use_vllm = args.use_vllm
        vllm_client = args.vllm_client
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                                  'Please install vLLM with `pip install vllm -U` to use it.')
            if self.vllm_mode == 'server':
                self.vllm_client: VLLMClient = vllm_client
                if self.accelerator.is_main_process:
                    self.vllm_client.get_engine_type()
                    vllm_use_async_engine = [self.vllm_client.use_async_engine]
                    use_gym_env = [self.vllm_client.use_gym_env]
                    enable_multi_turn = [self.vllm_client.enable_multi_turn]
                else:
                    vllm_use_async_engine = [False]
                    use_gym_env = [False]
                    enable_multi_turn = [self.enable_server_multi_turn]
                self.vllm_use_async_engine = broadcast_object_list(vllm_use_async_engine, from_process=0)[0]
                self.use_gym_env = broadcast_object_list(use_gym_env, from_process=0)[0]
                self.enable_server_multi_turn = broadcast_object_list(enable_multi_turn, from_process=0)[0]
                if self.use_gym_env:
                    self.reward_func_names = ['gym_reward']

            elif self.vllm_mode == 'colocate':
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size '
                        f'({self.accelerator.num_processes}) evenly.')

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration([
                        list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                        for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                    ])
                self.enable_offload = self.args.offload_model or self.args.offload_optimizer
                context = self.offload_context if self.enable_offload else nullcontext

                with context():
                    self.engine = self.prepare_vllm(model)
                    if self.args.sleep_level > 0:
                        self.engine.engine.sleep(self.args.sleep_level)

        else:
            from swift.llm import PtEngine
            self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit

    def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
        # prepare global batch data here
        new_data_iterator = self._replace_data_iterator(data_iterator)
        return self._origin_train_step(forward_step_func, new_data_iterator, model, optimizer, opt_param_scheduler,
                                       config)

    def loss_func(self, output_tensor: torch.Tensor, *, ref_logps: torch.Tensor, labels: torch.Tensor,
                  packed_seq_params):
        args = get_args()
        num_samples = packed_seq_params.num_samples

        logps = self.get_logps(output_tensor, labels, packed_seq_params)
        loss, chosen_rewards, rejected_rewards = self.dummy_dpo_trainer.dpo_loss(
            logps[:num_samples],
            logps[num_samples:],
            ref_logps[:num_samples],
            ref_logps[num_samples:],
        )
        if args.rpo_alpha:
            loss_mask = labels != -100
            num_tokens = packed_seq_params.cu_seqlens_q[num_samples] // args.context_parallel_size
            loss_mask[:, num_tokens:] = 0
            nll_loss = torch.concat([torch.sum(output_tensor * loss_mask)[None], loss_mask.sum()[None]])
            if args.context_parallel_size > 1:
                nll_loss = all_reduce(nll_loss, group=mpu.get_context_parallel_group())
            nll_loss = nll_loss[0] / nll_loss[1]
            loss = loss + args.rpo_alpha * nll_loss
        loss = loss.mean()
        metric = {
            'loss': loss.clone().detach(),
            'logps/chosen': logps[:num_samples].mean(),
            'logps/rejected': logps[num_samples:].mean(),
            'rewards/chosen': chosen_rewards.mean(),
            'rewards/rejected': rejected_rewards.mean(),
            'rewards/accuracies': (chosen_rewards > rejected_rewards).float().mean(),
            'rewards/margins': (chosen_rewards - rejected_rewards).mean(),
        }
        if args.rpo_alpha:
            metric['nll_loss'] = nll_loss.detach()
        reporting_metric = loss.new_tensor(list(metric.values()))
        torch.distributed.all_reduce(
            reporting_metric, torch.distributed.ReduceOp.AVG, group=mpu.get_data_parallel_group())
        reporting_metric = {k: reporting_metric[i] for i, k in enumerate(metric.keys())}
        # fix megatron-lm bug
        # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
        loss = loss / mpu.get_context_parallel_world_size()
        return loss, reporting_metric

    def _replace_data_iterator(self, data_iterator):
        args = get_args()
        num_iters_per_step = args.global_batch_size // (args.micro_batch_size * mpu.get_data_parallel_world_size())
        res = []
        with torch.no_grad(), self.null_ref_context() as ref_model:
            for i in range(num_iters_per_step):
                res.append(self.ref_forward(ref_model, data_iterator))
        return iter(res)

    def forward_step(self, data_iterator, model):
        with torch.no_grad():
            data = next(data_iterator)

        ref_logps = data.pop('logps')
        with self.stimer:
            output_tensor = model(**data)
        return output_tensor, partial(
            self.loss_func,
            ref_logps=ref_logps,
            labels=data.get('labels'),
            packed_seq_params=data.get('packed_seq_params'))
