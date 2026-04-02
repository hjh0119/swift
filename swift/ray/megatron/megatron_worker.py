# Copyright (c) ModelScope Contributors. All rights reserved.
"""MegatronWorker — single-GPU Ray actor wrapping a Megatron model.

Training:   init_model(trainable=True) → setup() → train_step(...)* → finalize()
Inference:  init_model(trainable=False) → forward(batch)*
"""
import os
import torch
from typing import Any, Dict, List, Optional

from swift.utils import get_logger

logger = get_logger()


class MegatronWorker:
    """Thin RPC shell.  Not ``@ray.remote`` — decorated by WorkerGroup."""

    def init_model(
        self,
        argv: List[str],
        trainable: bool = True,
        pipeline_cls_path: Optional[str] = None,
        trainer_cls_path: Optional[str] = None,
    ):
        self.trainable = trainable

        if pipeline_cls_path:
            import importlib
            mod, cls = pipeline_cls_path.rsplit('.', 1)
            pipeline_cls = getattr(importlib.import_module(mod), cls)
        else:
            from swift.megatron.pipelines.train.rlhf import MegatronRLHF
            pipeline_cls = MegatronRLHF

        self._pipeline = pipeline_cls(argv)
        self._trainer_cls_path = trainer_cls_path

        if trainable:
            self._init_trainable()
        else:
            self._init_inference()

        self._register_parallel_info()

    def _init_trainable(self):
        p = self._pipeline
        args = p.args
        self._train_dataset, self._val_dataset = p._prepare_dataset()
        args.init_iters(self._train_dataset, self._val_dataset)

        if self._trainer_cls_path:
            import importlib
            mod, cls = self._trainer_cls_path.rsplit('.', 1)
            trainer_cls = getattr(importlib.import_module(mod), cls)
            self.trainer = trainer_cls(args, p.template)
        else:
            self.trainer = p.prepare_trainer()

    def _init_inference(self):
        """Create model only — no optimizer, no dataset, no DDP."""
        from megatron.core.transformer.module import Float16Module

        from swift.megatron.model import get_mcore_model

        p = self._pipeline
        args = p.args
        if args.train_iters is None:
            args.train_iters = 1

        models = get_mcore_model(args, p.template.config)
        model_id = args.ref_model or args.model
        p.bridge.load_weights(models, model_id)

        for m in models:
            m.requires_grad_(False)
            m.eval()

        if args.fp16 or args.bf16:
            models = [Float16Module(m.config, m) for m in models]

        self._infer_models = models
        self._args = args
        self._template = p.template
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Training API  (delegates to BaseMegatronTrainer primitives)
    # ------------------------------------------------------------------

    def setup(self) -> Dict[str, Any]:
        """Configure DDP, build data iterators. Returns loop metadata."""
        trainer = self.trainer
        self._train_data_iterator, self._val_data_iterator = (
            trainer.setup_training(self._train_dataset, self._val_dataset))
        return {
            'train_iters': trainer.args.train_iters,
            'iteration': trainer.state.iteration,
        }

    def train_step(self) -> Dict[str, Any]:
        """One training step. Algorithm-agnostic — reads from local iterator."""
        trainer = self.trainer
        trainer.run_train_step(self._train_data_iterator, self._val_data_iterator)

        state = trainer.state
        result = {
            'iteration': state.iteration,
        }
        return result

    def get_current_batch(self) -> Optional[Dict[str, Any]]:
        """Peek at the next micro-batches and stash them for train_step.

        Reads ``num_microbatches`` from the local data iterator, replaces
        the iterator with a chain(stashed, original) so the subsequent
        ``train_step`` re-reads them transparently.

        Returns the merged global batch (only meaningful on collectors).
        Not supported for Virtual-Pipeline parallelism.
        """
        import itertools
        args = self.trainer.args

        if isinstance(self._train_data_iterator, list):
            raise NotImplementedError('get_current_batch does not support VP parallelism')

        n_micro = args.num_microbatches
        batches = []
        for _ in range(n_micro):
            try:
                b = next(self._train_data_iterator)
                batches.append(b)
            except StopIteration:
                break

        if not batches:
            return None

        self._train_data_iterator = itertools.chain(iter(batches), self._train_data_iterator)

        merged = {}
        for k in batches[0]:
            vals = [b[k] for b in batches if k in b]
            if isinstance(vals[0], torch.Tensor):
                merged[k] = torch.cat(vals, dim=0)
            else:
                merged[k] = vals[0]
        return merged

    def finalize(self) -> Dict[str, Any]:
        from swift.utils import is_last_rank
        trainer = self.trainer
        trainer.finalize_training()
        self._pipeline._handle_trainer_state(trainer, is_last_rank())
        s = trainer.state
        return {
            'last_model_checkpoint': s.last_model_checkpoint,
            'best_model_checkpoint': s.best_model_checkpoint,
            'best_metric': s.best_metric,
        }

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def forward(
        self,
        raw_batch: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """PP-aware forward for inference workers.

        Returns ``{ref_logps: tensor (cpu)}`` on last PP stage, None otherwise.
        """
        from swift.megatron.utils import forward_step_helper
        from swift.utils import to_device

        batch = to_device(raw_batch, 'cuda', non_blocking=True)
        batch = self._prepare_batch_inference(batch)

        labels = batch.get('labels')
        packed_seq_params = batch.get('packed_seq_params')
        batch.pop('loss_scale', None)

        model = self._infer_models[0]

        with torch.no_grad():
            output = forward_step_helper(self._args, model, batch)

        if output is None or labels is None:
            return None

        num_samples = labels.shape[0] // 2
        if packed_seq_params is not None:
            num_samples = getattr(packed_seq_params, 'num_samples', num_samples)

        logps = self._compute_logps_inference(output, labels, packed_seq_params, num_samples * 2)
        return {'ref_logps': logps.cpu()}

    def _prepare_batch_inference(self, batch):
        from swift.megatron.trainers.utils import get_batch_on_this_cp_rank, get_batch_on_this_pp_rank
        args = self._args
        data = get_batch_on_this_pp_rank(args, batch)
        if args.padding_free:
            from swift.megatron.trainers.utils import get_packed_seq_params
            data['packed_seq_params'] = get_packed_seq_params(data)
        data = get_batch_on_this_cp_rank(args, data)
        return data

    def _compute_logps_inference(
        self,
        output_tensor,
        labels,
        packed_seq_params,
        num_samples,
    ):
        from megatron.core import mpu
        from torch.distributed.nn import all_reduce
        args = self._args
        per_token_logps = -output_tensor
        loss_mask = labels != -100
        per_token_logps = per_token_logps * loss_mask
        if args.padding_free:
            cu_seqlens = (packed_seq_params.cu_seqlens_q[:num_samples + 1] // args.context_parallel_size)
            all_logps = per_token_logps.new_zeros((num_samples, ))
            for i in range(num_samples):
                s, e = cu_seqlens[i], cu_seqlens[i + 1]
                all_logps[i] = per_token_logps[:, s:e].sum()
        else:
            all_logps = per_token_logps.sum(-1)
        if args.context_parallel_size > 1:
            all_logps = all_reduce(all_logps, group=mpu.get_context_parallel_group())
        return all_logps

    # ------------------------------------------------------------------
    # Parallel info
    # ------------------------------------------------------------------

    def _register_parallel_info(self):
        from megatron.core import mpu
        self._dp_rank = mpu.get_data_parallel_rank()
        self._dp_size = mpu.get_data_parallel_world_size()
        self._is_collector = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0)

    def get_parallel_info(self) -> Dict[str, Any]:
        return {
            'dp_rank': self._dp_rank,
            'dp_size': self._dp_size,
            'is_collector': self._is_collector,
        }

    def call_trainer(self, method_name: str, *args, **kwargs) -> Any:
        """Proxy to call an arbitrary method on the internal trainer."""
        return getattr(self.trainer, method_name)(*args, **kwargs)

    def ping(self) -> str:
        mode = 'train' if self.trainable else 'infer'
        return '%s_rank%s' % (mode, os.environ.get('RANK', '?'))
