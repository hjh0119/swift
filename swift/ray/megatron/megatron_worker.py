# Copyright (c) ModelScope Contributors. All rights reserved.
"""MegatronWorker: a single-GPU Ray actor in a Megatron distributed group.

Two operation modes:

1. **Training** (``trainable=True``): initializes the full pipeline
   and calls ``trainer.train()`` directly.  The training loop stays
   inside ``BaseMegatronTrainer.train()`` -- no decomposition.
   Cross-group coordination hooks are injected via the trainer's
   ``on_train_step_start`` method.

2. **Inference** (``trainable=False``): loads a frozen model via the
   same Pipeline path (ensuring consistent arg processing), but
   skips optimizer and data preparation.
"""
import os
import torch
from typing import Any, Dict, List, Optional

from swift.utils import get_logger

logger = get_logger()


class MegatronWorker:
    """Single-GPU actor in a distributed Megatron model group.

    Not decorated with ``@ray.remote`` -- the WorkerGroup factory
    applies ``ray.remote(num_gpus=...)`` at creation time.
    """

    def init_model(
        self,
        argv: List[str],
        trainable: bool = True,
        pipeline_cls_path: Optional[str] = None,
    ):
        """Initialize via the standard Pipeline path for both modes.

        Both trainable and inference modes go through the Pipeline
        class to ensure consistent argument processing, template
        initialization, and model setup.
        """
        self.trainable = trainable
        self._argv = argv

        if pipeline_cls_path:
            module_path, cls_name = pipeline_cls_path.rsplit('.', 1)
            import importlib
            module = importlib.import_module(module_path)
            pipeline_cls = getattr(module, cls_name)
        else:
            from swift.megatron.pipelines.train.rlhf import MegatronRLHF
            pipeline_cls = MegatronRLHF

        self._pipeline = pipeline_cls(argv)

        if trainable:
            self._setup_trainable()
        else:
            self._setup_inference()

        self._register_parallel_info()

    def _setup_trainable(self):
        """Prepare trainer + dataset, ready for ``run_training``."""
        args = self._pipeline.args
        train_dataset, val_dataset = self._pipeline._prepare_dataset()
        args.init_iters(train_dataset, val_dataset)

        self.trainer = self._pipeline.prepare_trainer()
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

    def _setup_inference(self):
        """Freeze the model loaded by Pipeline for inference-only use.

        Reuses ``MegatronRLHFTrainer.prepare_model`` results
        (already created in Pipeline.__init__) but skips optimizer.
        """
        from megatron.core import tensor_parallel

        # args = self._pipeline.args
        self.trainer = self._pipeline.prepare_trainer()

        for m in self.trainer.unwrapped_models:
            m.requires_grad_(False)
            m.eval()
            for param in m.parameters():
                tensor_parallel.\
                    set_defaults_if_not_set_tensor_model_parallel_attributes(
                        param)

    # ------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------

    def run_training(self, **runtime_kwargs) -> dict:
        """Run the full training loop via ``trainer.train()``.

        The training loop remains entirely inside
        ``BaseMegatronTrainer.train()`` -- no decomposition.
        This avoids the coupling/fragility issues of splitting the
        loop between driver and worker.

        Cross-group hooks (e.g. remote ref logps) can be injected
        by setting attributes on the trainer before calling this.
        The trainer's ``on_train_step_start`` hook provides the
        insertion point for per-step cross-group coordination.
        """
        if not self.trainable:
            raise RuntimeError('run_training() requires trainable=True')

        for key, value in runtime_kwargs.items():
            setattr(self.trainer, key, value)

        return self._pipeline.run()

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def compute_logps(
        self,
        raw_batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Compute log-probabilities using the trainer's ``get_logps``.

        Delegates to the Megatron PP schedule for correct pipeline
        parallelism, then uses ``MegatronRLHFTrainer.get_logps`` to
        avoid duplicating logps computation logic.
        """
        from megatron.core import mpu
        from megatron.core.pipeline_parallel import get_forward_backward_func

        trainer = self.trainer
        args = trainer.args
        forward_backward_func = get_forward_backward_func()

        def _forward_step(data_iterator, model):
            batch = trainer._prepare_batch(next(data_iterator))
            labels = batch.get('labels')
            packed_seq_params = batch.get('packed_seq_params')

            unwrapped = model.module if hasattr(model, 'module') else model
            input_tensor = unwrapped.get_input_tensor()
            if input_tensor is not None:
                unwrapped.set_input_tensor(input_tensor)

            output_tensor = model(**batch)

            num_samples = labels.shape[0] if labels is not None else 0
            if packed_seq_params is not None:
                num_samples = getattr(packed_seq_params, 'num_samples', num_samples)

            def _loss_func(output_tensor):
                logps = trainer.get_logps(output_tensor, labels, packed_seq_params, num_samples)
                return logps.sum(), {'logps': logps.cpu()}

            return output_tensor, _loss_func

        num_microbatches = max(1, raw_batch.get('_num_microbatches', 1))

        with torch.no_grad():
            losses_reduced = forward_backward_func(
                forward_step_func=_forward_step,
                data_iterator=iter([raw_batch]),
                model=trainer.wrapped_models if hasattr(trainer, 'wrapped_models') else trainer.unwrapped_models,
                num_microbatches=num_microbatches,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                forward_only=True,
            )

        if mpu.is_pipeline_last_stage():
            return losses_reduced[0]['logps']
        return None

    # ------------------------------------------------------------------
    # Parallel info registration (for DP-aware dispatch)
    # ------------------------------------------------------------------

    def _register_parallel_info(self):
        from megatron.core import mpu

        self._dp_rank = mpu.get_data_parallel_rank()
        self._dp_size = mpu.get_data_parallel_world_size()
        self._is_collector = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0)
        logger.info('Worker rank=%s dp_rank=%d dp_size=%d is_collector=%s', os.environ.get('RANK', '?'), self._dp_rank,
                    self._dp_size, self._is_collector)

    def get_parallel_info(self) -> Dict[str, Any]:
        return {
            'dp_rank': self._dp_rank,
            'dp_size': self._dp_size,
            'is_collector': self._is_collector,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def ping(self) -> str:
        mode = 'train' if self.trainable else 'infer'
        rank = os.environ.get('RANK', '?')
        return '%s_worker_rank%s_alive' % (mode, rank)

    def get_node_info(self) -> Dict[str, Any]:
        import ray
        return {
            'node_ip': ray.util.get_node_ip_address(),
            'rank': int(os.environ.get('RANK', -1)),
            'world_size': int(os.environ.get('WORLD_SIZE', -1)),
            'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        }
