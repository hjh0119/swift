# Copyright (c) ModelScope Contributors. All rights reserved.
"""Algorithm-specific Ray training loops.

Class hierarchy::

    RayTrainer                  # default loop (SFT / KTO)
      └─ DPORayTrainer          # DPO with optional remote ref
"""
import ray
from typing import Any, Dict, Optional

from swift.utils import get_logger

logger = get_logger()


class RayTrainer:
    """Base class.  Subclasses override ``_train_loop``."""

    def __init__(self, worker_groups: Dict[str, Any]):
        self.worker_groups = worker_groups

    @property
    def train_group(self):
        return self.worker_groups['train']

    def fit(self) -> Any:
        tg = self.train_group
        meta = tg.broadcast('setup')[0]
        train_iters = meta['train_iters']
        iteration = meta['iteration']

        try:
            iteration = self._train_loop(tg, train_iters, iteration)
        except Exception:
            logger.exception('Failed at iteration %d', iteration)
            raise
        finally:
            results = tg.broadcast('finalize')

        return results[0] if results else None

    def _train_loop(self, tg, train_iters, iteration):
        """Default loop: plain train_step each iteration."""
        while iteration < train_iters:
            step_results = tg.broadcast('train_step')
            iteration = self._extract_iteration(step_results)
        return iteration

    def _extract_iteration(self, step_results):
        for r in step_results:
            if r and 'iteration' in r:
                return r['iteration']
        return 0


class DPORayTrainer(RayTrainer):
    """DPO training with an optional remote ref group.

    Without ref group:  workers use local ref models (standard path).
    With ref group:
      1. Peek at the next batches on train collectors
      2. Forward each batch on the matching ref-group worker
      3. Inject ref_logps into train workers' trainer
      4. Train step (trainer's loss_func reads ref_logps via _split_ref_policy)
    """

    REMOTE_REF_TRAINER_CLS = ('swift.ray.megatron.ray_megatron_trainer.RayMegatronDPOTrainer')

    @property
    def ref_group(self) -> Optional[Any]:
        return self.worker_groups.get('ref')

    @property
    def has_remote_ref(self) -> bool:
        return self.ref_group is not None

    def get_train_init_kwargs(self, group_names=None) -> dict:
        has_ref = (group_names is not None and 'ref' in group_names) or self.has_remote_ref
        if has_ref:
            return {'trainer_cls_path': self.REMOTE_REF_TRAINER_CLS}
        return {}

    def _train_loop(self, tg, train_iters, iteration):
        ref = self.ref_group

        while iteration < train_iters:
            if ref is not None:
                self._compute_and_set_ref_logps(tg, ref)
            step_results = tg.broadcast('train_step')
            iteration = self._extract_iteration(step_results)

        return iteration

    def _compute_and_set_ref_logps(self, tg, ref):
        """Fetch batches → forward on ref → set ref_logps on train trainers."""
        global_batches = tg.broadcast('get_current_batch')

        dp_batches: Dict[int, Any] = {}
        for batch, dp_r, is_coll in zip(
                global_batches,
                tg._dp_rank_map,
                tg._collect_mask,
        ):
            if is_coll and batch is not None:
                dp_batches[dp_r] = batch

        ref_logps_by_dp: Dict[int, Any] = {}
        pending, meta = [], []
        for train_dp, batch in dp_batches.items():
            target_dp = train_dp % ref.dp_size
            found = False
            for w, dp_r, is_coll in zip(
                    ref._workers,
                    ref._dp_rank_map,
                    ref._collect_mask,
            ):
                if dp_r == target_dp:
                    pending.append(w.forward.remote(batch))
                    if is_coll and not found:
                        meta.append((train_dp, len(pending) - 1))
                        found = True

        if pending:
            results = ray.get(pending)
            for train_dp, idx in meta:
                r = results[idx]
                if r is not None and 'ref_logps' in r:
                    ref_logps_by_dp[train_dp] = r['ref_logps']

        set_refs = []
        for w, dp_r in zip(tg._workers, tg._dp_rank_map):
            logps = ref_logps_by_dp.get(dp_r)
            set_refs.append(w.call_trainer.remote('set_ref_logps', logps))
        ray.get(set_refs)
