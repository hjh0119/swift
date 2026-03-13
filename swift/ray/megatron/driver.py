# Copyright (c) ModelScope Contributors. All rights reserved.
"""Driver: the coordinator on the Ray head node that orchestrates
distributed Megatron training across multiple WorkerGroups.

The driver does NOT own the training loop.  Workers call
``trainer.train()`` directly -- the loop stays inside
``BaseMegatronTrainer.train()`` where it belongs.

The driver's responsibilities are:
  1. Create resource pools and placement groups
  2. Spawn worker actors and initialize models
  3. Inject cross-group hooks (e.g. remote ref model handle)
     into workers before training starts
  4. Launch ``run_training()`` on all training workers
  5. Shutdown and cleanup

Cross-group coordination (e.g. DPO ref logps) happens via the
trainer's ``on_train_step_start`` hook, which subclass trainers
override to call remote inference workers.

Class hierarchy::

    MegatronDriver (base)
      - Resource pool / worker group lifecycle
      - Inject hooks, launch training

    DPODriver(MegatronDriver)
      - Manages ref group for DPO/KTO
"""
from abc import ABC
from typing import Any, Dict, List, Optional

from swift.utils import get_logger

logger = get_logger()


class MegatronDriver(ABC):
    """Base driver for Ray-based Megatron training.

    Args:
        group_argv: Mapping from group name to argv list.
        group_gpus: Mapping from group name to GPU count.
        colocate_groups: Groups that share the same GPUs.
    """

    def __init__(
        self,
        group_argv: Dict[str, List[str]],
        group_gpus: Dict[str, int],
        colocate_groups: Optional[List[List[str]]] = None,
    ):
        self.group_argv = group_argv
        self.group_gpus = group_gpus
        self.colocate_groups = colocate_groups or []

        self._manager = None
        self._worker_groups: Dict[str, Any] = {}

    @property
    def train_group(self):
        return self._worker_groups.get('train')

    def run(self) -> Any:
        """Full lifecycle: init -> train -> shutdown."""
        import ray
        ray.init(ignore_reinit_error=True)

        try:
            self._create_pools()
            self._init_workers()
            self._prepare_training()
            result = self._launch_training()
            return result
        finally:
            self._shutdown()

    def _create_pools(self):
        from .resource_pool import ResourcePool, ResourcePoolManager

        colocated_sets = {frozenset(gl) for gl in self.colocate_groups}

        pool_mapping: Dict[str, 'ResourcePool'] = {}
        assigned = set()

        for colocated in colocated_sets:
            max_gpus = max(self.group_gpus.get(g, 0) for g in colocated)
            if max_gpus <= 0:
                continue
            shared_pool = ResourcePool([max_gpus], max_colocate_count=len(colocated))
            for g in colocated:
                pool_mapping[g] = shared_pool
                assigned.add(g)

        for group_name, gpus in self.group_gpus.items():
            if group_name in assigned or gpus <= 0:
                continue
            pool_mapping[group_name] = ResourcePool([gpus])

        self._manager = ResourcePoolManager(pool_mapping)
        self._manager.create_all()
        logger.info('Resource pools created for groups: %s', list(pool_mapping.keys()))

    def _init_workers(self):
        from .worker_group import WorkerGroup

        for group_name, argv in self.group_argv.items():
            gpus = self.group_gpus.get(group_name, 0)
            if gpus <= 0:
                logger.info('Skipping group %r (gpus=0).', group_name)
                continue

            pool = self._manager.get_pool(group_name)
            colocate_count = pool.max_colocate_count
            num_gpus = 1.0 / colocate_count if colocate_count > 1 else 1.0

            wg = WorkerGroup.from_pool(group_name, pool, num_gpus=num_gpus)

            trainable = self._is_trainable(group_name)
            logger.info('Initializing group %r: %d GPUs, trainable=%s', group_name, gpus, trainable)
            wg.broadcast('init_model', argv, trainable=trainable)
            wg.build_dispatch_info()
            logger.info('Group %r ready: dp_size=%d', group_name, wg.dp_size)

            self._worker_groups[group_name] = wg

    def _is_trainable(self, group_name: str) -> bool:
        return group_name == 'train'

    def _prepare_training(self):
        """Hook for subclasses to inject cross-group coordination.

        Called after all groups are initialized but before training
        starts.  Override to set trainer attributes, inject remote
        model handles, etc.
        """
        pass

    def _launch_training(self) -> Any:
        """Launch training on the train group.

        Workers call ``trainer.train()`` directly.  The training loop
        runs entirely inside the worker process.
        """
        train_group = self.train_group
        if train_group is None:
            raise RuntimeError('No train group configured.')

        results = train_group.broadcast('run_training')
        logger.info('Training complete.')
        return results[0] if results else None

    def _shutdown(self):
        import ray
        if self._manager is not None:
            self._manager.destroy_all()
        ray.shutdown()
        logger.info('Ray shutdown complete.')


class DPODriver(MegatronDriver):
    """Driver for DPO/KTO training with a separate ref model group.

    When ref GPUs > 0, a separate inference group is created and
    injected into the train workers.  The train workers' trainer
    (a DPO/KTO trainer subclass) uses the remote ref group in its
    ``on_train_step_start`` hook to compute ref logps.

    When ref GPUs = 0, the trainer uses its built-in in-process ref
    model (no cross-group coordination needed).
    """

    @property
    def ref_group(self):
        return self._worker_groups.get('ref')

    def _prepare_training(self):
        if self.ref_group is not None:
            logger.info('DPO: injecting remote ref group (%d workers) '
                        'into train workers.', len(self.ref_group))
            self.train_group.broadcast('_set_remote_ref_group', self.ref_group)
        else:
            logger.info('DPO: using in-process ref model.')
