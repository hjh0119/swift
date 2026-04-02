# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import ray
import yaml
from typing import Any, Dict, List, Optional

from swift.utils import get_logger

logger = get_logger()

# ======================================================================
# Trainer registry
# ======================================================================

_TRAINER_REGISTRY: Dict[str, Dict[str, Any]] = {
    'dpo': {
        'trainer': 'swift.ray.megatron.ray_trainer.DPORayTrainer',
        'groups': {
            'train': True,
            'ref': False
        },
    },
    'kto': {
        'trainer': 'swift.ray.megatron.ray_trainer.RayTrainer',
        'groups': {
            'train': True
        },
    },
}


def register_ray_trainer(rlhf_type: str, trainer_cls_path: str, groups: Dict[str, bool]):
    """Register a new algorithm.

    Args:
        rlhf_type: Algorithm key used in YAML ``rlhf_type``.
        trainer_cls_path: Dotted import path to the Trainer class.
        groups: ``{group_name: trainable}`` mapping.  The key is the
            YAML section name; the value indicates whether the group
            is trainable (``True``) or inference-only (``False``).
    """
    _TRAINER_REGISTRY[rlhf_type] = {
        'trainer': trainer_cls_path,
        'groups': groups,
    }


# ======================================================================
# YAML parsing
# ======================================================================


def _build_group_argv(shared: Dict[str, Any], group: Dict[str, Any]) -> List[str]:
    """Merge shared + group-specific config into a CLI argv list."""
    merged = {**shared, **group}
    skip = {'gpus', 'colocate_groups', 'rlhf_type'}
    argv: List[str] = []
    for key, val in merged.items():
        if key in skip or val is None:
            continue
        argv.append('--%s' % key)
        if isinstance(val, bool):
            argv.append('true' if val else 'false')
        elif isinstance(val, (list, tuple)):
            argv.extend(str(v) for v in val)
        elif isinstance(val, dict):
            argv.append(json.dumps(val))
        else:
            argv.append(str(val))
    return argv


def parse_ray_config(config_path: str) -> Dict[str, Any]:
    """Parse a Ray YAML config file.

    Returns a dict with ``rlhf_type``, ``group_argv``, ``group_gpus``,
    ``colocate_groups``.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    rlhf_type = raw.pop('rlhf_type', 'dpo')
    entry = _TRAINER_REGISTRY.get(rlhf_type)
    if entry is None:
        raise ValueError('Unknown rlhf_type %r. Available: %s' % (rlhf_type, list(_TRAINER_REGISTRY)))

    colocate_groups = raw.pop('colocate_groups', [])
    group_names = list(entry['groups'].keys())

    group_sections: Dict[str, dict] = {}
    for g in group_names:
        group_sections[g] = raw.pop(g, {})
    shared = dict(raw)

    group_argv: Dict[str, List[str]] = {}
    group_gpus: Dict[str, int] = {}
    for g, cfg in group_sections.items():
        group_gpus[g] = cfg.pop('gpus', 0)
        group_argv[g] = _build_group_argv(shared, cfg)

    return {
        'rlhf_type': rlhf_type,
        'group_argv': group_argv,
        'group_gpus': group_gpus,
        'colocate_groups': colocate_groups,
    }


# ======================================================================
# Pipeline
# ======================================================================


class MegatronRayPipeline:
    """Algorithm-agnostic Ray Megatron training pipeline.

    Handles all infrastructure (Ray, placement groups, worker groups)
    then delegates the training loop to an algorithm-specific Trainer.
    """

    def __init__(self, config_path: str):
        parsed = parse_ray_config(config_path)
        self.rlhf_type = parsed['rlhf_type']
        self.group_argv = parsed['group_argv']
        self.group_gpus = parsed['group_gpus']
        self.colocate_groups = parsed['colocate_groups']

        self._entry = _TRAINER_REGISTRY[self.rlhf_type]
        self.resource_pool_manager = None
        self.worker_groups: Dict[str, Any] = {}

    def run(self) -> Any:
        """Full lifecycle: init → train → shutdown."""
        ray.init(ignore_reinit_error=True)
        try:
            self._create_pools()
            ray_trainer = self._create_trainer()
            self._init_worker_groups(ray_trainer)
            ray_trainer.worker_groups = self.worker_groups
            return ray_trainer.fit()
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    def _create_pools(self):
        from .resource_pool import ResourcePool, ResourcePoolManager

        colocated_sets = {frozenset(g) for g in self.colocate_groups}
        pool_mapping: Dict[str, ResourcePool] = {}
        assigned: set = set()

        for colocated in colocated_sets:
            max_gpus = max(self.group_gpus.get(g, 0) for g in colocated)
            if max_gpus <= 0:
                continue
            shared = ResourcePool([max_gpus], max_colocate_count=len(colocated))
            for g in colocated:
                pool_mapping[g] = shared
                assigned.add(g)

        for name, gpus in self.group_gpus.items():
            if name in assigned or gpus <= 0:
                continue
            pool_mapping[name] = ResourcePool([gpus])

        self.resource_pool_manager = ResourcePoolManager(pool_mapping)
        self.resource_pool_manager.create_all()

    def _init_worker_groups(self, ray_trainer):
        from .worker_group import WorkerGroup

        group_trainable = self._entry['groups']
        pool_port_used: dict = {}
        group_names = set(g for g, gpus in self.group_gpus.items() if gpus > 0)
        extra_train_kwargs = {}
        if hasattr(ray_trainer, 'get_train_init_kwargs'):
            extra_train_kwargs = ray_trainer.get_train_init_kwargs(group_names)

        for group_name, argv in self.group_argv.items():
            gpus = self.group_gpus.get(group_name, 0)
            if gpus <= 0:
                continue

            pool = self.resource_pool_manager.get_pool(group_name)
            colocate_count = pool.max_colocate_count
            num_gpus = 1.0 / colocate_count if colocate_count > 1 else 1.0

            master_port = None
            pool_id = id(pool)
            if colocate_count > 1:
                if pool_id in pool_port_used:
                    master_port = pool.discover_free_port()
                else:
                    pool_port_used[pool_id] = True

            wg = WorkerGroup.from_pool(group_name, pool, num_gpus=num_gpus, master_port=master_port)

            trainable = group_trainable.get(group_name, False)
            init_kwargs = dict(trainable=trainable)
            if trainable:
                init_kwargs.update(extra_train_kwargs)
            wg.broadcast('init_model', argv, **init_kwargs)
            wg.build_dispatch_info()

            self.worker_groups[group_name] = wg

    def _create_trainer(self):
        cls_path = self._entry['trainer']
        mod_path, cls_name = cls_path.rsplit('.', 1)
        import importlib
        mod = importlib.import_module(mod_path)
        trainer_cls = getattr(mod, cls_name)
        return trainer_cls({})

    def _shutdown(self):
        if self.resource_pool_manager is not None:
            self.resource_pool_manager.destroy_all()
        ray.shutdown()


def main():
    import sys
    config_path = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--config' and i + 1 < len(sys.argv[1:]):
            config_path = sys.argv[i + 2]
            break
    if config_path is None:
        raise ValueError('Usage: python -m swift.ray.megatron.pipeline --config <yaml>')

    return MegatronRayPipeline(config_path).run()


if __name__ == '__main__':
    main()
