# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .megatron_worker import MegatronWorker
    from .pipeline import MegatronRayPipeline, parse_ray_config, register_ray_trainer
    from .ray_megatron_trainer import RayMegatronDPOTrainer
    from .ray_trainer import DPORayTrainer, RayTrainer
    from .resource_pool import ResourcePool, ResourcePoolManager
    from .worker_group import WorkerGroup


def __getattr__(name):
    _imports = {
        'MegatronWorker': '.megatron_worker',
        'MegatronRayPipeline': '.pipeline',
        'parse_ray_config': '.pipeline',
        'register_ray_trainer': '.pipeline',
        'RayMegatronDPOTrainer': '.ray_megatron_trainer',
        'RayTrainer': '.ray_trainer',
        'DPORayTrainer': '.ray_trainer',
        'ResourcePool': '.resource_pool',
        'ResourcePoolManager': '.resource_pool',
        'WorkerGroup': '.worker_group',
    }
    if name in _imports:
        import importlib
        return getattr(importlib.import_module(_imports[name], __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
