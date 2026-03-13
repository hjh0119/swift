# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .driver import DPODriver, MegatronDriver
    from .megatron_worker import MegatronWorker
    from .resource_pool import ResourcePool, ResourcePoolManager
    from .worker_group import WorkerGroup


def __getattr__(name):
    _lazy_imports = {
        'MegatronDriver': '.driver',
        'DPODriver': '.driver',
        'MegatronWorker': '.megatron_worker',
        'ResourcePool': '.resource_pool',
        'ResourcePoolManager': '.resource_pool',
        'WorkerGroup': '.worker_group',
    }
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name], __name__)
        return getattr(module, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
