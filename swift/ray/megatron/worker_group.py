# Copyright (c) ModelScope Contributors. All rights reserved.
"""WorkerGroup — manages a set of MegatronWorker Ray actors
forming one Megatron distributed model group.
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from swift.utils import get_logger

if TYPE_CHECKING:
    from .resource_pool import ResourcePool

logger = get_logger()


class WorkerGroup:
    """A group of ``MegatronWorker`` Ray actors.

    After ``init_model`` on all workers, call ``build_dispatch_info()``
    to build the DP-rank map used by the RayTrainer.
    """

    def __init__(self, name: str, worker_handles: List[Any]):
        self.name = name
        self._workers = list(worker_handles)
        self._dp_rank_map: Optional[List[int]] = None
        self._collect_mask: Optional[List[bool]] = None
        self._dp_size: Optional[int] = None

    @property
    def world_size(self) -> int:
        return len(self._workers)

    @property
    def dp_size(self) -> int:
        if self._dp_size is None:
            raise RuntimeError('Call build_dispatch_info() first.')
        return self._dp_size

    def __len__(self) -> int:
        return len(self._workers)

    def build_dispatch_info(self):
        """Query workers for DP rank / collector info."""
        import ray
        infos = ray.get([w.get_parallel_info.remote() for w in self._workers])
        self._dp_rank_map = [i['dp_rank'] for i in infos]
        self._collect_mask = [i['is_collector'] for i in infos]
        self._dp_size = infos[0]['dp_size']

    def broadcast(self, method_name: str, *args, **kwargs) -> List[Any]:
        """Call same method with same args on all workers, block."""
        import ray
        refs = [getattr(w, method_name).remote(*args, **kwargs) for w in self._workers]
        return ray.get(refs)

    @classmethod
    def from_pool(
        cls,
        name: str,
        resource_pool: 'ResourcePool',
        worker_cls: Any = None,
        num_gpus: float = 1.0,
        master_port: Optional[int] = None,
    ) -> 'WorkerGroup':
        """Spawn actors on a ``ResourcePool``."""
        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        if worker_cls is None:
            from .megatron_worker import MegatronWorker
            worker_cls = ray.remote(num_gpus=num_gpus)(MegatronWorker)

        placements = resource_pool.get_placements(master_port=master_port)
        workers = []
        for p in placements:
            env_vars = {
                'RANK': str(p['rank']),
                'LOCAL_RANK': '0',
                'WORLD_SIZE': str(p['world_size']),
                'MASTER_ADDR': str(p['master_addr']),
                'MASTER_PORT': str(p['master_port']),
                'CUDA_DEVICE_MAX_CONNECTIONS': '1',
            }
            w = worker_cls.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=p['pg'], placement_group_bundle_index=p['bundle_idx']),
                runtime_env=RuntimeEnv(env_vars=env_vars),
            ).remote()
            workers.append(w)
        return cls(name, workers)

    def ping(self) -> List[str]:
        return self.broadcast('ping')
