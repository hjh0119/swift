# Copyright (c) ModelScope Contributors. All rights reserved.
"""WorkerGroup: manages a set of Ray actors forming one Megatron
distributed model group.

A WorkerGroup owns N MegatronWorker actors that together form a single
Megatron distributed world (with their own TP/PP/CP/EP/SP).  The group
exposes high-level SPMD APIs: all workers execute the same method and
the group collects results.

Dispatch modes (inspired by verl):
  - **broadcast**: same data to all workers (ONE_TO_ALL)
  - **dp_dispatch**: split data by DP rank, workers with the same DP
    rank receive the same chunk (ND_COMPUTE)
  - **per_worker**: each worker gets a different element from a list

Collect modes:
  - **all**: return all results as a list
  - **collect**: return only results from collector ranks (one per DP
    group, typically last-PP + TP0 + CP0), then concatenate
"""
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from swift.utils import get_logger

if TYPE_CHECKING:
    from .resource_pool import ResourcePool

logger = get_logger()


def _chunk_data(data, num_chunks):
    """Split a dict-of-tensors or dict-of-lists into ``num_chunks`` pieces."""
    import torch
    if isinstance(data, dict):
        if not data:
            return [{}] * num_chunks
        keys = list(data.keys())
        first = data[keys[0]]
        if isinstance(first, torch.Tensor):
            total = first.shape[0]
        elif isinstance(first, list):
            total = len(first)
        else:
            return [data] * num_chunks

        chunk_size = (total + num_chunks - 1) // num_chunks
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            chunk = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    chunk[k] = v[start:end]
                elif isinstance(v, list):
                    chunk[k] = v[start:end]
                else:
                    chunk[k] = v
            chunks.append(chunk)
        return chunks
    elif isinstance(data, list):
        chunk_size = (len(data) + num_chunks - 1) // num_chunks
        return [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    else:
        return [data] * num_chunks


def _concat_data(results):
    """Concatenate a list of dict-of-tensors or plain tensors."""
    import torch
    results = [r for r in results if r is not None]
    if not results:
        return None
    if isinstance(results[0], dict):
        merged = {}
        for k in results[0]:
            vals = [r[k] for r in results]
            if isinstance(vals[0], torch.Tensor):
                merged[k] = torch.cat(vals, dim=0)
            elif isinstance(vals[0], list):
                merged[k] = [item for sublist in vals for item in sublist]
            else:
                merged[k] = vals[0]
        return merged
    elif isinstance(results[0], torch.Tensor):
        return torch.cat(results, dim=0)
    else:
        return results


class _DispatchProxy:
    """Proxy that turns ``group.broadcast.method(args)`` into
    ``group.broadcast('method', args)``, providing IDE-friendly
    attribute access without requiring decorator annotations on workers.
    """

    def __init__(self, group: 'WorkerGroup', dispatch_fn_name: str):
        self._group = group
        self._fn_name = dispatch_fn_name

    def __getattr__(self, method_name: str):
        dispatch_fn = getattr(self._group, self._fn_name)

        def _call(*args, **kwargs):
            return dispatch_fn(method_name, *args, **kwargs)

        _call.__name__ = '%s.%s' % (self._fn_name, method_name)
        return _call


class WorkerGroup:
    """A group of MegatronWorker Ray actors forming one distributed model.

    After ``init_model`` is called on all workers, call
    ``build_dispatch_info()`` to query each worker's DP rank and build
    the mapping needed for DP-aware dispatch.

    Dispatch proxies for convenient attribute-style calling::

        group.bcast.init_model(argv, trainable=True)   # broadcast
        group.dp.compute_logps(batch)                   # dp_dispatch

    Or use string-based calls directly::

        group.broadcast('init_model', argv, trainable=True)
        group.dp_dispatch('compute_logps', batch)

    Args:
        name: Human-readable name for this group (e.g. ``'train'``).
        worker_handles: List of Ray actor handles (MegatronWorker).
    """

    def __init__(self, name: str, worker_handles: List[Any]):
        self.name = name
        self._workers = list(worker_handles)
        self._dp_rank_map: Optional[List[int]] = None
        self._collect_mask: Optional[List[bool]] = None
        self._dp_size: Optional[int] = None

        self.bcast = _DispatchProxy(self, 'broadcast')
        self.dp = _DispatchProxy(self, 'dp_dispatch')

    @property
    def world_size(self) -> int:
        return len(self._workers)

    @property
    def dp_size(self) -> int:
        if self._dp_size is None:
            raise RuntimeError('DP info not built. Call build_dispatch_info() first.')
        return self._dp_size

    def __len__(self) -> int:
        return len(self._workers)

    def build_dispatch_info(self):
        """Query all workers for their Megatron parallel info.

        Builds ``_dp_rank_map`` (which DP group each worker belongs to)
        and ``_collect_mask`` (which workers should return results to
        the driver).
        """
        import ray
        infos = ray.get([w.get_parallel_info.remote() for w in self._workers])

        self._dp_rank_map = [info['dp_rank'] for info in infos]
        self._collect_mask = [info['is_collector'] for info in infos]
        self._dp_size = infos[0]['dp_size']

        logger.info('WorkerGroup %r dispatch info: dp_size=%d, '
                    'dp_rank_map=%s, collectors=%s', self.name, self._dp_size, self._dp_rank_map,
                    [i for i, c in enumerate(self._collect_mask) if c])

    def broadcast(
        self,
        method_name: str,
        *args,
        **kwargs,
    ) -> List[Any]:
        """Broadcast same args to all workers (ONE_TO_ALL) and wait."""
        import ray
        refs = [getattr(w, method_name).remote(*args, **kwargs) for w in self._workers]
        return ray.get(refs)

    def broadcast_async(
        self,
        method_name: str,
        *args,
        **kwargs,
    ) -> List[Any]:
        """Broadcast without blocking."""
        return [getattr(w, method_name).remote(*args, **kwargs) for w in self._workers]

    def dp_dispatch(
        self,
        method_name: str,
        *args,
        collect: bool = True,
        **kwargs,
    ) -> Any:
        """DP-aware dispatch: split data by DP, collect from collectors.

        Data args (positional) are split into ``dp_size`` chunks.
        Workers with the same DP rank receive the same chunk.
        Only collector workers' results are kept and concatenated.

        Args:
            method_name: Remote method to call.
            *args: Data arguments to split by DP. Each arg must be
                a dict-of-tensors, list, or tensor.
            collect: If True, filter by collector mask and concat.
                If False, return all results.
            **kwargs: Broadcast kwargs (same to all workers).

        Returns:
            Concatenated results from collector workers (if collect=True),
            or list of all results.
        """
        import ray

        if self._dp_rank_map is None:
            raise RuntimeError('DP info not built. Call build_dispatch_info() first.')

        dp_size = self._dp_size
        chunked_args = [_chunk_data(arg, dp_size) for arg in args]

        refs = []
        for i, worker in enumerate(self._workers):
            dp_rank = self._dp_rank_map[i]
            worker_args = tuple(ca[dp_rank] for ca in chunked_args)
            method = getattr(worker, method_name)
            refs.append(method.remote(*worker_args, **kwargs))

        results = ray.get(refs)

        if collect and self._collect_mask is not None:
            collected = [r for r, m in zip(results, self._collect_mask) if m]
            return _concat_data(collected)
        return results

    def dp_dispatch_async(
        self,
        method_name: str,
        *args,
        **kwargs,
    ) -> List[Any]:
        """DP-aware dispatch without blocking or collecting.

        Returns a list of Ray ObjectRefs (one per worker).
        """
        if self._dp_rank_map is None:
            raise RuntimeError('DP info not built. Call build_dispatch_info() first.')

        dp_size = self._dp_size
        chunked_args = [_chunk_data(arg, dp_size) for arg in args]

        refs = []
        for i, worker in enumerate(self._workers):
            dp_rank = self._dp_rank_map[i]
            worker_args = tuple(ca[dp_rank] for ca in chunked_args)
            method = getattr(worker, method_name)
            refs.append(method.remote(*worker_args, **kwargs))
        return refs

    def execute(
        self,
        method_name: str,
        *args,
        **kwargs,
    ) -> List[Any]:
        """Alias for broadcast (backwards compatibility)."""
        return self.broadcast(method_name, *args, **kwargs)

    def execute_async(
        self,
        method_name: str,
        *args,
        **kwargs,
    ) -> List[Any]:
        """Alias for broadcast_async."""
        return self.broadcast_async(method_name, *args, **kwargs)

    @classmethod
    def from_pool(
        cls,
        name: str,
        resource_pool: 'ResourcePool',
        worker_cls: Any = None,
        num_gpus: float = 1.0,
    ) -> 'WorkerGroup':
        """Create a WorkerGroup by spawning actors on a ResourcePool.

        Args:
            name: Group name.
            resource_pool: The ResourcePool to allocate from.
            worker_cls: The Ray remote class to instantiate. Defaults
                to MegatronWorker.
            num_gpus: GPU fraction per actor. Use ``1.0`` for exclusive,
                ``1.0 / pool.max_colocate_count`` for colocated.

        Returns:
            A WorkerGroup with all actors created and scheduled.
        """
        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        if worker_cls is None:
            from .megatron_worker import MegatronWorker
            worker_cls = ray.remote(num_gpus=num_gpus)(MegatronWorker)

        placements = resource_pool.get_placements()
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

            worker = worker_cls.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=p['pg'], placement_group_bundle_index=p['bundle_idx']),
                runtime_env=RuntimeEnv(env_vars=env_vars),
            ).remote()
            workers.append(worker)

        logger.info('WorkerGroup %r created: %d workers', name, len(workers))
        return cls(name, workers)

    def ping(self) -> List[str]:
        """Health check: returns a list of pong strings."""
        return self.broadcast('ping')
