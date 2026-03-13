# Copyright (c) ModelScope Contributors. All rights reserved.
"""GPU resource pool management for Ray-based distributed training.

Design references:
  - verl: ResourcePool / RayResourcePool / ResourcePoolManager
  - slime: create_placement_groups with single PG + offset slicing

Key concepts:
  - ResourcePool: a set of GPUs on one or more nodes, backed by one
    Ray placement group.  Multiple WorkerGroups can share the same
    pool (colocated) by using fractional GPU resources.
  - ResourcePoolManager: maps logical group names to ResourcePools.
    Handles both separated (different pools) and colocated (same pool)
    layouts.
"""
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from swift.utils import get_logger

logger = get_logger()


@dataclass
class ResourcePool:
    """A pool of GPU resources backed by a single Ray placement group.

    Args:
        process_on_nodes: Number of GPUs per node. E.g. ``[4, 4]``
            means 4 GPUs on each of 2 nodes (8 total).
        max_colocate_count: How many WorkerGroups can share the same
            GPUs.  ``1`` = exclusive, ``>1`` = colocated.
    """

    process_on_nodes: List[int]
    max_colocate_count: int = 1
    cpus_per_worker: int = 4

    _pg: Any = field(default=None, repr=False, init=False)
    _master_info: Optional[Tuple[str, int]] = field(default=None, repr=False, init=False)

    @property
    def world_size(self) -> int:
        return sum(self.process_on_nodes)

    @property
    def num_nodes(self) -> int:
        return len(self.process_on_nodes)

    @property
    def placement_group(self):
        if self._pg is None:
            raise RuntimeError('Placement group not created. Call create() first.')
        return self._pg

    @property
    def master_addr(self) -> str:
        if self._master_info is None:
            raise RuntimeError('Master info not discovered. Call create() first.')
        return self._master_info[0]

    @property
    def master_port(self) -> int:
        if self._master_info is None:
            raise RuntimeError('Master info not discovered. Call create() first.')
        return self._master_info[1]

    def create(self):
        """Create the Ray placement group and discover master address."""
        import ray
        from ray.util.placement_group import placement_group as ray_placement_group

        bundles = []
        for n_gpus in self.process_on_nodes:
            for _ in range(n_gpus):
                total_cpus = self.cpus_per_worker * self.max_colocate_count
                bundle = {'GPU': 1, 'CPU': total_cpus}
                bundles.append(bundle)

        self._pg = ray_placement_group(bundles, strategy='PACK')
        ray.get(self._pg.ready())
        logger.info('Placement group created: %d bundles, world_size=%d', len(bundles), self.world_size)
        self._master_info = self._discover_master()

    def _discover_master(self) -> Tuple[str, int]:
        """Discover master address/port on the node hosting bundle 0.

        Uses SO_REUSEADDR to mitigate TOCTOU races when multiple
        pools discover ports concurrently on the same node.
        """
        import ray
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        @ray.remote(num_gpus=0, num_cpus=0.01)
        def _get_addr_port():
            addr = ray.util.get_node_ip_address()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', 0))
            port = s.getsockname()[1]
            s.close()
            return addr, port

        addr, port = ray.get(
            _get_addr_port.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._pg, placement_group_bundle_index=0), ).remote())
        logger.info('Master discovered: %s:%d', addr, port)
        return addr, port

    def get_placements(self) -> List[Dict[str, Any]]:
        """Return per-rank placement dicts for creating Ray actors.

        Each dict contains: ``pg``, ``bundle_idx``, ``rank``,
        ``world_size``, ``master_addr``, ``master_port``.
        """
        ws = self.world_size
        return [{
            'pg': self._pg,
            'bundle_idx': i,
            'rank': i,
            'world_size': ws,
            'master_addr': self.master_addr,
            'master_port': self.master_port,
        } for i in range(ws)]

    def destroy(self):
        """Remove the placement group and release resources."""
        if self._pg is not None:
            import ray
            ray.util.remove_placement_group(self._pg)
            self._pg = None
            self._master_info = None


class ResourcePoolManager:
    """Manages multiple ResourcePools and maps group names to them.

    Supports two GPU allocation patterns:

    1. **Separated** -- each group gets its own dedicated pool::

        manager = ResourcePoolManager({
            'train': ResourcePool([4]),
            'ref':   ResourcePool([2]),
        })

    2. **Colocated** -- multiple groups share one pool::

        shared = ResourcePool([8], max_colocate_count=2)
        manager = ResourcePoolManager({
            'train': shared,
            'ref':   shared,
        })

    Deduplicates pool creation so shared pools are created only once.
    """

    def __init__(self, pool_mapping: Dict[str, ResourcePool]):
        self._pool_mapping = pool_mapping
        self._created = False

    @property
    def group_names(self) -> List[str]:
        return list(self._pool_mapping.keys())

    def get_pool(self, group_name: str) -> ResourcePool:
        if group_name not in self._pool_mapping:
            raise KeyError('Unknown group: %r. Available: %s' % (group_name, self.group_names))
        return self._pool_mapping[group_name]

    def create_all(self):
        """Create placement groups for all unique pools."""
        if self._created:
            return
        seen: set = set()
        for name, pool in self._pool_mapping.items():
            pool_id = id(pool)
            if pool_id not in seen:
                seen.add(pool_id)
                logger.info('Creating resource pool for group(s) '
                            'containing %r: world_size=%d', name, pool.world_size)
                pool.create()
        self._created = True

    def destroy_all(self):
        """Destroy all placement groups."""
        seen: set = set()
        for pool in self._pool_mapping.values():
            pool_id = id(pool)
            if pool_id not in seen:
                seen.add(pool_id)
                pool.destroy()
        self._created = False

    def __enter__(self):
        self.create_all()
        return self

    def __exit__(self, *exc):
        self.destroy_all()
