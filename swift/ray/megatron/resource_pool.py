# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from swift.utils import find_free_port, get_logger

logger = get_logger()


@dataclass
class ResourcePool:
    """A pool of GPU resources backed by a single Ray placement group.

    Args:
        process_on_nodes: GPUs per node. ``[4, 4]`` = 8 GPUs on 2 nodes.
        max_colocate_count: How many WorkerGroups share these GPUs.
    """

    process_on_nodes: List[int]
    max_colocate_count: int = 1

    _pg: Any = field(default=None, repr=False, init=False)
    _master_info: Optional[Tuple[str, int]] = field(default=None, repr=False, init=False)

    @property
    def world_size(self) -> int:
        return sum(self.process_on_nodes)

    @property
    def placement_group(self):
        if self._pg is None:
            raise RuntimeError('Call create() first.')
        return self._pg

    @property
    def master_addr(self) -> str:
        if self._master_info is None:
            raise RuntimeError('Call create() first.')
        return self._master_info[0]

    @property
    def master_port(self) -> int:
        if self._master_info is None:
            raise RuntimeError('Call create() first.')
        return self._master_info[1]

    def create(self):
        """Create the Ray placement group and discover master addr/port."""
        import ray
        from ray.util.placement_group import placement_group

        bundles = []
        for n_gpus in self.process_on_nodes:
            for _ in range(n_gpus):
                bundles.append({
                    'GPU': 1,
                    'CPU': self.max_colocate_count,
                })

        self._pg = placement_group(bundles, strategy='PACK')
        ray.get(self._pg.ready())
        self._master_info = self._discover_master()

    def _run_on_bundle0(self, remote_fn):
        """Schedule a lightweight Ray task on bundle 0 and return result."""
        import ray
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        return ray.get(
            remote_fn.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._pg, placement_group_bundle_index=0), ).remote())

    def _discover_master(self) -> Tuple[str, int]:
        """Find master IP + free port on the node hosting bundle 0."""
        import ray

        @ray.remote(num_gpus=0, num_cpus=0.01)
        def _probe():
            addr = ray.util.get_node_ip_address()
            port = find_free_port()
            return addr, port

        return self._run_on_bundle0(_probe)

    def get_placements(
        self,
        master_port: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Per-rank placement dicts for creating Ray actors."""
        ws = self.world_size
        port = master_port if master_port is not None else self.master_port
        return [{
            'pg': self._pg,
            'bundle_idx': i,
            'rank': i,
            'world_size': ws,
            'master_addr': self.master_addr,
            'master_port': port,
        } for i in range(ws)]

    def discover_free_port(self) -> int:
        """Find another free port on the master node (for co-location)."""
        import ray

        @ray.remote(num_gpus=0, num_cpus=0.01)
        def _port():
            return find_free_port()

        return self._run_on_bundle0(_port)

    def destroy(self):
        if self._pg is not None:
            import ray
            ray.util.remove_placement_group(self._pg)
            self._pg = None
            self._master_info = None


class ResourcePoolManager:
    """Maps group names to ``ResourcePool`` instances.

    Supports separated (each group its own pool) and co-located
    (multiple groups share one pool) layouts.  Deduplicates ``create()``
    calls on shared pools.
    """

    def __init__(self, pool_mapping: Dict[str, ResourcePool]):
        self._pools = pool_mapping
        self._created = False

    def get_pool(self, group_name: str) -> ResourcePool:
        if group_name not in self._pools:
            raise KeyError('Unknown group %r' % group_name)
        return self._pools[group_name]

    def create_all(self):
        if self._created:
            return
        seen: set = set()
        for pool in self._pools.values():
            pid = id(pool)
            if pid not in seen:
                seen.add(pid)
                pool.create()
        self._created = True

    def destroy_all(self):
        seen: set = set()
        for pool in self._pools.values():
            pid = id(pool)
            if pid not in seen:
                seen.add(pid)
                pool.destroy()
        self._created = False

    def __enter__(self):
        self.create_all()
        return self

    def __exit__(self, *exc):
        self.destroy_all()
