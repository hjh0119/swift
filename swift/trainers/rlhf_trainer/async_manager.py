"""
Async Training Manager for RLVR.

This module provides a clean interface for managing async rollout generation
in RLVR training, decoupled from the trainer implementation.

The AsyncTrainingManager handles:
1. Coordination between rollout generation and training
2. Policy version management
3. Integration with vLLM server for distributed rollout
"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from swift.utils import get_logger

if TYPE_CHECKING:
    from .vllm_client import VLLMClient

logger = get_logger()


@dataclass
class AsyncConfig:
    """Configuration for async training."""
    max_staleness: int = 2
    prefetch_count: int = 1
    poll_interval: float = 0.1
    timeout: float = 300.0


@dataclass
class RolloutResult:
    """Container for rollout results with metadata."""
    data: Any
    policy_version: int
    timestamp: float = field(default_factory=time.time)


class AsyncTrainingManager:
    """Manager for async rollout generation in RLVR training.

    This class provides:
    1. Clean interface for async rollout control
    2. Prefetch mechanism for overlapping generation and training
    3. Policy version tracking and staleness control
    4. Integration with VLLMClient for server-side async

    Parameters:
        vllm_client: VLLMClient instance for rollout server communication
        config: AsyncConfig with async training settings
    """

    def __init__(
        self,
        vllm_client: Optional['VLLMClient'] = None,
        config: Optional[AsyncConfig] = None,
    ):
        self.vllm_client = vllm_client
        self.config = config or AsyncConfig()

        # State
        self._running = False
        self._policy_version = 0
        self._version_lock = threading.Lock()

        # Prefetch queues
        self._train_queue: Queue[RolloutResult] = Queue()
        self._eval_queue: Queue[RolloutResult] = Queue()

        # Mode detection
        self._use_server_async = (vllm_client is not None and self.config.max_staleness > 1)

        # Statistics
        self._stats = {
            'rollouts_generated': 0,
            'rollouts_consumed': 0,
        }
        self._stats_lock = threading.Lock()

        logger.info(f'AsyncTrainingManager initialized: max_staleness={self.config.max_staleness}, '
                    f'server_mode={self._use_server_async}')

    # ==================== Lifecycle ====================

    def start(self) -> bool:
        """Start async generation."""
        if self._running:
            return True

        if self._use_server_async:
            try:
                result = self.vllm_client.start_async_generate(max_staleness=self.config.max_staleness, )
                logger.info(f'Started server async generate: {result}')
                self._running = True
                return True
            except Exception as e:
                logger.error(f'Failed to start server async: {e}')
                return False
        else:
            self._running = True
            return True

    def stop(self) -> bool:
        """Stop async generation."""
        if not self._running:
            return True

        if self._use_server_async:
            try:
                result = self.vllm_client.stop_async_generate()
                logger.info(f'Stopped server async generate: {result}')
            except Exception as e:
                logger.error(f'Error stopping server async: {e}')

        self._running = False
        return True

    def is_running(self) -> bool:
        """Check if async generation is running."""
        return self._running

    # ==================== Rollout Control ====================

    def push_requests(
        self,
        requests: List[Dict],
        config: Optional[Dict] = None,
    ) -> bool:
        """Push requests for async generation."""
        if not self._running:
            return False

        if self._use_server_async:
            try:
                result = self.vllm_client.push_inputs(requests, config)
                return result.get('success', False)
            except Exception as e:
                logger.error(f'Failed to push requests: {e}')
                return False

        return True

    def get_rollout(
        self,
        timeout: Optional[float] = None,
        for_eval: bool = False,
    ) -> Optional[RolloutResult]:
        """Get a rollout result."""
        if timeout is None:
            timeout = self.config.timeout

        queue = self._eval_queue if for_eval else self._train_queue

        try:
            result = queue.get(timeout=min(0.1, timeout))
            with self._stats_lock:
                self._stats['rollouts_consumed'] += 1
            return result
        except Empty:
            pass

        if self._use_server_async:
            try:
                response = self.vllm_client.get_samples(
                    timeout=timeout,
                    max_staleness=self.config.max_staleness,
                )
                if response.get('success'):
                    result = RolloutResult(
                        data=response['samples'],
                        policy_version=response.get('policy_version', 0),
                        timestamp=response.get('timestamp', time.time()),
                    )
                    with self._stats_lock:
                        self._stats['rollouts_consumed'] += 1
                    return result
            except Exception as e:
                logger.error(f'Failed to get samples: {e}')

        return None

    def has_rollout_ready(self, for_eval: bool = False) -> bool:
        """Check if a rollout is ready without blocking."""
        queue = self._eval_queue if for_eval else self._train_queue
        return not queue.empty()

    # ==================== Weight Update Coordination ====================

    def on_weights_updated(self) -> int:
        """Called after weights are updated. Returns new policy version."""
        with self._version_lock:
            self._policy_version += 1
            return self._policy_version

    def pause_for_weight_update(self) -> Dict[str, Any]:
        """Pause generation before weight update."""
        if not self._running:
            return {'success': False, 'message': 'Not running'}

        if self._use_server_async:
            try:
                return self.vllm_client.pause_async_generate()
            except Exception as e:
                return {'success': False, 'message': str(e)}

        return {'success': True}

    def resume_after_weight_update(self) -> Dict[str, Any]:
        """Resume generation after weight update."""
        if not self._running:
            return {'success': False, 'message': 'Not running'}

        if self._use_server_async:
            try:
                return self.vllm_client.resume_async_generate()
            except Exception as e:
                return {'success': False, 'message': str(e)}

        return {'success': True}

    def get_policy_version(self) -> int:
        """Get current policy version."""
        with self._version_lock:
            return self._policy_version

    # ==================== Prefetch Support ====================

    def prefetch(
        self,
        generate_fn: Callable[[], Any],
        for_eval: bool = False,
    ) -> None:
        """Start prefetching rollout in background."""

        def _prefetch_worker():
            try:
                data = generate_fn()
                result = RolloutResult(
                    data=data,
                    policy_version=self.get_policy_version(),
                )
                queue = self._eval_queue if for_eval else self._train_queue
                queue.put(result)
                with self._stats_lock:
                    self._stats['rollouts_generated'] += 1
            except Exception as e:
                logger.error(f'Prefetch failed: {e}')

        thread = threading.Thread(target=_prefetch_worker, daemon=True)
        thread.start()

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        with self._stats_lock:
            stats = dict(self._stats)
        with self._version_lock:
            stats['policy_version'] = self._policy_version

        stats['running'] = self._running
        stats['server_mode'] = self._use_server_async
        stats['train_queue_size'] = self._train_queue.qsize()
        stats['eval_queue_size'] = self._eval_queue.qsize()

        return stats

    # ==================== Queue Access (for compatibility) ====================

    @property
    def train_queue(self) -> Queue:
        """Access train queue directly."""
        return self._train_queue

    @property
    def eval_queue(self) -> Queue:
        """Access eval queue directly."""
        return self._eval_queue

    def put_train_result(self, result: Any) -> None:
        """Put result in train queue."""
        rollout_result = RolloutResult(
            data=result,
            policy_version=self.get_policy_version(),
        )
        self._train_queue.put(rollout_result)

    def put_eval_result(self, result: Any) -> None:
        """Put result in eval queue."""
        rollout_result = RolloutResult(
            data=result,
            policy_version=self.get_policy_version(),
        )
        self._eval_queue.put(rollout_result)


def create_async_manager(
    vllm_client: Optional['VLLMClient'] = None,
    max_staleness: int = 2,
    **kwargs,
) -> AsyncTrainingManager:
    """Create an AsyncTrainingManager with common settings."""
    config = AsyncConfig(max_staleness=max_staleness, **kwargs)
    return AsyncTrainingManager(vllm_client=vllm_client, config=config)
