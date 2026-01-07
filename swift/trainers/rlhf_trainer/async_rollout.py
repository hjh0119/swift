"""
Async Rollout Infrastructure for Fully Asynchronous Policy Training.

This module provides a simplified, high-performance async rollout implementation
combining best practices from:
- AReaL: uvloop-based runner, clean lifecycle, shutdown hooks
- verl: staleness_threshold proportion control
- slime: simplicity and minimal overhead

Design principles:
1. Single unified AsyncRolloutRunner class
2. Integrated staleness management for capacity control
3. Minimal public API: submit/wait/pause/resume/destroy
4. Thread-safe with proper synchronization

Usage:
    runner = AsyncRolloutRunner(max_queue_size=100, max_staleness=2)
    runner.initialize(generate_fn=my_generate)

    # Submit work
    runner.submit(requests)

    # Get results
    results = runner.wait(count=32)

    # Weight update
    runner.pause()
    update_weights()
    runner.resume()
    runner.increment_version()

    # Cleanup
    runner.destroy()
"""

from __future__ import annotations
import asyncio
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Generic, List, Optional, TypeVar

from swift.utils import get_logger

# Use uvloop for better performance if available
try:
    import uvloop
    HAS_UVLOOP = True
except ImportError:
    HAS_UVLOOP = False

logger = get_logger()

T = TypeVar('T')

# ==================== Exceptions ====================


class AsyncRolloutError(Exception):
    """Base exception for async rollout errors."""
    pass


class QueueFullError(AsyncRolloutError):
    """Raised when the task queue is full."""
    pass


class RolloutTimeoutError(AsyncRolloutError, TimeoutError):
    """Raised when waiting for rollout results times out."""
    pass


class RolloutShutdownError(AsyncRolloutError):
    """Raised when rollout runner has been shut down."""
    pass


# ==================== Data Structures ====================


@dataclass
class RolloutStats:
    """Statistics for rollout generation tracking."""
    running: int = 0
    accepted: int = 0
    rejected: int = 0
    submitted: int = 0

    def copy(self) -> 'RolloutStats':
        return RolloutStats(
            running=self.running,
            accepted=self.accepted,
            rejected=self.rejected,
            submitted=self.submitted,
        )


@dataclass
class TimedResult(Generic[T]):
    """Wrapper for task results with timing and version info."""
    data: T
    policy_version: int
    task_id: int = 0
    create_time: int = field(default_factory=lambda: time.monotonic_ns())
    complete_time: int = 0

    @property
    def latency_ms(self) -> float:
        """Get latency in milliseconds."""
        if self.complete_time == 0:
            return 0.0
        return (self.complete_time - self.create_time) / 1e6


@dataclass
class _TaskInput:
    """Internal wrapper for task input."""
    requests: List[Dict]
    config: Optional[Dict]
    policy_version: int
    task_id: int
    create_time: int = field(default_factory=lambda: time.monotonic_ns())


# ==================== Async Rollout Runner ====================


class AsyncRolloutRunner(Generic[T]):
    """Unified high-performance async rollout runner.

    This class provides:
    1. Single asyncio event loop in a background thread (uvloop for performance)
    2. Integrated staleness management with capacity control
    3. Thread-safe submit/wait/pause/resume/destroy APIs
    4. Shutdown hooks and health monitoring
    5. Fail-fast error propagation for generation/reward errors

    Staleness Control Formula (from verl):
        max_samples = (max_staleness + current_version + 1) * consumer_batch_size
        staleness_capacity = max_samples - (accepted + running)
        capacity = min(concurrency_capacity, staleness_capacity)

    Parameters:
        max_queue_size: Maximum size for input/output queues
        max_staleness: Maximum allowed version difference (0 = sync, >0 = async)
        max_concurrent: Maximum concurrent generations
        consumer_batch_size: Expected batch size for consuming (for capacity calc)
        poll_wait_time: Time to wait for task completion in asyncio.wait
        poll_sleep_time: Time to sleep between poll cycles when idle
        fail_fast: If True, propagate generation errors to main thread (default: True)
        max_consecutive_errors: Max consecutive errors before fail-fast triggers (default: 3)
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        max_staleness: int = 2,
        max_concurrent: Optional[int] = None,
        consumer_batch_size: int = 1,
        poll_wait_time: float = 0.02,
        poll_sleep_time: float = 0.1,
        fail_fast: bool = True,
        max_consecutive_errors: int = 3,
    ):
        # Configuration
        self.max_queue_size = max_queue_size
        self.max_staleness = max_staleness
        self.max_concurrent = max_concurrent or max_queue_size
        self.consumer_batch_size = max(1, consumer_batch_size)
        self.poll_wait_time = poll_wait_time
        self.poll_sleep_time = poll_sleep_time
        self.fail_fast = fail_fast
        self.max_consecutive_errors = max_consecutive_errors

        # Error tracking for fail-fast
        self._consecutive_errors = 0
        self._last_error: Optional[Exception] = None
        self._last_error_time: float = 0.0

        # Statistics
        self._stats = RolloutStats()
        self._stats_lock = threading.Lock()

        # Version tracking
        self._version = 0
        self._version_lock = threading.Lock()

        # High-performance queues using deque + asyncio.Condition (verl pattern)
        # Input queue: thread-safe with threading.Condition
        self._input_queue: Deque[_TaskInput] = deque(maxlen=max_queue_size)
        self._input_lock = threading.Lock()
        self._input_not_empty = threading.Condition(self._input_lock)

        # Output queue: async-safe with asyncio.Condition (for efficient waiting)
        self._output_queue: Deque[TimedResult[T]] = deque(maxlen=max_queue_size)
        self._output_lock: Optional[asyncio.Lock] = None
        self._output_condition: Optional[asyncio.Condition] = None

        # Thread control
        self._exiting = threading.Event()
        self._paused = threading.Event()
        self._loop_ready = threading.Event()

        # Async loop references
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._input_signal: Optional[asyncio.Event] = None

        # Thread exception propagation (fail-fast)
        self._thread_exception: Optional[Exception] = None
        self._thread_exception_lock = threading.Lock()

        # Shutdown hooks (LIFO order)
        self._shutdown_hooks: List[Callable[[], Awaitable[None]]] = []
        self._shutdown_hooks_lock = threading.Lock()

        # Task ID counter
        self._task_id_counter = 0
        self._task_id_lock = threading.Lock()

        # Background thread
        self._thread: Optional[threading.Thread] = None

        # Generate function (set in initialize)
        self._generate_fn: Optional[Callable] = None

        logger.info(f'AsyncRolloutRunner initialized: max_queue={max_queue_size}, '
                    f'max_staleness={max_staleness}, max_concurrent={self.max_concurrent}')

    # ==================== Initialization ====================

    def initialize(
        self,
        generate_fn: Callable[[List[Dict], Optional[Dict]], Any],
    ) -> None:
        """Initialize and start the background thread.

        Args:
            generate_fn: Function that takes (requests, config) and returns results.
                        Can be sync or async.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning('Runner already initialized, skipping')
            return

        self._generate_fn = generate_fn

        # Reset state
        self._exiting.clear()
        self._paused.clear()
        self._loop_ready.clear()

        self._thread = threading.Thread(
            target=self._run_thread,
            daemon=True,
            name='AsyncRolloutRunner',
        )
        self._thread.start()

        # Wait for loop to be ready
        if not self._loop_ready.wait(timeout=10.0):
            raise RuntimeError('Failed to initialize async loop within timeout')

        logger.info('AsyncRolloutRunner started successfully')

    def destroy(self, timeout: float = 10.0) -> None:
        """Shutdown the runner and wait for cleanup."""
        if not self._exiting.is_set():
            self._exiting.set()
            self._paused.clear()
            self._signal_new_input()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(f'Background thread did not exit within {timeout}s')

        logger.info('AsyncRolloutRunner destroyed')

    def register_shutdown_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        """Register async cleanup function for shutdown (called in LIFO order)."""
        with self._shutdown_hooks_lock:
            if self._exiting.is_set():
                logger.warning('Shutdown hook registered after shutdown started')
                return
            self._shutdown_hooks.append(hook)

    # ==================== Capacity Control ====================

    def get_capacity(self) -> int:
        """Calculate available capacity for new rollouts."""
        with self._stats_lock:
            with self._version_lock:
                version = self._version

            # Concurrency-based capacity
            concurrency_cap = self.max_concurrent - self._stats.running

            # Staleness-based capacity (verl formula)
            max_samples = (self.max_staleness + version + 1) * self.consumer_batch_size
            staleness_cap = max_samples - (self._stats.accepted + self._stats.running)

            return max(0, min(concurrency_cap, staleness_cap))

    def is_stale(self, sample_version: int) -> bool:
        """Check if a sample is too stale."""
        with self._version_lock:
            return (self._version - sample_version) > self.max_staleness

    # ==================== Public API: Submit ====================

    def submit(
        self,
        requests: List[Dict],
        config: Optional[Dict] = None,
        policy_version: Optional[int] = None,
    ) -> int:
        """Submit requests for async generation.

        Args:
            requests: List of inference requests
            config: Optional generation config
            policy_version: Policy version (uses current if None)

        Returns:
            Task ID for tracking
        """
        self._check_health()

        with self._task_id_lock:
            task_id = self._task_id_counter
            self._task_id_counter += 1

        if policy_version is None:
            with self._version_lock:
                policy_version = self._version

        task_input = _TaskInput(
            requests=requests,
            config=config,
            policy_version=policy_version,
            task_id=task_id,
        )

        with self._input_lock:
            if len(self._input_queue) >= self.max_queue_size:
                raise QueueFullError(f'Input queue full (size={self.max_queue_size}). '
                                     'Increase max_queue_size or wait for tasks to complete.')
            self._input_queue.append(task_input)
            self._input_not_empty.notify()

        self._signal_new_input()
        return task_id

    # ==================== Public API: Wait ====================

    def wait(
        self,
        count: int,
        timeout: Optional[float] = None,
        with_timing: bool = False,
    ) -> List[T] | List[TimedResult[T]]:
        """Wait for specified number of results.

        Args:
            count: Number of results to wait for
            timeout: Maximum wait time (None = wait indefinitely)
            with_timing: Return TimedResult with metadata if True

        Returns:
            List of results (or TimedResult if with_timing=True)
        """
        if timeout is None:
            timeout = float(7 * 24 * 3600)  # 7 days

        deadline = time.time() + timeout
        results: List[TimedResult[T]] = []

        while len(results) < count:
            self._check_health()

            if self._exiting.is_set():
                raise RolloutShutdownError('Runner is shutting down')

            remaining = deadline - time.time()
            if remaining <= 0:
                raise RolloutTimeoutError(f'Timeout waiting for {count} results, got {len(results)}')

            # Use event-driven wait via asyncio from background thread
            result = self._wait_output_one(min(self.poll_sleep_time, remaining))
            if result is not None:
                results.append(result)

        # Mark as consumed for capacity tracking
        with self._stats_lock:
            self._stats.accepted = max(0, self._stats.accepted - len(results))

        if with_timing:
            return results
        return [r.data for r in results]

    def get_batch(self, max_count: int, timeout: float = 0.0) -> List[T]:
        """Get up to max_count results without blocking.

        Args:
            max_count: Maximum number of results to retrieve
            timeout: Time to wait for first result (0 = non-blocking)

        Returns:
            List of available results (may be empty)
        """
        results: List[TimedResult[T]] = []

        # Wait for first result if timeout > 0
        if timeout > 0 and len(self._output_queue) == 0:
            result = self._wait_output_one(timeout)
            if result is not None:
                results.append(result)

        # Drain up to max_count (non-blocking)
        while len(results) < max_count and len(self._output_queue) > 0:
            result = self._pop_output()
            if result is not None:
                results.append(result)
            else:
                break

        if results:
            with self._stats_lock:
                self._stats.accepted = max(0, self._stats.accepted - len(results))

        return [r.data for r in results]

    # ==================== Version Control ====================

    def increment_version(self) -> int:
        """Increment policy version after weight update."""
        with self._version_lock:
            self._version += 1
            new_version = self._version
        logger.info(f'Policy version incremented to {new_version}')
        return new_version

    def get_version(self) -> int:
        """Get current policy version."""
        with self._version_lock:
            return self._version

    def set_version(self, version: int) -> None:
        """Set policy version explicitly."""
        with self._version_lock:
            self._version = version

    # ==================== Pause/Resume ====================

    def pause(self) -> None:
        """Pause new task execution (for weight updates)."""
        self._paused.set()
        logger.debug('Runner paused')

    def resume(self) -> None:
        """Resume task execution."""
        self._paused.clear()
        self._signal_new_input()
        logger.debug('Runner resumed')

    def is_paused(self) -> bool:
        """Check if runner is paused."""
        return self._paused.is_set()

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self._stats_lock:
            stats = self._stats.copy()
        with self._version_lock:
            version = self._version

        result = {
            'policy_version': version,
            'capacity': self.get_capacity(),
            'input_queue_size': len(self._input_queue),
            'output_queue_size': len(self._output_queue),
            'running': stats.running,
            'accepted': stats.accepted,
            'rejected': stats.rejected,
            'submitted': stats.submitted,
            'paused': self._paused.is_set(),
            'exiting': self._exiting.is_set(),
            'max_staleness': self.max_staleness,
            'max_concurrent': self.max_concurrent,
            'consecutive_errors': self._consecutive_errors,
        }

        # Include last error info if available
        if self._last_error is not None:
            result['last_error'] = str(self._last_error)
            result['last_error_time'] = self._last_error_time

        # Include thread exception if fail-fast triggered
        with self._thread_exception_lock:
            if self._thread_exception is not None:
                result['thread_exception'] = str(self._thread_exception)

        return result

    # ==================== Internal Methods ====================

    def _check_health(self) -> None:
        """Check thread health and propagate exceptions (fail-fast)."""
        with self._thread_exception_lock:
            if self._thread_exception is not None:
                raise RolloutShutdownError(
                    f'Runner thread failed: {self._thread_exception}') from self._thread_exception

    def _signal_new_input(self) -> None:
        """Signal the async loop that new input is available."""
        loop = self._loop
        signal = self._input_signal
        if loop is not None and signal is not None:
            try:
                loop.call_soon_threadsafe(signal.set)
            except RuntimeError:
                pass  # Loop closed

    def _update_stats(self, **kwargs) -> None:
        """Thread-safe stats update."""
        with self._stats_lock:
            for key, delta in kwargs.items():
                if hasattr(self._stats, key):
                    current = getattr(self._stats, key)
                    setattr(self._stats, key, max(0, current + delta))

    def _wait_output_one(self, timeout: float) -> Optional[TimedResult[T]]:
        """Wait for one output result with timeout (thread-safe, blocking).

        This method is called from the main thread to wait for results.
        It uses polling with short sleeps since the output queue is managed
        by the async loop in the background thread.
        """
        deadline = time.time() + timeout
        while True:
            # Try to pop from output queue
            result = self._pop_output()
            if result is not None:
                return result

            # Check timeout
            remaining = deadline - time.time()
            if remaining <= 0:
                return None

            # Short sleep to avoid busy waiting
            time.sleep(min(0.01, remaining))

    def _pop_output(self) -> Optional[TimedResult[T]]:
        """Pop one result from output queue (thread-safe, non-blocking)."""
        # Output queue is only appended from async loop, safe to read length
        if len(self._output_queue) == 0:
            return None
        try:
            return self._output_queue.popleft()
        except IndexError:
            return None

    def _pop_input(self) -> Optional[_TaskInput]:
        """Pop one task from input queue (thread-safe, non-blocking)."""
        with self._input_lock:
            if len(self._input_queue) == 0:
                return None
            return self._input_queue.popleft()

    def _run_thread(self) -> None:
        """Entry point for background thread."""
        try:
            if HAS_UVLOOP:
                loop = uvloop.new_event_loop()
            else:
                loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_ready.set()
            loop.run_until_complete(self._run_async_loop())
        except Exception as e:
            with self._thread_exception_lock:
                self._thread_exception = e
            logger.error(f'Runner thread failed: {e}', exc_info=True)
        finally:
            self._exiting.set()
            self._loop_ready.set()
            if self._loop is not None:
                self._loop.close()
                self._loop = None

    async def _run_async_loop(self) -> None:
        """Main async event loop."""
        self._input_signal = asyncio.Event()
        running_tasks: Dict[int, asyncio.Task] = {}
        task_metadata: Dict[int, _TaskInput] = {}

        try:
            while not self._exiting.is_set():
                # Handle pause
                if self._paused.is_set():
                    await asyncio.sleep(self.poll_sleep_time)
                    continue

                # Submit new tasks based on capacity
                await self._submit_pending_tasks(running_tasks, task_metadata)

                # Wait for tasks to complete
                if not running_tasks:
                    await self._wait_for_new_tasks()
                    continue

                # Wait for any task to complete
                done, _ = await asyncio.wait(
                    list(running_tasks.values()),
                    timeout=self.poll_wait_time,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Process completed tasks
                for task in done:
                    tid = int(task.get_name())
                    meta = task_metadata.pop(tid, None)
                    del running_tasks[tid]

                    if meta is None:
                        continue

                    try:
                        results = await task
                        await self._process_results(results, meta)
                        # Reset error counter on success
                        self._consecutive_errors = 0
                    except asyncio.CancelledError:
                        logger.debug(f'Task {tid} cancelled')
                        self._update_stats(running=-1, rejected=1)
                    except Exception as e:
                        logger.error(f'Task {tid} failed: {e}', exc_info=True)
                        self._update_stats(running=-1, rejected=1)

                        # Track consecutive errors for fail-fast
                        self._consecutive_errors += 1
                        self._last_error = e
                        self._last_error_time = time.time()

                        if self.fail_fast and self._consecutive_errors >= self.max_consecutive_errors:
                            # Store exception and trigger shutdown
                            with self._thread_exception_lock:
                                self._thread_exception = RuntimeError(
                                    f'Rollout generation failed {self._consecutive_errors} consecutive times. '
                                    f'Last error: {e}')
                                self._thread_exception.__cause__ = e
                            logger.critical(f'FAIL-FAST: {self._consecutive_errors} consecutive rollout errors. '
                                            f'Shutting down runner. Last error: {e}')
                            self._exiting.set()
                            return

        finally:
            self._input_signal = None

            # Run shutdown hooks in reverse order (LIFO)
            with self._shutdown_hooks_lock:
                hooks = list(reversed(self._shutdown_hooks))
            for hook in hooks:
                try:
                    await hook()
                except Exception as e:
                    logger.error(f'Shutdown hook failed: {e}')

            # Cancel remaining tasks
            for task in running_tasks.values():
                if not task.done():
                    task.cancel()
            if running_tasks:
                await asyncio.gather(*running_tasks.values(), return_exceptions=True)

    async def _submit_pending_tasks(
        self,
        running_tasks: Dict[int, asyncio.Task],
        task_metadata: Dict[int, _TaskInput],
    ) -> None:
        """Submit pending tasks up to capacity."""
        while not self._paused.is_set():
            # Check capacity
            if self.get_capacity() <= 0:
                break

            # Get next task from input queue (using thread-safe method)
            task_input = self._pop_input()
            if task_input is None:
                break

            # Check if already stale before submitting
            if self.is_stale(task_input.policy_version):
                logger.debug(f'Dropping pre-stale task {task_input.task_id}')
                self._update_stats(rejected=1)
                continue

            # Create async task
            coro = self._execute_generation(task_input)
            task = asyncio.create_task(coro, name=str(task_input.task_id))

            running_tasks[task_input.task_id] = task
            task_metadata[task_input.task_id] = task_input
            self._update_stats(running=1, submitted=1)

    async def _execute_generation(self, task_input: _TaskInput) -> List[T]:
        """Execute generation for a task."""
        if self._generate_fn is None:
            raise RuntimeError('Generate function not set')

        result = self._generate_fn(task_input.requests, task_input.config)
        # Handle both sync and async generate functions
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _process_results(
        self,
        results: List[T],
        task_input: _TaskInput,
    ) -> None:
        """Process and filter results based on staleness."""
        # Check staleness
        if self.is_stale(task_input.policy_version):
            logger.debug(f'Dropping stale results: version {task_input.policy_version}, '
                         f'current {self.get_version()}')
            self._update_stats(running=-1, rejected=1)
            return

        # Wrap results with timing info
        complete_time = time.monotonic_ns()

        for result in results:
            timed_result = TimedResult(
                data=result,
                policy_version=task_input.policy_version,
                task_id=task_input.task_id,
                create_time=task_input.create_time,
                complete_time=complete_time,
            )
            # Append to output deque (thread-safe for single writer)
            if len(self._output_queue) >= self.max_queue_size:
                logger.warning('Output queue full, dropping result')
                self._update_stats(running=-1, rejected=1)
                return
            self._output_queue.append(timed_result)

        self._update_stats(running=-1, accepted=1)

    async def _wait_for_new_tasks(self) -> None:
        """Wait for new tasks to be submitted."""
        signal = self._input_signal
        if signal is None:
            await asyncio.sleep(self.poll_sleep_time)
            return

        # Double-check pattern to avoid race condition
        while not self._exiting.is_set() and not self._paused.is_set():
            if len(self._input_queue) > 0:
                return
            signal.clear()
            if len(self._input_queue) > 0 or self._exiting.is_set():
                return
            try:
                await asyncio.wait_for(signal.wait(), timeout=self.poll_sleep_time)
            except asyncio.TimeoutError:
                pass


# ==================== Factory Function ====================


def create_async_rollout_runner(
    max_queue_size: int = 100,
    max_staleness: int = 2,
    max_concurrent: Optional[int] = None,
    consumer_batch_size: int = 1,
) -> AsyncRolloutRunner:
    """Create an async rollout runner with common defaults.

    Args:
        max_queue_size: Maximum size for sample queue
        max_staleness: Maximum allowed version difference (0=sync, >0=async)
        max_concurrent: Maximum concurrent rollouts
        consumer_batch_size: Expected batch size for training

    Returns:
        Configured AsyncRolloutRunner instance
    """
    return AsyncRolloutRunner(
        max_queue_size=max_queue_size,
        max_staleness=max_staleness,
        max_concurrent=max_concurrent,
        consumer_batch_size=consumer_batch_size,
    )
