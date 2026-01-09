"""Staleness-aware capacity manager for asynchronous rollout generation.

This module provides the StalenessManager class which manages capacity
and staleness constraints for asynchronous rollout generation in RL training.

The staleness control ensures that rollouts don't become too stale (off-policy)
by limiting acceptance based on the current model version and maximum allowed
offpolicyness.

References:
    - AReaL: areal/core/staleness_manager.py
    - verl: verl/experimental/fully_async_policy/message_queue.py
"""

from __future__ import annotations
import threading
from dataclasses import dataclass
from typing import Protocol


@dataclass
class RolloutStat:
    """Statistics for rollout generation tracking.

    Attributes:
        enqueued: Number of rollouts waiting in input queue
        running: Number of rollouts currently being executed
        accepted: Number of completed rollouts accepted for training
        rejected: Number of completed rollouts rejected (stale or filtered)
    """

    enqueued: int = 0
    running: int = 0
    accepted: int = 0
    rejected: int = 0

    def copy(self) -> RolloutStat:
        """Create a copy of current statistics."""
        return RolloutStat(
            enqueued=self.enqueued,
            running=self.running,
            accepted=self.accepted,
            rejected=self.rejected,
        )


class VersionProvider(Protocol):
    """Protocol for version provider interface."""

    def get_version(self) -> int:
        """Get current model/policy version."""
        ...


class StalenessManager:
    """Manages rollout capacity based on staleness and concurrency constraints.

    The manager ensures that:
    1. The number of concurrent rollouts doesn't exceed the configured maximum
    2. Rollouts don't become too stale (off-policy) by limiting acceptance based on
       the current model version and maximum allowed offpolicyness

    Staleness Control Formula:
        max_samples = (max_staleness + current_version + 1) * consumer_batch_size
        capacity = min(concurrency_limit, max_samples - current_samples)

    This ensures that by the time samples are consumed, they won't exceed
    the maximum allowed staleness.

    Parameters:
        version_provider: Provider for current model version (implements get_version())
        max_concurrent_rollouts: Maximum number of concurrent rollouts allowed
        consumer_batch_size: Expected batch size for consuming rollouts during training
        max_staleness: Maximum allowed offpolicyness (version difference) for rollouts.
                      0 means synchronous (no staleness allowed), >0 means async.

    Example:
        >>> class Engine:
        ...     def __init__(self):
        ...         self._version = 0
        ...     def get_version(self):
        ...         return self._version
        >>>
        >>> engine = Engine()
        >>> manager = StalenessManager(
        ...     version_provider=engine,
        ...     max_concurrent_rollouts=8,
        ...     consumer_batch_size=4,
        ...     max_staleness=2,
        ... )
        >>> manager.get_capacity()  # Initially: 8 concurrent, 12 staleness = 8
        8
        >>> manager.on_rollout_submitted()
        >>> manager.get_capacity()  # Now: 7 concurrent, 11 staleness = 7
        7
    """

    def __init__(
        self,
        version_provider: VersionProvider,
        max_concurrent_rollouts: int,
        consumer_batch_size: int,
        max_staleness: int,
    ):
        self.version_provider = version_provider
        self.max_concurrent_rollouts = max(1, max_concurrent_rollouts)
        self.consumer_batch_size = max(1, consumer_batch_size)
        self.max_staleness = max(0, max_staleness)

        # Thread-safe access to rollout statistics
        self._lock = threading.Lock()
        self._stat = RolloutStat()

    def get_pending_limit(self) -> int:
        """Get the maximum number of pending rollouts allowed.

        Returns:
            Maximum number of pending rollouts (enqueued + running)
        """
        return (self.max_staleness + 1) * self.consumer_batch_size

    def get_capacity(self) -> int:
        """Calculate available capacity for new rollouts.

        Considers both concurrency limits and staleness constraints.
        Obtains current model version from version_provider.

        Returns:
            Number of new rollout slots available. Can be negative if over capacity.
        """
        with self._lock:
            current_version = self.version_provider.get_version()

            # Calculate concurrency-based capacity
            concurrency_capacity = self.max_concurrent_rollouts - self._stat.running

            # Calculate staleness-based capacity
            # max_samples = samples that can be consumed before becoming stale
            sample_cnt = self._stat.accepted + self._stat.running
            max_samples = (self.max_staleness + current_version + 1) * self.consumer_batch_size
            staleness_capacity = max_samples - sample_cnt

            # Return the minimum of both constraints
            return max(0, min(concurrency_capacity, staleness_capacity))

    def is_stale(self, sample_version: int) -> bool:
        """Check if a sample is too stale based on current version.

        Args:
            sample_version: The version when the sample was generated

        Returns:
            True if the sample is stale and should be rejected
        """
        current_version = self.version_provider.get_version()
        return (current_version - sample_version) > self.max_staleness

    def on_rollout_enqueued(self) -> None:
        """Callback when a rollout is enqueued as a pending input task."""
        with self._lock:
            self._stat.enqueued += 1

    def on_rollout_submitted(self) -> None:
        """Callback when a rollout starts execution (moves from queue to running)."""
        with self._lock:
            self._stat.enqueued = max(0, self._stat.enqueued - 1)
            self._stat.running += 1

    def on_rollout_accepted(self) -> None:
        """Callback when a rollout completes successfully and is accepted."""
        with self._lock:
            self._stat.running = max(0, self._stat.running - 1)
            self._stat.accepted += 1

    def on_rollout_rejected(self) -> None:
        """Callback when a rollout completes but is rejected (stale or filtered)."""
        with self._lock:
            self._stat.running = max(0, self._stat.running - 1)
            self._stat.rejected += 1

    def on_result_consumed(self, count: int = 1) -> None:
        """Callback when accepted results are consumed by trainer.

        Args:
            count: Number of results consumed
        """
        with self._lock:
            self._stat.accepted = max(0, self._stat.accepted - count)

    def reset(self) -> None:
        """Reset all statistics to zero."""
        with self._lock:
            self._stat = RolloutStat()

    def get_stats(self) -> RolloutStat:
        """Get a snapshot of current rollout statistics.

        Returns:
            Copy of current rollout statistics
        """
        with self._lock:
            return self._stat.copy()

    def get_summary(self) -> str:
        """Get a summary string of current statistics.

        Returns:
            Human-readable summary of current state
        """
        stats = self.get_stats()
        version = self.version_provider.get_version()
        capacity = self.get_capacity()
        return (f'version={version}, capacity={capacity}, '
                f'enqueued={stats.enqueued}, running={stats.running}, '
                f'accepted={stats.accepted}, rejected={stats.rejected}')
