"""Resource tracking utilities for monitoring memory and CPU usage.

This module provides utilities for tracking resource consumption (memory, CPU, execution time)
during pipeline operations. Based on patterns from the Spapros pipeline.

Example usage:
    from utility_module._resource_tracker import ResourceTracker

    tracker = ResourceTracker("selection_rf_nmf")
    tracker.start()

    # ... your operation ...

    metrics = tracker.stop()
    print(f"Operation took {metrics.duration_minutes:.2f} min")
    print(f"Memory increased by {metrics.memory_increase_gb:.2f} GB")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Container for resource usage metrics.

    Attributes:
        operation: Name of the operation being tracked.
        start_time: ISO-formatted start timestamp.
        end_time: ISO-formatted end timestamp.
        duration_seconds: Total execution time in seconds.
        duration_minutes: Total execution time in minutes.
        memory_before_gb: Memory usage before operation (GB).
        memory_after_gb: Memory usage after operation (GB).
        memory_increase_gb: Memory increase during operation (GB).
        memory_increase_percent: Memory increase as percentage of initial memory.
        cpu_percent_start: CPU usage at start (optional).
        cpu_percent_end: CPU usage at end (optional).
    """

    operation: str
    start_time: str
    end_time: str
    duration_seconds: float
    duration_minutes: float
    memory_before_gb: float
    memory_after_gb: float
    memory_increase_gb: float
    memory_increase_percent: float
    cpu_percent_start: Optional[float] = None
    cpu_percent_end: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON export.

        Returns:
            Dictionary representation of metrics.
        """
        return asdict(self)


class ResourceTracker:
    """Track memory and CPU usage for operations.

    This class provides methods to track resource consumption before and after
    operations, including memory usage, CPU utilization, and execution time.

    Example:
        tracker = ResourceTracker("my_operation")
        tracker.start()
        # ... do work ...
        metrics = tracker.stop()

        # Save to CSV
        import pandas as pd
        df = pd.DataFrame([metrics.to_dict()])
        df.to_csv("metrics.csv")
    """

    def __init__(self, operation_name: str):
        """Initialize resource tracker.

        Args:
            operation_name: Name of operation being tracked (e.g., 'selection', 'evaluation').
        """
        self.operation_name = operation_name
        self.start_time_epoch = None
        self.memory_before = None
        self.cpu_before = None

        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not installed - memory tracking disabled")
            logger.warning("Install with: pip install psutil")

        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None

    def start(self) -> None:
        """Record start metrics.

        Captures the current timestamp, memory usage, and CPU utilization.
        Logs the start metrics to the logger.
        """
        self.start_time_epoch = time.time()

        if self.process:
            self.memory_before = self.process.memory_info().rss / (1024**3)  # GB
            self.cpu_before = self.process.cpu_percent(interval=0.1)

            logger.info(f"[{self.operation_name}] Start metrics:")
            logger.info(f"  Time: {self._format_timestamp(self.start_time_epoch)}")
            logger.info(f"  Memory: {self.memory_before:.2f} GB")
            logger.info(f"  CPU: {self.cpu_before:.1f}%")
        else:
            logger.info(
                f"[{self.operation_name}] Started at {self._format_timestamp(self.start_time_epoch)}"
            )

    def stop(self) -> ResourceMetrics:
        """Record end metrics and return summary.

        Captures the end timestamp, memory usage, and CPU utilization,
        then calculates deltas and returns a ResourceMetrics object.

        Returns:
            ResourceMetrics object with complete metrics.
        """
        end_time_epoch = time.time()
        duration_seconds = end_time_epoch - self.start_time_epoch
        duration_minutes = duration_seconds / 60

        if self.process:
            memory_after = self.process.memory_info().rss / (1024**3)  # GB
            cpu_after = self.process.cpu_percent(interval=0.1)
            memory_increase = memory_after - self.memory_before
            memory_increase_pct = (
                (memory_increase / self.memory_before * 100)
                if self.memory_before > 0
                else 0
            )

            logger.info(f"[{self.operation_name}] End metrics:")
            logger.info(f"  Time: {self._format_timestamp(end_time_epoch)}")
            logger.info(
                f"  Duration: {duration_minutes:.2f} min ({duration_seconds:.1f} s)"
            )
            logger.info(f"  Memory after: {memory_after:.2f} GB")
            logger.info(
                f"  Memory increase: {memory_increase:.2f} GB ({memory_increase_pct:.1f}%)"
            )
            logger.info(f"  CPU: {cpu_after:.1f}%")

            metrics = ResourceMetrics(
                operation=self.operation_name,
                start_time=self._format_timestamp(self.start_time_epoch),
                end_time=self._format_timestamp(end_time_epoch),
                duration_seconds=round(duration_seconds, 2),
                duration_minutes=round(duration_minutes, 2),
                memory_before_gb=round(self.memory_before, 2),
                memory_after_gb=round(memory_after, 2),
                memory_increase_gb=round(memory_increase, 2),
                memory_increase_percent=round(memory_increase_pct, 1),
                cpu_percent_start=round(self.cpu_before, 1),
                cpu_percent_end=round(cpu_after, 1),
            )
        else:
            # Fallback without psutil
            logger.info(
                f"[{self.operation_name}] Completed in {duration_minutes:.2f} min"
            )

            metrics = ResourceMetrics(
                operation=self.operation_name,
                start_time=self._format_timestamp(self.start_time_epoch),
                end_time=self._format_timestamp(end_time_epoch),
                duration_seconds=round(duration_seconds, 2),
                duration_minutes=round(duration_minutes, 2),
                memory_before_gb=0.0,
                memory_after_gb=0.0,
                memory_increase_gb=0.0,
                memory_increase_percent=0.0,
            )

        return metrics

    @staticmethod
    def _format_timestamp(timestamp: float) -> str:
        """Format timestamp as ISO string.

        Args:
            timestamp: Unix timestamp (seconds since epoch).

        Returns:
            ISO-formatted datetime string (YYYY-MM-DD HH:MM:SS).
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

    @staticmethod
    def get_current_memory_gb() -> float:
        """Get current memory usage in GB.

        Returns:
            Current RSS memory usage in gigabytes, or 0.0 if psutil unavailable.
        """
        if PSUTIL_AVAILABLE:
            return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        return 0.0

    @staticmethod
    def log_checkpoint(checkpoint_name: str) -> None:
        """Log memory at a specific checkpoint.

        Useful for tracking memory at intermediate points during long operations.

        Args:
            checkpoint_name: Descriptive name for this checkpoint.
        """
        if PSUTIL_AVAILABLE:
            mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
            logger.info(f"Memory at {checkpoint_name}: {mem_gb:.2f} GB")
        else:
            logger.debug(
                f"Checkpoint: {checkpoint_name} (memory logging disabled - psutil not installed)"
            )
