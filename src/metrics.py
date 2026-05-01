"""Performance metrics module."""

import time
from typing import Dict, List
from collections import deque


class PerformanceMetrics:
    """Track performance metrics."""

    def __init__(self, window_size: int = 30):
        """Initialize metrics.

        Args:
            window_size: Window size for FPS calculation
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.tracking_times = deque(maxlen=window_size)
        self.frame_count = 0
        self.start_time = time.time()

    def record_frame_time(self, elapsed: float) -> None:
        """Record frame processing time."""
        self.frame_times.append(elapsed)
        self.frame_count += 1

    def record_detection_time(self, elapsed: float) -> None:
        """Record detection processing time."""
        self.detection_times.append(elapsed)

    def record_tracking_time(self, elapsed: float) -> None:
        """Record tracking processing time."""
        self.tracking_times.append(elapsed)

    def get_fps(self) -> float:
        """Get current FPS."""
        if not self.frame_times:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_average_detection_time(self) -> float:
        """Get average detection time in ms."""
        if not self.detection_times:
            return 0.0
        return (sum(self.detection_times) / len(self.detection_times)) * 1000

    def get_average_tracking_time(self) -> float:
        """Get average tracking time in ms."""
        if not self.tracking_times:
            return 0.0
        return (sum(self.tracking_times) / len(self.tracking_times)) * 1000

    def get_summary(self) -> Dict[str, float]:
        """Get metrics summary."""
        return {
            'fps': self.get_fps(),
            'avg_detection_time_ms': self.get_average_detection_time(),
            'avg_tracking_time_ms': self.get_average_tracking_time(),
            'total_frames': self.frame_count,
            'elapsed_seconds': time.time() - self.start_time,
        }

    def reset(self) -> None:
        """Reset metrics."""
        self.frame_times.clear()
        self.detection_times.clear()
        self.tracking_times.clear()
        self.frame_count = 0
        self.start_time = time.time()
