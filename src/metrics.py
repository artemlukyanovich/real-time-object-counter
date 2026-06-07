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
        self.io_wait_times = deque(maxlen=window_size)
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

    def record_io_wait(self, elapsed: float) -> None:
        """Record time the main loop spent waiting for the next frame.

        In the synchronous pipeline this is the decode time sitting on the
        critical path; with the async pipeline it is the time blocked on the
        frame queue, which should approach zero when decode is faster than
        inference. The single clearest indicator of the async speedup.
        """
        self.io_wait_times.append(elapsed)

    def get_fps(self) -> float:
        """Get current throughput FPS (frames pushed to the display per second).

        With frame skipping enabled this counts every output frame, including the
        cheap skipped frames where boxes are extrapolated rather than re-detected.
        """
        if not self.frame_times:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_inference_fps(self) -> float:
        """Get the model's raw inference rate (inferences per second).

        Derived from the time spent inside the detect+track pass only, so it
        reflects the true model cost regardless of how many frames are skipped.
        """
        if not self.detection_times:
            return 0.0
        avg = sum(self.detection_times) / len(self.detection_times)
        return 1.0 / avg if avg > 0 else 0.0

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

    def get_average_io_wait(self) -> float:
        """Get average per-frame I/O wait in ms (see record_io_wait)."""
        if not self.io_wait_times:
            return 0.0
        return (sum(self.io_wait_times) / len(self.io_wait_times)) * 1000

    def get_summary(self) -> Dict[str, float]:
        """Get metrics summary."""
        return {
            'fps': self.get_fps(),
            'inference_fps': self.get_inference_fps(),
            'avg_detection_time_ms': self.get_average_detection_time(),
            'avg_tracking_time_ms': self.get_average_tracking_time(),
            'avg_io_wait_ms': self.get_average_io_wait(),
            'total_frames': self.frame_count,
            'elapsed_seconds': time.time() - self.start_time,
        }

    def reset(self) -> None:
        """Reset metrics."""
        self.frame_times.clear()
        self.detection_times.clear()
        self.tracking_times.clear()
        self.io_wait_times.clear()
        self.frame_count = 0
        self.start_time = time.time()
