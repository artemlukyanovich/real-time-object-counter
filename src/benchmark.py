"""Pipeline performance benchmarking.

Usage example::

    bench = PipelineBenchmark()

    bench.start_frame()

    bench.start("detection")
    detections, tracks = tracker.update(frame)
    bench.stop("detection")

    bench.start("reid")
    reid_result = reid_manager.update(frame, tracks, frame_idx)
    bench.stop("reid")

    bench.end_frame()

    # At the end of the run:
    print(bench.summary())
    bench.save("run_001.json")
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


class _StageTimer:
    """Accumulates elapsed times for a single named pipeline stage."""

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.times: List[float] = []

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = time.perf_counter() - self._start
        self.times.append(elapsed)
        self._start = None
        return elapsed

    def mean_ms(self) -> float:
        if not self.times:
            return 0.0
        return (sum(self.times) / len(self.times)) * 1000.0

    def total_ms(self) -> float:
        return sum(self.times) * 1000.0

    def count(self) -> int:
        return len(self.times)


class PipelineBenchmark:
    """Measures per-frame and per-stage timing for the full pipeline.

    Recognised stage names (use these as keys for ``start``/``stop``):
        - ``"detection"``
        - ``"tracking"``
        - ``"crop"``
        - ``"embedding"``
        - ``"similarity"``
        - ``"reid"``
        - ``"render"``

    Any other stage name is also accepted.
    """

    def __init__(self, output_dir: str = "outputs/benchmarks") -> None:
        self.output_dir = Path(output_dir)
        self._stages: Dict[str, _StageTimer] = defaultdict(_StageTimer)
        self._frame_times: List[float] = []
        self._frame_start: Optional[float] = None
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Frame-level timing
    # ------------------------------------------------------------------

    def start_frame(self) -> None:
        """Call at the beginning of each frame processing loop."""
        self._frame_start = time.perf_counter()

    def end_frame(self) -> float:
        """Call at the end of each frame. Returns elapsed time in seconds."""
        if self._frame_start is None:
            return 0.0
        elapsed = time.perf_counter() - self._frame_start
        self._frame_times.append(elapsed)
        self._frame_count += 1
        self._frame_start = None
        return elapsed

    # ------------------------------------------------------------------
    # Stage-level timing
    # ------------------------------------------------------------------

    def start(self, stage: str) -> None:
        """Start timing a named pipeline stage."""
        self._stages[stage].start()

    def stop(self, stage: str) -> float:
        """Stop timing a named stage. Returns elapsed time in seconds."""
        return self._stages[stage].stop()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg if avg > 0 else 0.0

    def summary(self) -> dict:
        """Return a dict with FPS and per-stage mean latencies (ms)."""
        result: dict = {
            "frames": self._frame_count,
            "fps": round(self.fps(), 2),
        }
        for name, timer in self._stages.items():
            result[f"{name}_mean_ms"] = round(timer.mean_ms(), 3)
            result[f"{name}_calls"] = timer.count()
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: str = "benchmark.json") -> Path:
        """Write summary JSON to ``output_dir / filename``."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        return path

    def reset(self) -> None:
        """Clear all accumulated measurements."""
        self._stages.clear()
        self._frame_times.clear()
        self._frame_count = 0
        self._frame_start = None
