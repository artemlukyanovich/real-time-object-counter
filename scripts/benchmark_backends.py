"""Compare inference speed across model backends (.pt / .onnx / .engine).

Runs the same video through each model and reports per-frame latency and FPS,
so you can see what ONNX / TensorRT export actually buys you on this machine.

This benchmarks the detection forward pass (ObjectDetector.detect) — the part
that export accelerates. Tracking/counting overhead is backend-independent and
is intentionally excluded for a clean apples-to-apples comparison.

Usage
-----
    python -m scripts.benchmark_backends \
        --source data/input/video.mp4 \
        --models models/yolov8n.pt models/yolov8n.onnx models/yolov8n.engine \
        --max-frames 300 --warmup 20

Notes
-----
- The first frames of each run are slow (lazy init, TensorRT engine load,
  CUDA warmup); --warmup excludes them from the timing.
- A .engine only runs on the GPU it was built on; if it is missing simply omit
  it from --models.
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running as `python -m scripts.benchmark_backends` from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.detector import ObjectDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed across model backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video file path or webcam index.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="One or more model paths to compare (.pt / .onnx / .engine).",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Number of timed frames per model (after warmup).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Frames to run before timing starts (excluded from results).",
    )
    return parser.parse_args()


def _normalize_source(source: str):
    return int(source) if source.isdigit() else source


def benchmark_model(
    model_path: str,
    source,
    conf: float,
    device: str,
    max_frames: int,
    warmup: int,
) -> dict:
    """Run one model over the source and return latency stats (milliseconds)."""
    detector = ObjectDetector(model_path, conf, device)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Failed to open source: {source}")

    timings_ms: list = []
    frames_seen = 0

    try:
        while frames_seen < warmup + max_frames:
            ok, frame = cap.read()
            if not ok:
                if isinstance(source, str):
                    # Loop the file if it is shorter than the requested budget.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            start = time.perf_counter()
            detector.detect(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if frames_seen >= warmup:
                timings_ms.append(elapsed_ms)
            frames_seen += 1
    finally:
        cap.release()

    arr = np.asarray(timings_ms, dtype=np.float64)
    avg = float(arr.mean()) if arr.size else 0.0
    return {
        "model": Path(model_path).name,
        "backend": detector.backend,
        "frames": int(arr.size),
        "avg_ms": avg,
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "fps": (1000.0 / avg) if avg > 0 else 0.0,
    }


def print_results(results: list) -> None:
    baseline_fps = results[0]["fps"] if results else 0.0

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Backend inference benchmark")
        for col in ("Model", "Backend", "Frames", "Avg ms", "p50 ms", "p95 ms", "FPS", "Speedup"):
            table.add_column(col, justify="right" if col not in ("Model", "Backend") else "left")
        for r in results:
            speedup = (r["fps"] / baseline_fps) if baseline_fps > 0 else 0.0
            table.add_row(
                r["model"], r["backend"], str(r["frames"]),
                f"{r['avg_ms']:.2f}", f"{r['p50_ms']:.2f}", f"{r['p95_ms']:.2f}",
                f"{r['fps']:.1f}", f"{speedup:.2f}x",
            )
        Console().print(table)
    except ImportError:
        print("\nBackend inference benchmark")
        for r in results:
            speedup = (r["fps"] / baseline_fps) if baseline_fps > 0 else 0.0
            print(
                f"  {r['model']:<24} {r['backend']:<9} "
                f"avg={r['avg_ms']:6.2f}ms  fps={r['fps']:6.1f}  speedup={speedup:.2f}x"
            )


def main() -> None:
    args = parse_args()
    source = _normalize_source(args.source)

    results = []
    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"Skipping missing model: {model_path}")
            continue
        print(f"\nBenchmarking {model_path} ...")
        results.append(
            benchmark_model(
                model_path, source, args.conf, args.device,
                args.max_frames, args.warmup,
            )
        )

    if results:
        print_results(results)


if __name__ == "__main__":
    main()
