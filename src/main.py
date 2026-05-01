"""Main application entry point for the real-time object counter."""

import argparse
import time
from typing import Optional, Union

import cv2

from src.config import Config
from src.video_source import VideoSource
from src.detector import ObjectDetector
from src.tracker import CentroidTracker
from src.counter import ObjectCounter
from src.metrics import PerformanceMetrics
from src.renderer import FrameRenderer


class ObjectCounterApp:
    """Real-time object detection, tracking, and counting application."""

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        source_override: Optional[Union[str, int]] = None,
    ) -> None:
        self.config = Config(config_path)
        self.source_override = source_override

        self.video_source = None
        self.detector = None
        self.tracker = None
        self.counter = None
        self.metrics = None
        self.renderer = None

        self._initialize_components()

    def _initialize_components(self) -> None:
        source = self.source_override
        if source is None:
            source = self.config.get("video.source", 0)

        width = self.config.get("video.frame_width", 1280)
        height = self.config.get("video.frame_height", 720)
        self.video_source = VideoSource(source, width, height)

        model = self.config.get("detector.model", "yolov8n.pt")
        confidence = self.config.get("detector.confidence_threshold", 0.5)
        device = self.config.get("detector.device", "cuda")
        self.detector = ObjectDetector(model, confidence, device)

        fps = self._get_configured_fps()
        max_disappeared = self._resolve_max_disappeared(fps)
        max_distance = self._resolve_max_distance(fps)
        self.tracker = CentroidTracker(max_disappeared, max_distance)

        print(
            "Tracker settings: "
            f"fps={fps:.2f}, "
            f"max_disappeared={max_disappeared}, "
            f"max_distance={max_distance:.2f}"
        )

        self.counter = ObjectCounter()
        self.metrics = PerformanceMetrics()

        font_size = self.config.get("display.font_size", 1.0)
        self.renderer = FrameRenderer(font_size)

    def _get_configured_fps(self) -> float:
        """Get FPS value used for FPS-dependent tracker settings."""
        fps = self.config.get("video.fps", 30)

        if fps is None or fps <= 0:
            return 30.0

        return float(fps)

    def _resolve_max_distance(self, fps: float) -> float:
        """Resolve max centroid matching distance from config."""
        max_distance = self.config.get_raw("tracker.max_distance")

        if max_distance is not None:
            return float(max_distance)

        base = self.config.get("tracker.auto_max_distance_base", 50)
        reference_fps = self.config.get("tracker.auto_max_distance_reference_fps", 30)
        cap = self.config.get("tracker.auto_max_distance_cap", 200)

        if fps <= 0:
            fps = float(reference_fps)

        resolved_distance = float(base) * float(reference_fps) / fps

        if cap is not None:
            resolved_distance = min(resolved_distance, float(cap))

        return resolved_distance

    def _resolve_max_disappeared(self, fps: float) -> int:
        """Resolve max disappeared frames from config."""
        max_disappeared = self.config.get_raw("tracker.max_disappeared")

        if max_disappeared is not None:
            return int(max_disappeared)

        auto_seconds = self.config.get("tracker.auto_max_disappeared_seconds", 1.5)
        resolved_frames = round(fps * float(auto_seconds))

        min_frames = self.config.get("tracker.auto_max_disappeared_min", 5)
        max_frames = self.config.get("tracker.auto_max_disappeared_max", 50)

        resolved_frames = max(int(min_frames), resolved_frames)

        if max_frames is not None:
            resolved_frames = min(resolved_frames, int(max_frames))

        return resolved_frames

    def run(self) -> None:
        print("Starting object counter. Press 'q' to exit.")
        print(f"Device: {self.config.get('detector.device', 'Not specified')}")

        try:
            while True:
                frame_start = time.perf_counter()

                success, frame = self.video_source.read()
                if not success:
                    print("End of video stream or camera disconnected.")
                    break

                detect_start = time.perf_counter()
                detections = self.detector.detect(frame)
                self.metrics.record_detection_time(
                    time.perf_counter() - detect_start
                )

                track_start = time.perf_counter()
                tracked_objects = self.tracker.update(detections)
                self.metrics.record_tracking_time(
                    time.perf_counter() - track_start
                )

                counts = self.counter.update(tracked_objects)

                frame = self._render_frame(frame, detections, tracked_objects, counts)

                self.metrics.record_frame_time(time.perf_counter() - frame_start)

                cv2.imshow("Object Counter", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.cleanup()

    def _render_frame(self, frame, detections, tracked_objects, counts):
        if self.config.get("display.show_detections", True):
            frame = self.renderer.render_detections(frame, detections)

        if self.config.get("display.show_tracking_ids", True):
            frame = self.renderer.render_tracks(frame, tracked_objects)

        if self.config.get("display.show_counts", True):
            frame = self.renderer.render_counts(
                frame,
                counts,
                self.counter.get_total_count(),
            )

        fps = self.metrics.get_fps()
        frame = self.renderer.render_fps(frame, fps)

        return frame

    def cleanup(self) -> None:
        if self.video_source:
            self.video_source.release()

        cv2.destroyAllWindows()

        if self.metrics:
            print("\nFinal Metrics:")
            for key, value in self.metrics.get_summary().items():
                print(f"  {key}: {value:.2f}")

        if self.counter:
            print("\nFinal Counts:")
            for class_name, count in self.counter.get_counts().items():
                print(f"  {class_name}: {count}")
            print(f"  Total: {self.counter.get_total_count()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time Object Counter")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: webcam index or path to video file",
    )

    return parser.parse_args()


def normalize_source(source: Optional[str]):
    if source is None:
        return None

    if source.isdigit():
        return int(source)

    return source


def main() -> None:
    args = parse_args()
    source = normalize_source(args.source)

    app = ObjectCounterApp(
        config_path=args.config,
        source_override=source,
    )
    app.run()


if __name__ == "__main__":
    main()