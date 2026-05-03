"""Main application entry point for the real-time object counter."""

import argparse
import time
from typing import Optional, Union

import cv2

from src.config import Config
from src.video_source import VideoSource
from src.detector import ObjectDetector
from src.tracker import ByteTracker
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
        track_activation_threshold = self.config.get(
            "tracker.track_activation_threshold", 0.25
        )
        lost_track_buffer = self.config.get("tracker.lost_track_buffer", 30)
        minimum_matching_threshold = self.config.get(
            "tracker.minimum_matching_threshold", 0.8
        )
        self.tracker = ByteTracker(
            frame_rate=int(fps),
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )

        print(
            "Tracker: ByteTrack | "
            f"fps={fps:.0f}, "
            f"activation_threshold={track_activation_threshold}, "
            f"lost_track_buffer={lost_track_buffer}, "
            f"matching_threshold={minimum_matching_threshold}"
        )

        crossing_lines = self.config.get("counter.crossing_lines", None)
        self.counter = ObjectCounter(crossing_lines=crossing_lines)
        self.metrics = PerformanceMetrics()

        font_size = self.config.get("display.font_size", 1.0)
        self.renderer = FrameRenderer(font_size)

    def _get_configured_fps(self) -> float:
        """Get FPS value used for FPS-dependent tracker settings."""
        fps = self.config.get("video.fps", 30)

        if fps is None or fps <= 0:
            return 30.0

        return float(fps)

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

        show_counts = self.config.get("display.show_counts", True)

        if self.counter.has_crossing_lines():
            frame = self.renderer.render_crossing_lines(
                frame, self.counter.get_crossing_line_defs()
            )

        zone_panel_height = 0
        if show_counts and counts:
            frame = self.renderer.render_counts(
                frame,
                counts,
                self.counter.get_total_count(),
            )
            zone_panel_height = 70 + len(counts) * 28

        if self.counter.has_crossing_lines() and show_counts:
            y_start = zone_panel_height + 20 if zone_panel_height else 10
            frame = self.renderer.render_line_counts(
                frame,
                self.counter.get_line_counts(),
                y_start=y_start,
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

            if self.counter.has_crossing_lines():
                print("\nLine Crossing Counts:")
                for line_name, dirs in self.counter.get_line_counts().items():
                    in_total = sum(dirs["in"].values())
                    out_total = sum(dirs["out"].values())
                    print(f"  [{line_name}]  in={in_total}  out={out_total}")
                    for class_name, count in dirs["in"].items():
                        print(f"    in  {class_name}: {count}")
                    for class_name, count in dirs["out"].items():
                        print(f"    out {class_name}: {count}")


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