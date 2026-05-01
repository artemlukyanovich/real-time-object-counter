"""Main application module."""

import cv2
import time
import argparse
from pathlib import Path

from src.config import Config
from src.video_source import VideoSource
from src.detector import ObjectDetector
from src.tracker import CentroidTracker
from src.counter import ObjectCounter
from src.metrics import PerformanceMetrics
from src.renderer import FrameRenderer


class ObjectCounterApp:
    """Main application for real-time object counting."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        """Initialize application.

        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.video_source = None
        self.detector = None
        self.tracker = None
        self.counter = None
        self.metrics = None
        self.renderer = None
        self.video_writer = None

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all application components."""
        # Video source
        source = self.config.get('video.source', 0)
        width = self.config.get('video.frame_width', 1280)
        height = self.config.get('video.frame_height', 720)
        self.video_source = VideoSource(source, width, height)

        # Detector
        model = self.config.get('detector.model', 'yolov8n')
        conf = self.config.get('detector.confidence_threshold', 0.5)
        device = self.config.get('detector.device', 'cuda')
        self.detector = ObjectDetector(model, conf, device)

        # Tracker
        max_disappeared = self.config.get('tracker.max_disappeared', 50)
        max_distance = self.config.get('tracker.max_distance', 50)
        self.tracker = CentroidTracker(max_disappeared, max_distance)

        # Counter
        self.counter = ObjectCounter()

        # Metrics
        self.metrics = PerformanceMetrics()

        # Renderer
        font_size = self.config.get('display.font_size', 1.0)
        self.renderer = FrameRenderer(font_size)

    def run(self) -> None:
        """Run the application."""
        print("Starting object counter...")

        try:
            while True:
                frame_start = time.time()

                # Read frame
                ret, frame = self.video_source.read()
                if not ret:
                    print("End of video or camera disconnected")
                    break

                # Detection
                detect_start = time.time()
                detections = self.detector.detect(frame)
                detect_time = time.time() - detect_start
                self.metrics.record_detection_time(detect_time)

                # Tracking
                track_start = time.time()
                tracked_objects = self.tracker.update(detections)
                track_time = time.time() - track_start
                self.metrics.record_tracking_time(track_time)

                # Counting
                counts = self.counter.update(tracked_objects)

                # Rendering
                if self.config.get('display.show_detections', True):
                    frame = self.renderer.render_detections(frame, detections)

                if self.config.get('display.show_tracking_ids', True):
                    frame = self.renderer.render_tracks(frame, tracked_objects)

                if self.config.get('display.show_counts', True):
                    frame = self.renderer.render_counts(frame, counts,
                                                       self.counter.get_total_count())

                fps = self.metrics.get_fps()
                frame = self.renderer.render_fps(frame, fps)

                # Display
                cv2.imshow("Object Counter", frame)

                # Record metrics
                frame_time = time.time() - frame_start
                self.metrics.record_frame_time(frame_time)

                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.video_source:
            self.video_source.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

        # Print final metrics
        metrics_summary = self.metrics.get_summary()
        print("\nFinal Metrics:")
        for key, value in metrics_summary.items():
            print(f"  {key}: {value:.2f}")

        print(f"\nFinal Counts:")
        for class_name, count in self.counter.get_counts().items():
            print(f"  {class_name}: {count}")
        print(f"  Total: {self.counter.get_total_count()}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-time Object Counter")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--video", type=str, default=None,
                       help="Path to video file (overrides config)")
    parser.add_argument("--webcam", action="store_true",
                       help="Use webcam instead of video file")

    args = parser.parse_args()

    app = ObjectCounterApp(args.config)

    # Override video source if specified
    if args.webcam:
        app.config.config['video']['source'] = 0
        app._initialize_components()
    elif args.video:
        app.config.config['video']['source'] = args.video
        app._initialize_components()

    app.run()


if __name__ == "__main__":
    main()
