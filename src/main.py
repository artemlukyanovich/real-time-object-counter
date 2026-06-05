"""Main application entry point for the real-time object counter."""

import argparse
import time
from typing import Dict, Optional, Union

import cv2

from src.config import Config
from src.video_source import VideoSource
from src.detector import ObjectDetector
from src.tracker import UltralyticsTracker
from src.counter import ObjectCounter
from src.metrics import PerformanceMetrics
from src.renderer import FrameRenderer


class ObjectCounterApp:
    """Real-time object detection, tracking, and counting application."""

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        source_override: Optional[Union[str, int]] = None,
        half_override: Optional[bool] = None,
    ) -> None:
        self.config = Config(config_path)
        self.source_override = source_override
        self.half_override = half_override

        self.video_source = None
        self.detector = None
        self.tracker = None
        self.counter = None
        self.metrics = None
        self.renderer = None
        self.reid_manager = None

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
        half = self.half_override
        if half is None:
            half = self.config.get("detector.half", False)
        self.detector = ObjectDetector(model, confidence, device, half)

        fps = self._get_configured_fps()
        algorithm = self.config.get("tracker.algorithm", "bytetrack")
        track_activation_threshold = self.config.get(
            "tracker.track_activation_threshold", 0.25
        )
        track_low_threshold = self.config.get("tracker.track_low_threshold", 0.1)
        lost_track_buffer = self._resolve_lost_track_buffer(fps)
        matching_cost_threshold = self.config.get(
            "tracker.matching_cost_threshold", 0.8
        )
        fuse_score = self.config.get("tracker.fuse_score", True)
        gmc_method = self.config.get("tracker.gmc_method", "sparseOptFlow")
        reid_weights = self.config.get(
            "tracker.reid_weights", "osnet_x0_25_market.pt"
        )
        proximity_threshold = self.config.get("tracker.proximity_threshold", 0.5)
        appearance_threshold = self.config.get("tracker.appearance_threshold", 0.25)
        allowed_classes = self.config.get("detector.allowed_classes", None)
        self.tracker = UltralyticsTracker(
            model=self.detector.model,
            conf_threshold=confidence,
            frame_rate=int(fps),
            algorithm=algorithm,
            track_activation_threshold=track_activation_threshold,
            track_low_threshold=track_low_threshold,
            lost_track_buffer=lost_track_buffer,
            matching_cost_threshold=matching_cost_threshold,
            fuse_score=fuse_score,
            gmc_method=gmc_method,
            reid_weights=reid_weights,
            proximity_threshold=proximity_threshold,
            appearance_threshold=appearance_threshold,
            allowed_classes=allowed_classes,
            device=self.detector.device,
            half=self.detector.half,
        )

        print(
            f"Tracker: {algorithm} (Ultralytics) | "
            f"fps={fps:.0f}, "
            f"activation_threshold={track_activation_threshold}, "
            f"lost_track_buffer={lost_track_buffer}, "
            f"matching_cost_threshold={matching_cost_threshold}"
        )

        crossing_lines = self.config.get("counter.crossing_lines", None)
        self.counter = ObjectCounter(crossing_lines=crossing_lines)
        self.metrics = PerformanceMetrics()

        font_size = self.config.get("display.font_size", 1.0)
        self.renderer = FrameRenderer(font_size)

        if self.config.get("reid.enabled", False):
            self._initialize_reid()

    def _initialize_reid(self) -> None:
        from src.cropper import ObjectCropper
        from src.embedder import ObjectEmbedder
        from src.object_memory import ObjectMemory
        from src.reid import ReIDManager

        emb_cfg_path = self.config.get(
            "reid.embeddings_config", "configs/embeddings/default.yaml"
        )
        emb_cfg = Config(emb_cfg_path)

        cropper = ObjectCropper(
            padding=emb_cfg.get("cropper.padding", 8),
            save_crops=emb_cfg.get("cropper.save_crops", False),
            output_dir=emb_cfg.get("cropper.output_dir", "outputs/crops"),
        )
        embedder = ObjectEmbedder(
            model_name=emb_cfg.get("embedder.model_name", "ViT-B-32"),
            pretrained=emb_cfg.get("embedder.pretrained", "laion2b_s34b_b79k"),
            device=emb_cfg.get("embedder.device", "cpu"),
            normalize=emb_cfg.get("embedder.normalize", True),
        )
        memory = ObjectMemory(
            similarity_threshold=emb_cfg.get("memory.similarity_threshold", 0.75),
            max_missing_frames=emb_cfg.get("memory.max_missing_frames", 90),
            max_embeddings_per_object=emb_cfg.get("memory.max_embeddings_per_object", 5),
            aggregation_method=emb_cfg.get("memory.aggregation_method", "mean"),
            ema_alpha=emb_cfg.get("memory.ema_alpha", 0.3),
            recent_n=emb_cfg.get("memory.recent_n", 3),
            weighted_decay=emb_cfg.get("memory.weighted_decay", 0.7),
        )
        min_track_age = self.config.get("reid.min_track_age", 1)
        self.reid_manager = ReIDManager(cropper, embedder, memory, min_track_age=min_track_age)
        self._reid_interval = self.config.get("reid.update_interval", 1)

        print(
            f"ReID: enabled | model={emb_cfg.get('embedder.model_name', 'ViT-B-32')} "
            f"device={emb_cfg.get('embedder.device', 'cpu')} "
            f"threshold={emb_cfg.get('memory.similarity_threshold', 0.75)} "
            f"update_interval={self._reid_interval} "
            f"min_track_age={min_track_age}"
        )

    def _resolve_lost_track_buffer(self, fps: float) -> int:
        """Resolve lost_track_buffer from config.

        If the config value is null, calculates it as:
            round(fps * auto_lost_track_buffer_seconds)
        """
        explicit = self.config.get_raw("tracker.lost_track_buffer")
        if explicit is not None:
            return int(explicit)

        seconds = self.config.get("tracker.auto_lost_track_buffer_seconds", 3.0)
        return max(1, round(fps * float(seconds)))

    def _get_configured_fps(self) -> float:
        """Resolve FPS for tracker initialisation.

        Priority:
        1. Explicit ``video.fps`` in config.
        2. FPS reported by the video source (file metadata or webcam driver).
        3. Fallback: 30 fps.
        """
        explicit = self.config.get_raw("video.fps")
        if explicit is not None:
            return float(explicit)

        source_fps = self.video_source.get_fps()
        if source_fps and source_fps > 0:
            return float(source_fps)

        return float(self.config.get("video.fallback_fps", 30))

    def run(self) -> None:
        print("Starting object counter. Press 'q' to exit.")
        print(f"Device: {self.config.get('detector.device', 'Not specified')}")

        frame_idx = 0

        try:
            while True:
                frame_start = time.perf_counter()

                success, frame = self.video_source.read()
                if not success:
                    print("End of video stream or camera disconnected.")
                    break

                detect_start = time.perf_counter()
                detections, tracked_objects = self.tracker.update(frame)
                elapsed = time.perf_counter() - detect_start
                self.metrics.record_detection_time(elapsed)
                self.metrics.record_tracking_time(0.0)

                track_to_object_id: Dict[int, int] = {}
                if self.reid_manager is not None:
                    if frame_idx % self._reid_interval == 0:
                        track_to_object_id = self.reid_manager.update(
                            frame, tracked_objects, frame_idx
                        )
                    else:
                        # Reuse last-known object IDs for the current tracks
                        track_to_object_id = {
                            tid: self.reid_manager.get_object_id(tid)
                            for tid in tracked_objects
                            if self.reid_manager.get_object_id(tid) is not None
                        }

                counts = self.counter.update(tracked_objects)

                frame = self._render_frame(
                    frame, detections, tracked_objects, counts, track_to_object_id
                )

                self.metrics.record_frame_time(time.perf_counter() - frame_start)

                cv2.imshow("Object Counter", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_idx += 1

        finally:
            self.cleanup()

    def _render_frame(
        self,
        frame,
        detections,
        tracked_objects,
        counts,
        track_to_object_id: Optional[Dict[int, int]] = None,
    ):
        if self.config.get("display.show_detections", True):
            frame = self.renderer.render_detections(frame, detections)

        reid_active = bool(track_to_object_id) and self.config.get(
            "display.show_object_ids", True
        )
        if self.config.get("display.show_tracking_ids", True):
            frame = self.renderer.render_tracks(
                frame,
                tracked_objects,
                object_ids=track_to_object_id if reid_active else None,
            )

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

        if (
            self.reid_manager is not None
            and self.config.get("display.show_reid_stats", True)
        ):
            frame = self.renderer.render_reid_stats(
                frame,
                unique_total=self.reid_manager.total_object_count(),
                active=self.reid_manager.active_object_count(),
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

    parser.add_argument(
        "--half",
        dest="half",
        action="store_true",
        default=None,
        help="Enable FP16 inference (overrides detector.half). "
        "Only effective for .pt models on CUDA.",
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
        half_override=args.half,
    )
    app.run()


if __name__ == "__main__":
    main()