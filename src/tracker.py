"""Object tracking module using Ultralytics built-in ByteTrack / BoT-SORT."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from ultralytics import YOLO


# Detection format shared across the project:
# ((x1, y1, x2, y2), class_name, confidence)
Detection = Tuple[Tuple[int, int, int, int], str, float]

_CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "trackers"
_RUNTIME_DIR = Path(__file__).parent.parent / ".runtime" / "trackers"

_SUPPORTED_ALGORITHMS = ("bytetrack", "botsort")


class ByteTracker:
    """Ultralytics tracker wrapper (ByteTrack or BoT-SORT).

    Uses model.track() (the official Ultralytics tracking API) so that
    detection and tracking run in a single inference pass.

    update() input:  BGR frame (numpy array)
    update() output: (detections, tracked_objects)
        detections:      List[((x1,y1,x2,y2), class_name, confidence)]
        tracked_objects: Dict[track_id, ((x1,y1,x2,y2), class_name)]
    """

    def __init__(
        self,
        model: YOLO,
        conf_threshold: float = 0.5,
        frame_rate: int = 30,
        algorithm: str = "bytetrack",
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ) -> None:
        """Initialise ByteTracker.

        Args:
            model: Loaded Ultralytics YOLO model (shared with the detector).
            conf_threshold: Detection confidence threshold passed to model.track().
            frame_rate: FPS hint (stored for informational use).
            algorithm: Tracking algorithm — "bytetrack" or "botsort".
            track_activation_threshold: Maps to track_high_thresh and
                new_track_thresh in the tracker YAML.
            lost_track_buffer: Maps to track_buffer (frames to keep a lost
                track alive before discarding it).
            minimum_matching_threshold: Maps to match_thresh (IoU threshold
                for associating detections to tracks).
        """
        if algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unknown tracker algorithm '{algorithm}'. "
                f"Supported: {_SUPPORTED_ALGORITHMS}"
            )
        self._model = model
        self._conf = conf_threshold
        self.algorithm = algorithm
        self._yaml_path = self._write_tracker_yaml(
            algorithm=algorithm,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )

    # ── public interface ─────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[Detection], Dict[int, Tuple[Tuple[int, int, int, int], str]]]:
        """Run detection + tracking on a single frame.

        Args:
            frame: Current BGR frame.

        Returns:
            Tuple of:
            - detections: all detected boxes (used for rendering raw bboxes).
            - tracked_objects: {track_id: (bbox, class_name)} for confirmed tracks.
        """
        results = self._model.track(
            frame,
            persist=True,
            tracker=self._yaml_path,
            conf=self._conf,
            verbose=False,
        )

        detections: List[Detection] = []
        tracked_objects: Dict[int, Tuple[Tuple[int, int, int, int], str]] = {}

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].int().tolist()
                conf = float(boxes.conf[i])
                class_name = result.names[int(boxes.cls[i])]
                detections.append(((x1, y1, x2, y2), class_name, conf))

                if boxes.id is not None:
                    track_id = int(boxes.id[i])
                    tracked_objects[track_id] = ((x1, y1, x2, y2), class_name)

        return detections, tracked_objects

    def reset(self) -> None:
        """Reset tracker state (clears all active tracks)."""
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            self._model.predictor.trackers = {}

    # ── internal ─────────────────────────────────────────────────────

    def _write_tracker_yaml(
        self,
        algorithm: str,
        track_activation_threshold: float,
        lost_track_buffer: int,
        minimum_matching_threshold: float,
    ) -> str:
        """Merge user params onto the configs/trackers template and write to .runtime/.

        Loads configs/trackers/{algorithm}.yaml as the base (carries all
        algorithm-specific defaults), applies the three user-facing parameter
        overrides, then writes the result to .runtime/trackers/{algorithm}.yaml.

        Parameter mapping (same for every algorithm):
            track_activation_threshold → track_high_thresh, new_track_thresh
            lost_track_buffer          → track_buffer
            minimum_matching_threshold → match_thresh
        """
        template_path = _CONFIGS_DIR / f"{algorithm}.yaml"
        with open(template_path) as fh:
            params = yaml.safe_load(fh)

        params["track_high_thresh"] = track_activation_threshold
        params["new_track_thresh"] = track_activation_threshold
        params["track_buffer"] = lost_track_buffer
        params["match_thresh"] = minimum_matching_threshold

        _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        runtime_path = _RUNTIME_DIR / f"{algorithm}.yaml"
        with open(runtime_path, "w") as fh:
            yaml.dump(params, fh, default_flow_style=False)

        return str(runtime_path)
