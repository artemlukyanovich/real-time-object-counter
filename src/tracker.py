"""Object tracking module using Ultralytics built-in ByteTrack / BoT-SORT."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from ultralytics import YOLO


# Detection format shared across the project:
# ((x1, y1, x2, y2), class_name, confidence)
Detection = Tuple[Tuple[int, int, int, int], str, float]

_RUNTIME_DIR = Path(__file__).parent.parent / ".runtime" / "trackers"

_SUPPORTED_ALGORITHMS = ("bytetrack", "botsort", "botsort_reid")

# Default parameters for each tracking algorithm.
# Applied as the base before user-level overrides from the active config.
_ALGORITHM_DEFAULTS: Dict[str, dict] = {
    "bytetrack": {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
    },
    "botsort": {
        "tracker_type": "botsort",
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.5,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "with_reid": False,
    },
    "botsort_reid": {
        "tracker_type": "botsort",
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.5,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "with_reid": True,
        "reid_weights": "osnet_x0_25_market.pt",
    },
}


class UltralyticsTracker:
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
        track_low_threshold: float = 0.1,
        lost_track_buffer: int = 30,
        matching_cost_threshold: float = 0.8,
        fuse_score: bool = True,
        gmc_method: str = "sparseOptFlow",
        reid_weights: str = "osnet_x0_25_market.pt",
        proximity_threshold: float = 0.5,
        appearance_threshold: float = 0.25,
        allowed_classes: Optional[List[str]] = None,
        device: str = "cuda",
    ) -> None:
        """Initialise UltralyticsTracker.

        Args:
            model: Loaded Ultralytics YOLO model (shared with the detector).
            conf_threshold: Detection confidence threshold passed to model.track().
            frame_rate: FPS hint (stored for informational use).
            algorithm: Tracking algorithm — "bytetrack", "botsort", or
                "botsort_reid" (BoT-SORT with appearance-based Re-ID).
            track_activation_threshold: Maps to track_high_thresh and
                new_track_thresh in the tracker YAML.
            track_low_threshold: Maps to track_low_thresh (second-stage
                low-confidence detection matching threshold).
            lost_track_buffer: Maps to track_buffer (frames to keep a lost
                track alive before discarding it).
            matching_cost_threshold: Maps to match_thresh (IoU cost threshold
                for associating detections to tracks). Higher values allow
                matching detections with lower overlap, keeping the same ID
                through partial occlusions. Lower values require tighter
                overlap, causing ID switches when an object is briefly hidden.
            fuse_score: BoT-SORT only. Fuse detection confidence into IoU
                cost matrix.
            gmc_method: BoT-SORT only. Global Motion Compensation method.
            reid_weights: botsort_reid only. Re-ID model weights file.
            proximity_threshold: botsort_reid only. Maps to proximity_thresh.
            appearance_threshold: botsort_reid only. Maps to appearance_thresh.
            allowed_classes: Optional list of class names to detect/track.
                None or empty list = all classes. Unknown names are warned and
                skipped. Example: ["person", "car"].
            device: Inference device passed to model.track() ("cuda" or "cpu").
                Should match the device the shared model was loaded on. For ONNX
                models this selects the ONNX Runtime execution provider; for
                .pt / .engine the model is already bound to its device.
        """
        if algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unknown tracker algorithm '{algorithm}'. "
                f"Supported: {_SUPPORTED_ALGORITHMS}"
            )
        self._model = model
        self._conf = conf_threshold
        self._device = device
        self.algorithm = algorithm
        self._allowed_class_ids = self._resolve_class_ids(allowed_classes)
        self._yaml_path = self._write_tracker_yaml(
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
            classes=self._allowed_class_ids,
            device=self._device,
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

    def _resolve_model_names(self) -> Optional[Dict[int, str]]:
        """Return the model's {class_id: name} mapping across all backends.

        For .pt models ``model.names`` is populated at load time. Exported
        backends (.engine / .onnx) leave ``model.names`` as None until the
        underlying AutoBackend is instantiated — which first happens on the
        initial inference; the names then live on ``model.predictor.model.names``.
        A one-off warmup predict triggers that load.
        """
        names = getattr(self._model, "names", None)
        if names:
            return names

        # Exported backend: force the predictor/AutoBackend to load, then read names.
        self._model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        predictor = getattr(self._model, "predictor", None)
        backend = getattr(predictor, "model", None) if predictor is not None else None
        return getattr(backend, "names", None) if backend is not None else None

    def _resolve_class_ids(
        self, allowed_classes: Optional[List[str]]
    ) -> Optional[List[int]]:
        """Map class names to YOLO class IDs using the loaded model.

        Returns None (= all classes) if allowed_classes is None/empty.
        Unknown names are warned and skipped.
        """
        if not allowed_classes:
            return None

        names = self._resolve_model_names()
        if not names:
            print(
                "Warning: could not read class names from the model; ignoring "
                "allowed_classes (all classes will be detected)."
            )
            return None

        name_to_id = {v: k for k, v in names.items()}
        ids = []
        for name in allowed_classes:
            if name in name_to_id:
                ids.append(name_to_id[name])
            else:
                print(
                    f"Warning: allowed_classes: '{name}' not found in model. "
                    f"Available: {sorted(name_to_id)}"
                )

        return ids if ids else None

    def _write_tracker_yaml(
        self,
        algorithm: str,
        track_activation_threshold: float,
        track_low_threshold: float,
        lost_track_buffer: int,
        matching_cost_threshold: float,
        fuse_score: bool,
        gmc_method: str,
        reid_weights: str,
        proximity_threshold: float,
        appearance_threshold: float,
    ) -> str:
        """Build a tracker YAML for Ultralytics and write it to .runtime/trackers/.

        Starts from the algorithm defaults (_ALGORITHM_DEFAULTS), then applies
        all user-facing overrides from the active config file.

        Parameter mapping:
            track_activation_threshold → track_high_thresh, new_track_thresh
            track_low_threshold        → track_low_thresh
            lost_track_buffer          → track_buffer
            matching_cost_threshold    → match_thresh
            fuse_score                 → fuse_score        (botsort, botsort_reid)
            gmc_method                 → gmc_method        (botsort, botsort_reid)
            proximity_threshold        → proximity_thresh  (botsort, botsort_reid)
            appearance_threshold       → appearance_thresh (botsort, botsort_reid)
            reid_weights               → reid_weights      (botsort_reid)
        """
        params = dict(_ALGORITHM_DEFAULTS[algorithm])

        params["track_high_thresh"] = track_activation_threshold
        params["new_track_thresh"] = track_activation_threshold
        params["track_low_thresh"] = track_low_threshold
        params["track_buffer"] = lost_track_buffer
        params["match_thresh"] = matching_cost_threshold

        if algorithm in ("botsort", "botsort_reid"):
            params["fuse_score"] = fuse_score
            params["gmc_method"] = gmc_method
            params["proximity_thresh"] = proximity_threshold
            params["appearance_thresh"] = appearance_threshold

        if algorithm == "botsort_reid":
            params["reid_weights"] = reid_weights

        _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        runtime_path = _RUNTIME_DIR / f"{algorithm}.yaml"
        with open(runtime_path, "w") as fh:
            yaml.dump(params, fh, default_flow_style=False)

        return str(runtime_path)
