"""Object tracking module using ByteTrack (via the supervision library)."""

import numpy as np
import supervision as sv
from typing import Dict, List, Tuple


# Re-export the Detection type alias used by detector.py so imports stay consistent.
Detection = Tuple[Tuple[int, int, int, int], str, float]


class ByteTracker:
    """Wrap supervision's ByteTracker to match the project's tracker interface.

    Input:  List[((x1, y1, x2, y2), class_name, confidence)]
    Output: Dict[track_id, ((x1, y1, x2, y2), class_name)]
    """

    def __init__(
        self,
        frame_rate: int = 30,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ) -> None:
        """Initialise ByteTracker.

        Args:
            frame_rate: Video frame rate — used to convert the lost_track_buffer
                (in frames) into a time window for track re-identification.
            track_activation_threshold: Minimum detection confidence required to
                activate a new track.
            lost_track_buffer: Number of frames to keep a lost track alive before
                discarding it.
            minimum_matching_threshold: IoU threshold for matching detections to
                existing tracks.
        """
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

        # Bidirectional mapping between string class names and integer IDs
        # required by supervision's Detections.
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}
        self._next_class_id: int = 0

    # ------------------------------------------------------------------
    # Public interface (matches the old CentroidTracker signature)
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
    ) -> Dict[int, Tuple[Tuple[int, int, int, int], str]]:
        """Update tracker with detections from the current frame.

        Args:
            detections: List of (bbox, class_name, confidence).

        Returns:
            Dictionary of {track_id: (bbox, class_name)} for all active tracks.
        """
        sv_detections = self._to_sv_detections(detections)
        tracked = self._tracker.update_with_detections(sv_detections)

        if tracked.tracker_id is None or len(tracked) == 0:
            return {}

        result: Dict[int, Tuple[Tuple[int, int, int, int], str]] = {}
        for i, track_id in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = (int(v) for v in tracked.xyxy[i])
            class_name = self._id_to_name.get(int(tracked.class_id[i]), "unknown")
            result[int(track_id)] = ((x1, y1, x2, y2), class_name)

        return result

    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_class_id(self, class_name: str) -> int:
        if class_name not in self._name_to_id:
            self._name_to_id[class_name] = self._next_class_id
            self._id_to_name[self._next_class_id] = class_name
            self._next_class_id += 1
        return self._name_to_id[class_name]

    def _to_sv_detections(self, detections: List[Detection]) -> sv.Detections:
        if not detections:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty(0, dtype=np.float32),
                class_id=np.empty(0, dtype=np.int_),
            )

        xyxy = np.array(
            [[b[0], b[1], b[2], b[3]] for b, _, _ in detections],
            dtype=np.float32,
        )
        confidences = np.array([c for _, _, c in detections], dtype=np.float32)
        class_ids = np.array(
            [self._get_class_id(cn) for _, cn, _ in detections],
            dtype=np.int_,
        )

        return sv.Detections(xyxy=xyxy, confidence=confidences, class_id=class_ids)
