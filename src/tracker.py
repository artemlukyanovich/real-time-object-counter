"""Object tracking module."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from src.utils import get_centroid, distance
from collections import defaultdict


class CentroidTracker:
    """Track objects using centroid matching."""

    def __init__(self, max_disappeared: int = 50, max_distance: float = 50):
        """Initialize tracker.

        Args:
            max_disappeared: Max frames an object can disappear before removal
            max_distance: Max distance for centroid matching
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        self.objects: Dict[int, Tuple[int, int]] = {}
        self.disappeared: Dict[int, int] = defaultdict(int)
        self.class_names: Dict[int, str] = {}

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], str, float]]
               ) -> Dict[int, Tuple[Tuple[int, int, int, int], str]]:
        """Update tracker with new detections.

        Args:
            detections: List of (bbox, class_name, confidence) tuples

        Returns:
            Dictionary of {track_id: (bbox, class_name)}
        """
        if len(detections) == 0:
            # Mark all as disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._remove_track(track_id)
            return {}

        # Extract centroids from detections
        centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (bbox, _, _) in enumerate(detections):
            centroids[i] = get_centroid(bbox)

        # Match detections to existing objects
        matched_objects = {}
        used_detections = set()

        # Try to match existing objects to detections
        for track_id, centroid in self.objects.items():
            distances = [distance(centroid, c) for c in centroids]
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]

            if min_dist < self.max_distance and min_dist_idx not in used_detections:
                matched_objects[track_id] = detections[min_dist_idx]
                used_detections.add(min_dist_idx)
                self.disappeared[track_id] = 0
                self.objects[track_id] = tuple(centroids[min_dist_idx])

        # Register new detections
        for i, (bbox, class_name, _) in enumerate(detections):
            if i not in used_detections:
                self.objects[self.next_id] = tuple(centroids[i])
                self.class_names[self.next_id] = class_name
                matched_objects[self.next_id] = (bbox, class_name)
                self.next_id += 1

        # Handle disappeared objects
        for track_id in list(self.disappeared.keys()):
            if track_id not in matched_objects:
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._remove_track(track_id)

        return matched_objects

    def _remove_track(self, track_id: int) -> None:
        """Remove a track."""
        if track_id in self.objects:
            del self.objects[track_id]
        if track_id in self.disappeared:
            del self.disappeared[track_id]
        if track_id in self.class_names:
            del self.class_names[track_id]

    def get_tracks(self) -> Dict[int, Tuple[int, int]]:
        """Get current tracks as {track_id: centroid}."""
        return self.objects.copy()

    def reset(self) -> None:
        """Reset tracker."""
        self.objects.clear()
        self.disappeared.clear()
        self.class_names.clear()
