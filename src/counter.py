"""Object counting module."""

from typing import Dict, Optional, List, Tuple, Set
from src.utils import is_point_in_polygon, get_centroid
from collections import defaultdict


class ObjectCounter:
    """Count objects based on tracking data."""

    def __init__(self, count_zone: Optional[List[Tuple[int, int]]] = None):
        """Initialize counter.

        Args:
            count_zone: Optional polygon defining counting zone
        """
        self.count_zone = count_zone
        self.total_count = 0
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.counted_ids: Set[int] = set()
        self.frame_counts: Dict[int, int] = defaultdict(int)

    def update(self, tracked_objects: Dict[int, Tuple[Tuple[int, int, int, int], str]]
               ) -> Dict[str, int]:
        """Update counter with tracked objects.

        Args:
            tracked_objects: Dictionary of {track_id: (bbox, class_name)}

        Returns:
            Current counts by class
        """
        current_ids = set()

        for track_id, (bbox, class_name) in tracked_objects.items():
            current_ids.add(track_id)

            # Check if object is in counting zone
            if self.count_zone:
                centroid = get_centroid(bbox)
                if not is_point_in_polygon(centroid, self.count_zone):
                    continue

            # Count if not already counted
            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                self.total_count += 1
                self.class_counts[class_name] += 1

            self.frame_counts[track_id] += 1

        return dict(self.class_counts)

    def get_counts(self) -> Dict[str, int]:
        """Get current counts by class."""
        return dict(self.class_counts)

    def get_total_count(self) -> int:
        """Get total count."""
        return self.total_count

    def get_class_count(self, class_name: str) -> int:
        """Get count for specific class."""
        return self.class_counts.get(class_name, 0)

    def reset(self) -> None:
        """Reset counter."""
        self.total_count = 0
        self.class_counts.clear()
        self.counted_ids.clear()
        self.frame_counts.clear()

    def set_count_zone(self, zone: List[Tuple[int, int]]) -> None:
        """Set counting zone polygon."""
        self.count_zone = zone
