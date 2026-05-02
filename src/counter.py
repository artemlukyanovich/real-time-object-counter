"""Object counting module."""

from typing import Any, Dict, Optional, List, Tuple, Set
from src.utils import is_point_in_polygon, get_centroid, point_side_of_line
from collections import defaultdict


class ObjectCounter:
    """Count objects based on tracking data.

    Supports two counting modes (can be combined):
    - Zone-based: counts an object once when its centroid enters the count_zone polygon.
    - Line-crossing: counts each time an object's centroid crosses a defined line,
      separately for each direction ("in" / "out").
    """

    def __init__(
        self,
        count_zone: Optional[List[Tuple[int, int]]] = None,
        crossing_lines: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize counter.

        Args:
            count_zone: Optional polygon defining the counting zone.
            crossing_lines: Optional list of line definitions, each a dict with:
                - "name" (str): human-readable label for the line.
                - "points" (list): [[x1, y1], [x2, y2]] endpoints of the line.
        """
        self.count_zone = count_zone
        self.total_count = 0
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.counted_ids: Set[int] = set()
        self.frame_counts: Dict[int, int] = defaultdict(int)

        # Line-crossing state
        self._line_defs: List[Dict[str, Any]] = crossing_lines or []
        # track_id -> {line_name: last_side_sign}  (+1 left, -1 right)
        self._prev_sides: Dict[int, Dict[str, int]] = {}
        # line_name -> {"in": {class_name: count}, "out": {class_name: count}}
        self._line_counts: Dict[str, Dict[str, Dict[str, int]]] = {
            line["name"]: {"in": defaultdict(int), "out": defaultdict(int)}
            for line in self._line_defs
        }

    def update(
        self,
        tracked_objects: Dict[int, Tuple[Tuple[int, int, int, int], str]],
    ) -> Dict[str, int]:
        """Update counter with tracked objects.

        Args:
            tracked_objects: Dictionary of {track_id: (bbox, class_name)}.

        Returns:
            Current zone-based counts by class.
        """
        current_ids: Set[int] = set()

        for track_id, (bbox, class_name) in tracked_objects.items():
            current_ids.add(track_id)
            centroid = get_centroid(bbox)

            # Line-crossing is always evaluated, independent of zone.
            self._update_line_crossing(track_id, centroid, class_name)

            # Zone-based counting (skip objects outside zone when zone is set).
            if self.count_zone and not is_point_in_polygon(centroid, self.count_zone):
                continue

            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                self.total_count += 1
                self.class_counts[class_name] += 1

            self.frame_counts[track_id] += 1

        # Remove crossing state for tracks that have disappeared.
        for track_id in set(self._prev_sides) - current_ids:
            del self._prev_sides[track_id]

        return dict(self.class_counts)

    def _update_line_crossing(
        self,
        track_id: int,
        centroid: Tuple[int, int],
        class_name: str,
    ) -> None:
        if not self._line_defs:
            return

        if track_id not in self._prev_sides:
            self._prev_sides[track_id] = {}

        for line in self._line_defs:
            name = line["name"]
            p1, p2 = line["points"][0], line["points"][1]

            raw = point_side_of_line(centroid, p1, p2)
            curr_sign = 1 if raw > 0 else (-1 if raw < 0 else 0)

            if curr_sign == 0:
                # Centroid is exactly on the line – don't update state.
                continue

            prev_sign = self._prev_sides[track_id].get(name, 0)

            if prev_sign != 0 and prev_sign != curr_sign:
                # Side changed – line was crossed.
                # "in"  = moved from left (+) to right (-)
                # "out" = moved from right (-) to left (+)
                direction = "in" if prev_sign > 0 else "out"
                self._line_counts[name][direction][class_name] += 1

            self._prev_sides[track_id][name] = curr_sign

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_counts(self) -> Dict[str, int]:
        """Get current zone-based counts by class."""
        return dict(self.class_counts)

    def get_total_count(self) -> int:
        """Get total zone-based count."""
        return self.total_count

    def get_class_count(self, class_name: str) -> int:
        """Get zone-based count for a specific class."""
        return self.class_counts.get(class_name, 0)

    def get_line_counts(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Return line-crossing counts.

        Returns:
            {line_name: {"in": {class_name: count}, "out": {class_name: count}}}
        """
        return {
            name: {
                "in": dict(dirs["in"]),
                "out": dict(dirs["out"]),
            }
            for name, dirs in self._line_counts.items()
        }

    def has_crossing_lines(self) -> bool:
        """Return True if any crossing lines are configured."""
        return bool(self._line_defs)

    def get_crossing_line_defs(self) -> List[Dict[str, Any]]:
        """Return raw line definitions (for rendering)."""
        return self._line_defs

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all counters and state."""
        self.total_count = 0
        self.class_counts.clear()
        self.counted_ids.clear()
        self.frame_counts.clear()
        self._prev_sides.clear()
        for dirs in self._line_counts.values():
            dirs["in"].clear()
            dirs["out"].clear()

    def set_count_zone(self, zone: List[Tuple[int, int]]) -> None:
        """Set counting zone polygon."""
        self.count_zone = zone
