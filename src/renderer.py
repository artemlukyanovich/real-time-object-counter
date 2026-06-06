"""Rendering module for visualization."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


Detection = Tuple[Tuple[int, int, int, int], str, float]
TrackedObjects = Dict[int, Tuple[Tuple[int, int, int, int], str]]


class FrameRenderer:
    """Render detection, tracking, counting and performance data on frames."""

    def __init__(self, font_size: float = 0.7):
        """Initialize renderer.

        Args:
            font_size: Font size multiplier for OpenCV text rendering.
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_size = font_size
        self.colors = self._generate_colors()

        # Stable UI colors in BGR format.
        self.text_color = (255, 255, 255)
        self.dark_text_color = (0, 0, 0)
        self.panel_bg_color = (0, 0, 0)
        self.metric_color = (0, 255, 0)

    def render_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Render raw detections on frame.

        Args:
            frame: Input frame.
            detections: List of ((x1, y1, x2, y2), class_name, confidence).
            show_confidence: Whether confidence score should be shown.

        Returns:
            Frame with rendered detections.
        """
        frame = frame.copy()

        for bbox, class_name, confidence in detections:
            x1, y1, x2, y2 = bbox
            color = self.colors[hash(class_name) % len(self.colors)]

            # Draw detection bounding box.
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = class_name
            if show_confidence:
                label += f" {confidence:.2f}"

            self._draw_label(frame, label, x1, y1, bg_color=color)

        return frame

    def render_tracks(
        self,
        frame: np.ndarray,
        tracked_objects: TrackedObjects,
        show_ids: bool = True,
        object_ids: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """Render tracked objects on frame.

        Args:
            frame: Input frame.
            tracked_objects: Dictionary of {track_id: (bbox, class_name)}.
            show_ids: Whether to show track IDs.
            object_ids: Optional mapping {track_id: object_id} from ReIDManager.
                When provided, box colour is keyed by object_id (stable across
                tracker resets) and the label shows the persistent identity.
                Label format: "#N class [tM]" where N=object_id, M=track_id.

        Returns:
            Frame with rendered tracks.
        """
        frame = frame.copy()

        for track_id, (bbox, class_name) in tracked_objects.items():
            x1, y1, x2, y2 = bbox

            if object_ids is not None:
                obj_id = object_ids.get(track_id)
                if obj_id is not None:
                    color = self.colors[obj_id % len(self.colors)]
                    label = f"#{obj_id} {class_name} [t{track_id}]"
                else:
                    # track_id not yet resolved — fall back to temporary display
                    color = self.colors[track_id % len(self.colors)]
                    label = f"t{track_id} {class_name}"
            else:
                color = self.colors[track_id % len(self.colors)]
                label = f"ID {track_id}: {class_name}" if show_ids else class_name

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            self._draw_label(frame, label, x1, y1, bg_color=color)

        return frame

    def render_counts(
        self,
        frame: np.ndarray,
        counts: Dict[str, int],
        total_count: int,
    ) -> np.ndarray:
        """Render object counts on frame.

        Args:
            frame: Input frame.
            counts: Dictionary of counts by class.
            total_count: Total number of counted objects.

        Returns:
            Frame with rendered count panel.
        """
        frame = frame.copy()

        panel_height = 70 + len(counts) * 28

        # Draw semi-transparent black panel for readability.
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, panel_height), self.panel_bg_color, -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        y = 40
        cv2.putText(
            frame,
            f"Total: {total_count}",
            (20, y),
            self.font,
            self.font_size * 1.2,
            self.metric_color,
            2,
            cv2.LINE_AA,
        )

        for class_name, count in counts.items():
            y += 28
            cv2.putText(
                frame,
                f"{class_name}: {count}",
                (20, y),
                self.font,
                self.font_size,
                self.text_color,
                1,
                cv2.LINE_AA,
            )

        return frame

    def render_crossing_lines(
        self,
        frame: np.ndarray,
        crossing_lines: List[Dict],
    ) -> np.ndarray:
        """Draw crossing lines and their name labels on the frame.

        Args:
            frame: Input frame.
            crossing_lines: List of line defs, each {"name": str, "points": [[x1,y1],[x2,y2]]}.

        Returns:
            Frame with crossing lines drawn.
        """
        frame = frame.copy()
        line_color = (0, 220, 220)  # Cyan-ish

        for line_def in crossing_lines:
            name = line_def["name"]
            p1 = tuple(line_def["points"][0])
            p2 = tuple(line_def["points"][1])

            cv2.line(frame, p1, p2, line_color, 2, cv2.LINE_AA)

            # Label near the midpoint, offset slightly so it doesn't sit on the line.
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.putText(
                frame,
                name,
                (mid_x + 8, mid_y - 8),
                self.font,
                self.font_size,
                line_color,
                1,
                cv2.LINE_AA,
            )

        return frame

    def render_line_counts(
        self,
        frame: np.ndarray,
        line_counts: Dict,
        y_start: int = 10,
    ) -> np.ndarray:
        """Render a panel showing line-crossing counts.

        Args:
            frame: Input frame.
            line_counts: {line_name: {"in": {class: n}, "out": {class: n}}}.
            y_start: Top y-coordinate of the panel (allows stacking below zone count panel).

        Returns:
            Frame with count panel rendered.
        """
        if not line_counts:
            return frame

        frame = frame.copy()
        line_color = (0, 220, 220)

        # Compute how many text rows we need.
        row_count = 0
        for name, dirs in line_counts.items():
            row_count += 1  # line name header
            for direction in ("in", "out"):
                row_count += 1  # direction total
                row_count += len(dirs[direction])  # per-class breakdown

        row_h = 24
        panel_h = 12 + row_count * row_h + 8
        panel_x2 = 320

        overlay = frame.copy()
        cv2.rectangle(
            overlay, (10, y_start), (panel_x2, y_start + panel_h),
            self.panel_bg_color, -1,
        )
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        y = y_start + row_h

        for name, dirs in line_counts.items():
            cv2.putText(
                frame, f"[{name}]", (20, y),
                self.font, self.font_size, line_color, 1, cv2.LINE_AA,
            )
            y += row_h

            for direction, arrow in (("in", "\u2193"), ("out", "\u2191")):
                class_counts = dirs[direction]
                total = sum(class_counts.values())
                cv2.putText(
                    frame,
                    f"  {arrow} {direction}: {total}",
                    (20, y),
                    self.font,
                    self.font_size,
                    self.text_color,
                    1,
                    cv2.LINE_AA,
                )
                y += row_h

                for class_name, count in class_counts.items():
                    cv2.putText(
                        frame,
                        f"      {class_name}: {count}",
                        (20, y),
                        self.font,
                        self.font_size * 0.85,
                        self.text_color,
                        1,
                        cv2.LINE_AA,
                    )
                    y += row_h

        return frame

    def render_reid_stats(
        self,
        frame: np.ndarray,
        unique_total: int,
        active: int,
    ) -> np.ndarray:
        """Render a small Re-ID statistics panel below the FPS counter.

        Shows the total number of unique objects seen and how many are
        currently active, making it easy to verify that persistent IDs
        survive re-entries without inflating the count.

        Args:
            frame: Input frame.
            unique_total: Total unique objects registered since session start.
            active: Number of currently active objects in memory.

        Returns:
            Frame with ReID stats panel rendered.
        """
        frame = frame.copy()
        h, w = frame.shape[:2]

        lines = [
            f"ReID unique: {unique_total}",
            f"ReID active: {active}",
        ]

        line_h = 22
        padding_x = 8
        panel_h = len(lines) * line_h + 10

        # Measure widest line to size the panel.
        max_w = max(
            cv2.getTextSize(t, self.font, self.font_size * 0.75, 1)[0][0]
            for t in lines
        )
        panel_w = max_w + padding_x * 2

        # Position: top-right corner, below the FPS box (which may be two lines
        # tall when frame skipping shows both throughput and inference FPS).
        x1 = w - panel_w - 10
        y1 = 84
        x2 = w - 10
        y2 = y1 + panel_h

        cv2.rectangle(frame, (x1, y1), (x2, y2), self.panel_bg_color, -1)

        for i, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x1 + padding_x, y1 + (i + 1) * line_h),
                self.font,
                self.font_size * 0.75,
                self.metric_color,
                1,
                cv2.LINE_AA,
            )

        return frame

    def render_fps(
        self,
        frame: np.ndarray,
        fps: float,
        inference_fps: Optional[float] = None,
    ) -> np.ndarray:
        """Render FPS metric on frame.

        Args:
            frame: Input frame.
            fps: Current throughput (display) frames per second.
            inference_fps: Optional raw model inference rate. When provided (i.e.
                frame skipping is active), it is shown on a second line so the
                display rate and the true model cost can be read separately.

        Returns:
            Frame with rendered FPS.
        """
        frame = frame.copy()
        h, w = frame.shape[:2]

        lines = [f"FPS: {fps:.1f}"]
        if inference_fps is not None:
            lines.append(f"infer: {inference_fps:.1f}")

        line_h = 32
        max_w = max(
            cv2.getTextSize(t, self.font, self.font_size, 1)[0][0] for t in lines
        )

        x1 = w - max_w - 24
        y1 = 10
        x2 = w - 10
        y2 = y1 + line_h * len(lines)

        # Draw dark FPS background for stable readability.
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.panel_bg_color, -1)
        for i, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (x1 + 8, 32 + i * line_h),
                self.font,
                self.font_size,
                self.metric_color,
                1,
                cv2.LINE_AA,
            )

        return frame

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        bg_color: Tuple[int, int, int],
    ) -> None:
        """Draw readable label with colored background.

        Args:
            frame: Frame to draw on.
            text: Label text.
            x: Left label coordinate.
            y: Top object coordinate.
            bg_color: Label background color in BGR format.
        """
        text_size, baseline = cv2.getTextSize(text, self.font, self.font_size, 1)
        text_width, text_height = text_size

        # Keep label inside the frame if bbox is close to the top edge.
        label_y = max(y, text_height + 8)

        cv2.rectangle(
            frame,
            (x, label_y - text_height - 8),
            (x + text_width + 8, label_y),
            bg_color,
            -1,
        )

        # Use black text on colored labels to avoid white-on-yellow readability issues.
        cv2.putText(
            frame,
            text,
            (x + 4, label_y - 5),
            self.font,
            self.font_size,
            self.dark_text_color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _generate_colors(n: int = 256) -> List[Tuple[int, int, int]]:
        """Generate deterministic BGR colors for classes/tracks."""
        colors = []

        for i in range(n):
            hue = (i * 180 // n) % 180
            saturation = 200
            value = 255

            hsv = np.uint8([[[hue, saturation, value]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in bgr))

        return colors