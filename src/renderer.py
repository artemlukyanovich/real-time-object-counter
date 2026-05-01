"""Rendering module for visualization."""

from typing import Dict, List, Tuple

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
    ) -> np.ndarray:
        """Render tracked objects on frame.

        Args:
            frame: Input frame.
            tracked_objects: Dictionary of {track_id: (bbox, class_name)}.
            show_ids: Whether to show track IDs.

        Returns:
            Frame with rendered tracks.
        """
        frame = frame.copy()

        for track_id, (bbox, class_name) in tracked_objects.items():
            x1, y1, x2, y2 = bbox
            color = self.colors[track_id % len(self.colors)]

            # Draw tracking bounding box.
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID {track_id}: {class_name}" if show_ids else class_name
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

    def render_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Render FPS metric on frame.

        Args:
            frame: Input frame.
            fps: Current frames per second.

        Returns:
            Frame with rendered FPS.
        """
        frame = frame.copy()
        h, w = frame.shape[:2]

        text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(text, self.font, self.font_size, 1)[0]

        x1 = w - text_size[0] - 24
        y1 = 10
        x2 = w - 10
        y2 = 42

        # Draw dark FPS background for stable readability.
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.panel_bg_color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 8, 32),
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