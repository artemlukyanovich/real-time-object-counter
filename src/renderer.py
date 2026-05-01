"""Rendering module for visualization."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class FrameRenderer:
    """Render detection and tracking results on frames."""

    def __init__(self, font_size: float = 1.0):
        """Initialize renderer.

        Args:
            font_size: Font size multiplier
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_size = font_size
        self.colors = self._generate_colors()

    def render_detections(self, frame: np.ndarray,
                         detections: List[Tuple[Tuple[int, int, int, int], str, float]],
                         show_confidence: bool = True) -> np.ndarray:
        """Render detections on frame.

        Args:
            frame: Input frame
            detections: List of (bbox, class_name, confidence) tuples
            show_confidence: Whether to show confidence scores

        Returns:
            Rendered frame
        """
        frame = frame.copy()

        for bbox, class_name, confidence in detections:
            x1, y1, x2, y2 = bbox
            color = self.colors[hash(class_name) % len(self.colors)]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}"
            if show_confidence:
                label += f" {confidence:.2f}"

            label_size = cv2.getTextSize(label, self.font, self.font_size, 1)[0]
            y_label = max(y1 - 5, label_size[1] + 5)
            cv2.rectangle(frame, (x1, y_label - label_size[1] - 5),
                         (x1 + label_size[0], y_label), color, -1)
            cv2.putText(frame, label, (x1, y_label), self.font,
                       self.font_size, (255, 255, 255), 1)

        return frame

    def render_tracks(self, frame: np.ndarray,
                     tracked_objects: Dict[int, Tuple[Tuple[int, int, int, int], str]],
                     show_ids: bool = True) -> np.ndarray:
        """Render tracked objects on frame.

        Args:
            frame: Input frame
            tracked_objects: Dictionary of {track_id: (bbox, class_name)}
            show_ids: Whether to show track IDs

        Returns:
            Rendered frame
        """
        frame = frame.copy()

        for track_id, (bbox, class_name) in tracked_objects.items():
            x1, y1, x2, y2 = bbox
            color = self.colors[track_id % len(self.colors)]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with ID
            label = f"ID: {track_id}" if show_ids else class_name
            label_size = cv2.getTextSize(label, self.font, self.font_size, 1)[0]
            y_label = max(y1 - 5, label_size[1] + 5)
            cv2.rectangle(frame, (x1, y_label - label_size[1] - 5),
                         (x1 + label_size[0], y_label), color, -1)
            cv2.putText(frame, label, (x1, y_label), self.font,
                       self.font_size, (255, 255, 255), 1)

        return frame

    def render_counts(self, frame: np.ndarray,
                     counts: Dict[str, int],
                     total_count: int) -> np.ndarray:
        """Render count information on frame.

        Args:
            frame: Input frame
            counts: Dictionary of counts by class
            total_count: Total count

        Returns:
            Rendered frame
        """
        frame = frame.copy()
        h, w = frame.shape[:2]

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100 + len(counts) * 25), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Draw text
        y = 35
        cv2.putText(frame, f"Total: {total_count}", (20, y), self.font,
                   self.font_size * 1.5, (0, 255, 0), 2)

        for class_name, count in counts.items():
            y += 30
            cv2.putText(frame, f"{class_name}: {count}", (20, y), self.font,
                       self.font_size, (0, 255, 0), 1)

        return frame

    def render_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Render FPS on frame.

        Args:
            frame: Input frame
            fps: Frames per second

        Returns:
            Rendered frame
        """
        frame = frame.copy()
        h, w = frame.shape[:2]

        text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(text, self.font, self.font_size, 1)[0]
        cv2.rectangle(frame, (w - text_size[0] - 20, 10),
                     (w - 10, 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (w - text_size[0] - 15, 30), self.font,
                   self.font_size, (0, 255, 0), 1)

        return frame

    @staticmethod
    def _generate_colors(n: int = 256) -> List[Tuple[int, int, int]]:
        """Generate random colors."""
        colors = []
        for i in range(n):
            h = (i * 180 // n) % 180
            s = 200
            v = 255
            hsv = np.uint8([[[h, s, v]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in bgr))
        return colors
