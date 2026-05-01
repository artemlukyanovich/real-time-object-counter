"""Video source handling module."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class VideoSource:
    """Handle video input from webcam or file."""

    def __init__(self, source: int | str = 0, width: int = 1280, height: int = 720):
        """Initialize video source.

        Args:
            source: Camera index (0 for default) or path to video file
            width: Frame width
            height: Frame height
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.fps = 30
        self.frame_count = 0
        self.total_frames = 0
        self.is_video_file = False

        self._open()

    def _open(self) -> None:
        """Open video source."""
        if isinstance(self.source, str) and Path(self.source).exists():
            self.is_video_file = True
            self.cap = cv2.VideoCapture(str(self.source))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.cap = cv2.VideoCapture(self.source)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from source."""
        ret, frame = self.cap.read()

        if ret:
            self.frame_count += 1
            frame = cv2.resize(frame, (self.width, self.height))

        return ret, frame

    def release(self) -> None:
        """Release video source."""
        if self.cap:
            self.cap.release()

    def get_fps(self) -> float:
        """Get frames per second."""
        return self.fps

    def get_frame_count(self) -> int:
        """Get current frame number."""
        return self.frame_count

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
