"""Object crop extraction from video frames."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


class ObjectCropper:
    """Extracts crop images of detected/tracked objects from video frames."""

    def __init__(
        self,
        padding: int = 0,
        save_crops: bool = False,
        output_dir: str = "outputs/crops",
    ) -> None:
        self.padding = padding
        self.save_crops = save_crops
        self.output_dir = Path(output_dir)
        if save_crops:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_counter = 0

    def crop(
        self,
        frame: np.ndarray,
        bbox: BBox,
        track_id: Optional[int] = None,
        frame_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Extract a crop from frame given a bounding box.

        Args:
            frame: BGR image (H, W, 3).
            bbox: (x1, y1, x2, y2) in pixel coordinates.
            track_id: Optional track ID used for naming saved crops.
            frame_idx: Optional frame index used for naming saved crops.

        Returns:
            Cropped BGR image as numpy array. Empty array if bbox is invalid.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(w, x2 + self.padding)
        y2 = min(h, y2 + self.padding)

        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)

        crop = frame[y1:y2, x1:x2].copy()

        if self.save_crops and crop.size > 0:
            self._save_crop(crop, track_id, frame_idx)

        return crop

    def crop_all(
        self,
        frame: np.ndarray,
        tracked_objects: Dict[int, Tuple[BBox, str]],
        frame_idx: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        """Extract crops for all tracked objects.

        Args:
            frame: BGR image (H, W, 3).
            tracked_objects: Dict[track_id, ((x1,y1,x2,y2), class_name)]
            frame_idx: Optional frame index.

        Returns:
            Dict[track_id, crop_image] — only includes tracks with non-empty crops.
        """
        crops = {}
        for track_id, (bbox, _) in tracked_objects.items():
            crop = self.crop(frame, bbox, track_id=track_id, frame_idx=frame_idx)
            if crop.size > 0:
                crops[track_id] = crop
        return crops

    def _save_crop(
        self,
        crop: np.ndarray,
        track_id: Optional[int],
        frame_idx: Optional[int],
    ) -> None:
        frame_prefix = f"frame{frame_idx:06d}_" if frame_idx is not None else ""
        id_suffix = f"id{track_id}" if track_id is not None else f"{self._save_counter:06d}"
        filename = self.output_dir / f"{frame_prefix}{id_suffix}.jpg"
        cv2.imwrite(str(filename), crop)
        self._save_counter += 1
