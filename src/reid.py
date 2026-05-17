"""Object Re-Identification logic.

Answers the question: "Is this tracked object one we have seen before,
or is it genuinely new?"

Pipeline for each frame:
    tracked_objects (from Tracker)
        -> ObjectCropper  -> crop images
        -> ObjectEmbedder -> embedding vectors
        -> ObjectMemory   -> match against known objects
        -> Dict[track_id, object_id]
"""

from typing import Dict, Optional, Tuple

import numpy as np

from src.cropper import ObjectCropper
from src.embedder import ObjectEmbedder
from src.object_memory import ObjectMemory


BBox = Tuple[int, int, int, int]


class ReIDManager:
    """Resolves persistent object identities across tracker ID switches.

    Each tracker track_id is mapped to a persistent object_id that survives
    re-entries, occlusions, and tracker resets.

    Args:
        cropper: ObjectCropper instance for extracting bbox crops.
        embedder: ObjectEmbedder instance for generating feature vectors.
        memory: ObjectMemory instance for storing and searching known objects.
    """

    def __init__(
        self,
        cropper: ObjectCropper,
        embedder: ObjectEmbedder,
        memory: ObjectMemory,
    ) -> None:
        self.cropper = cropper
        self.embedder = embedder
        self.memory = memory

        # Maps currently active tracker track_id -> persistent object_id
        self._track_to_object: Dict[int, int] = {}

    def update(
        self,
        frame: np.ndarray,
        tracked_objects: Dict[int, Tuple[BBox, str]],
        frame_idx: int,
    ) -> Dict[int, int]:
        """Resolve object identities for all currently tracked objects.

        Args:
            frame: Current BGR frame (H, W, 3).
            tracked_objects: Dict[track_id, ((x1,y1,x2,y2), class_name)]
                             as returned by UltralyticsTracker.update().
            frame_idx: Monotonically increasing frame index.

        Returns:
            Dict[track_id, object_id] — persistent identity for every
            track that produced a valid crop+embedding.
        """
        self.memory.expire_old(frame_idx)

        result: Dict[int, int] = {}

        for track_id, (bbox, class_name) in tracked_objects.items():
            crop = self.cropper.crop(
                frame, bbox, track_id=track_id, frame_idx=frame_idx
            )
            if crop.size == 0:
                continue

            embedding = self.embedder.embed(crop)

            if track_id in self._track_to_object:
                # This track is already linked to a known object — just update.
                object_id = self._track_to_object[track_id]
                self.memory.update(object_id, bbox, embedding, track_id, frame_idx)
            else:
                # New tracker ID — try to match against memory (re-identification).
                matched_id, _score = self.memory.find_match(embedding, frame_idx)

                if matched_id is not None:
                    # Re-identified: link this tracker ID to the existing object.
                    object_id = matched_id
                    self.memory.update(object_id, bbox, embedding, track_id, frame_idx)
                else:
                    # No match — register as a new object.
                    object_id = self.memory.add(
                        class_name, bbox, embedding, track_id, frame_idx
                    )

                self._track_to_object[track_id] = object_id

            result[track_id] = self._track_to_object[track_id]

        # Remove stale mappings for tracks that are no longer active.
        active_track_ids = set(tracked_objects.keys())
        for tid in [t for t in self._track_to_object if t not in active_track_ids]:
            del self._track_to_object[tid]

        return result

    def get_object_id(self, track_id: int) -> Optional[int]:
        """Return the persistent object_id for a given track_id, if known."""
        return self._track_to_object.get(track_id)

    def active_object_count(self) -> int:
        return self.memory.active_count()

    def total_object_count(self) -> int:
        return self.memory.total_count()
