"""In-memory registry of known objects with embeddings."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.similarity import find_best_match


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class ObjectRecord:
    """State for a single known object."""

    object_id: int
    class_name: str
    bbox: BBox
    track_id: Optional[int]       # current tracker track_id; None when lost
    embeddings: List[np.ndarray]  # rolling buffer of past embeddings
    last_seen_frame: int
    last_seen_time: float = field(default_factory=time.time)
    is_active: bool = True

    def mean_embedding(self) -> np.ndarray:
        """Mean of all stored embeddings (float32)."""
        return np.mean(self.embeddings, axis=0).astype(np.float32)


class ObjectMemory:
    """In-memory registry of tracked objects with embeddings.

    Lifecycle:
      - ``add()`` registers a brand-new object and returns its object_id.
      - ``update()`` refreshes bbox/embedding for an already-known object.
      - ``find_match()`` searches active objects for the closest embedding.
      - ``expire_old()`` deactivates objects not seen for too long.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_missing_frames: int = 90,
        max_embeddings_per_object: int = 5,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_missing_frames = max_missing_frames
        self.max_embeddings_per_object = max_embeddings_per_object

        self._records: Dict[int, ObjectRecord] = {}
        self._next_id: int = 1

    def add(
        self,
        class_name: str,
        bbox: BBox,
        embedding: np.ndarray,
        track_id: Optional[int],
        frame_idx: int,
    ) -> int:
        """Register a new object. Returns the assigned object_id."""
        object_id = self._next_id
        self._next_id += 1
        self._records[object_id] = ObjectRecord(
            object_id=object_id,
            class_name=class_name,
            bbox=bbox,
            track_id=track_id,
            embeddings=[embedding.copy()],
            last_seen_frame=frame_idx,
        )
        return object_id

    def update(
        self,
        object_id: int,
        bbox: BBox,
        embedding: np.ndarray,
        track_id: Optional[int],
        frame_idx: int,
    ) -> None:
        """Update state for an existing object."""
        record = self._records[object_id]
        record.bbox = bbox
        record.track_id = track_id
        record.last_seen_frame = frame_idx
        record.last_seen_time = time.time()
        record.is_active = True

        record.embeddings.append(embedding.copy())
        if len(record.embeddings) > self.max_embeddings_per_object:
            record.embeddings.pop(0)

    def find_match(
        self,
        embedding: np.ndarray,
        current_frame: int,
    ) -> Tuple[Optional[int], float]:
        """Search active objects for the best embedding match.

        Only considers objects that are active and have been seen within
        ``max_missing_frames`` frames.

        Returns:
            (object_id, score) if a match above threshold is found,
            (None, best_score) otherwise.
        """
        candidate_ids = [
            oid
            for oid, rec in self._records.items()
            if rec.is_active
            and (current_frame - rec.last_seen_frame) <= self.max_missing_frames
        ]

        if not candidate_ids:
            return None, 0.0

        gallery = np.stack(
            [self._records[oid].mean_embedding() for oid in candidate_ids]
        )
        return find_best_match(
            embedding, gallery, candidate_ids, self.similarity_threshold
        )

    def expire_old(self, current_frame: int) -> List[int]:
        """Deactivate objects not seen within ``max_missing_frames``.

        Returns:
            List of object_ids that were deactivated in this call.
        """
        expired = []
        for oid, rec in self._records.items():
            if (
                rec.is_active
                and (current_frame - rec.last_seen_frame) > self.max_missing_frames
            ):
                rec.is_active = False
                expired.append(oid)
        return expired

    def get(self, object_id: int) -> Optional[ObjectRecord]:
        return self._records.get(object_id)

    def active_count(self) -> int:
        return sum(1 for r in self._records.values() if r.is_active)

    def total_count(self) -> int:
        return len(self._records)

    def __len__(self) -> int:
        return len(self._records)
