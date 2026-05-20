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
        min_track_age: Minimum consecutive frames a brand-new track must be
            visible before it is registered as a new object in memory (and
            counted in unique/active stats).  Tracks that match an existing
            object via embedding similarity are assigned immediately regardless
            of this value.  Default 1 = disabled (instant registration).
    """

    def __init__(
        self,
        cropper: ObjectCropper,
        embedder: ObjectEmbedder,
        memory: ObjectMemory,
        min_track_age: int = 1,
    ) -> None:
        self.cropper = cropper
        self.embedder = embedder
        self.memory = memory
        self.min_track_age = max(1, min_track_age)

        # Maps currently active tracker track_id -> persistent object_id
        self._track_to_object: Dict[int, int] = {}
        # First frame_idx at which each unconfirmed (pending) track was seen
        self._pending_first_frame: Dict[int, int] = {}

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

        # ── 1. Crop all tracked objects ──────────────────────────────────
        track_ids = list(tracked_objects.keys())
        crops = []
        valid_track_ids = []

        for track_id in track_ids:
            bbox, class_name = tracked_objects[track_id]
            known_object_id = self._track_to_object.get(track_id)
            crop = self.cropper.crop(
                frame, bbox, track_id=track_id, frame_idx=frame_idx,
                object_id=known_object_id,
            )
            if crop.size == 0:
                continue
            crops.append(crop)
            valid_track_ids.append(track_id)

        if not crops:
            # Clean up stale mappings
            active_track_ids = set(tracked_objects.keys())
            for tid in [t for t in self._track_to_object if t not in active_track_ids]:
                del self._track_to_object[tid]
            return result

        # ── 2. Embed all crops in a single batch pass ────────────────────
        embeddings = self.embedder.embed_batch(crops)  # (N, dim)

        # ── 3a. Matching pass — all find_match() calls against pre-existing memory ──
        # Confirmed tracks are updated; unconfirmed tracks are matched or queued.
        # No new objects are added to memory here, so two pending tracks that both
        # reach min_track_age in the same update() call cannot match each other.
        pending_to_confirm: list = []  # (track_id, bbox, class_name, embedding)

        for i, track_id in enumerate(valid_track_ids):
            bbox, class_name = tracked_objects[track_id]
            embedding = embeddings[i]

            if track_id in self._track_to_object:
                # Already confirmed — update memory and emit result.
                object_id = self._track_to_object[track_id]
                self.memory.update(object_id, bbox, embedding, track_id, frame_idx)
                result[track_id] = object_id
            else:
                matched_id, _score = self.memory.find_match(embedding, frame_idx)

                if matched_id is not None:
                    # Re-ID hit against an existing confirmed object — assign immediately.
                    self.memory.update(matched_id, bbox, embedding, track_id, frame_idx)
                    self._track_to_object[track_id] = matched_id
                    self._pending_first_frame.pop(track_id, None)
                    result[track_id] = matched_id
                else:
                    # No match — apply min_track_age gate using real frame distance.
                    if track_id not in self._pending_first_frame:
                        self._pending_first_frame[track_id] = frame_idx
                    age = frame_idx - self._pending_first_frame[track_id] + 1
                    if age >= self.min_track_age:
                        pending_to_confirm.append((track_id, bbox, class_name, embedding))
                    # else: still pending — not yet included in result or memory

        # ── 3b. Confirmation pass — add newly-confirmed tracks to memory ─────────
        # Runs after all matching is complete so new objects don't interfere with
        # each other's find_match() calls from step 3a.
        for track_id, bbox, class_name, embedding in pending_to_confirm:
            object_id = self.memory.add(class_name, bbox, embedding, track_id, frame_idx)
            self._track_to_object[track_id] = object_id
            del self._pending_first_frame[track_id]
            result[track_id] = object_id

        # ── 4. Remove stale mappings for tracks that are no longer active ────────
        active_track_ids = set(tracked_objects.keys())
        for tid in [t for t in self._track_to_object if t not in active_track_ids]:
            del self._track_to_object[tid]
        for tid in [t for t in self._pending_first_frame if t not in active_track_ids]:
            del self._pending_first_frame[tid]

        return result

    def get_object_id(self, track_id: int) -> Optional[int]:
        """Return the persistent object_id for a given track_id, if known."""
        return self._track_to_object.get(track_id)

    def active_object_count(self) -> int:
        return self.memory.active_count()

    def total_object_count(self) -> int:
        return self.memory.total_count()
