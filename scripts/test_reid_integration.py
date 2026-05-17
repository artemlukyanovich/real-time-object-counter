"""Integration test: YOLO → Tracker → Crop → Embedding → ReID on a real video.

Purpose
-------
Verify that the ReID pipeline works correctly on real detections, not synthetic
data.  The script collects per-frame statistics, saves crops to outputs/crops/,
and prints a summary that makes it easy to choose a good similarity_threshold.

Usage
-----
    python -m scripts.test_reid_integration --source data/input/your_video.mp4
    python -m scripts.test_reid_integration --source 0          # webcam
    python -m scripts.test_reid_integration --source data/input/video.mp4 \\
        --config configs/default.yaml \\
        --max-frames 300 \\
        --threshold 0.75

Output
------
- Crops saved to outputs/crops/ (named frame<N>_id<track_id>_obj<object_id>.jpg when object_id is known)
- Console log: per-frame event table
- Final summary: similarity distributions and re-id statistics
"""

import argparse
import sys
from pathlib import Path

# Allow running as `python -m scripts.test_reid_integration` from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.config import Config
from src.detector import ObjectDetector
from src.tracker import UltralyticsTracker
from src.cropper import ObjectCropper
from src.embedder import ObjectEmbedder
from src.object_memory import ObjectMemory
from src.reid import ReIDManager


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize_source(source: str):
    return int(source) if source.isdigit() else source


def _resolve_fps(cap: cv2.VideoCapture, config: Config) -> float:
    explicit = config.get_raw("video.fps")
    if explicit is not None:
        return float(explicit)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps and src_fps > 0:
        return src_fps
    return float(config.get("video.fallback_fps", 30))


# ── main ─────────────────────────────────────────────────────────────────────

def run(
    source,
    config_path: str,
    max_frames: int,
    threshold: float,
) -> None:
    config = Config(config_path)

    # --- Video source ---
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source '{source}'")
        sys.exit(1)

    fps = _resolve_fps(cap, config)
    w = int(config.get("video.frame_width", 1280))
    h = int(config.get("video.frame_height", 720))

    # --- Detector + Tracker (shared model, same as main.py) ---
    model_path = config.get("detector.model", "models/yolov8n.pt")
    confidence = config.get("detector.confidence_threshold", 0.5)
    device = config.get("detector.device", "cpu")
    allowed_classes = config.get("detector.allowed_classes", None)

    detector = ObjectDetector(model_path, confidence, device)

    lost_buffer = max(1, round(fps * config.get("tracker.auto_lost_track_buffer_seconds", 3.0)))
    tracker = UltralyticsTracker(
        model=detector.model,
        conf_threshold=confidence,
        frame_rate=int(fps),
        algorithm=config.get("tracker.algorithm", "bytetrack"),
        track_activation_threshold=config.get("tracker.track_activation_threshold", 0.5),
        track_low_threshold=config.get("tracker.track_low_threshold", 0.1),
        lost_track_buffer=lost_buffer,
        matching_cost_threshold=config.get("tracker.matching_cost_threshold", 0.8),
        allowed_classes=allowed_classes,
    )

    # --- ReID pipeline ---
    cropper = ObjectCropper(
        padding=8,
        save_crops=True,
        output_dir="outputs/crops",
    )
    embedder = ObjectEmbedder(device=device)
    memory = ObjectMemory(
        similarity_threshold=threshold,
        max_missing_frames=int(fps * 3),
    )
    reid = ReIDManager(cropper, embedder, memory)

    # --- Statistics accumulators ---
    # Collect similarity scores separately for:
    #   "same"   — track_id already linked to an object_id (continuous update)
    #   "match"  — new track_id matched to existing object_id (re-identification event)
    #   "new"    — new track_id, no match found → new object registered
    same_scores: list = []
    match_scores: list = []
    new_scores: list = []   # best score when no match was found
    reid_events: list = []  # (frame_idx, track_id, old_object_id → new? )

    # Track previous track→object mapping to detect re-id events
    prev_track_to_obj: dict = {}

    print(f"\n{'─'*72}")
    print(f"  Integration ReID test")
    print(f"  source    : {source}")
    print(f"  config    : {config_path}")
    print(f"  device    : {device}")
    print(f"  threshold : {threshold}")
    print(f"  max_frames: {max_frames}")
    print(f"{'─'*72}\n")
    print(f"{'frame':>6}  {'track_id':>8}  {'object_id':>9}  {'event':12}  {'score':>6}")
    print(f"{'─'*72}")

    frame_idx = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            print("\n[end of stream]")
            break

        # Resize to configured resolution
        frame = cv2.resize(frame, (w, h))

        # Detection + tracking
        _detections, tracked_objects = tracker.update(frame)

        if not tracked_objects:
            frame_idx += 1
            continue

        # ReID — we hook into the internal state to capture per-track scores
        # by manually stepping through the logic and recording scores.
        memory.expire_old(frame_idx)

        for track_id, (bbox, class_name) in tracked_objects.items():
            # Pre-look up already-resolved object_id so saved crops include it.
            known_obj_id = reid._track_to_object.get(track_id)
            crop = cropper.crop(
                frame, bbox, track_id=track_id, frame_idx=frame_idx,
                object_id=known_obj_id,
            )
            if crop.size == 0:
                continue

            embedding = embedder.embed(crop)

            if track_id in reid._track_to_object:
                # Continuous update — same track still active
                obj_id = reid._track_to_object[track_id]
                memory.update(obj_id, bbox, embedding, track_id, frame_idx)
                # Score: similarity to own mean embedding (before update adds new vec)
                record = memory.get(obj_id)
                if record and len(record.embeddings) > 1:
                    mean_emb = np.mean(record.embeddings[:-1], axis=0).astype(np.float32)
                    from src.similarity import cosine_similarity
                    s = cosine_similarity(embedding, mean_emb)
                    same_scores.append(s)
                event = "continuous"
                score_str = f"{same_scores[-1]:.4f}" if same_scores else "  n/a"
            else:
                matched_id, score = memory.find_match(embedding, frame_idx)
                if matched_id is not None:
                    # Re-identification
                    reid._track_to_object[track_id] = matched_id
                    memory.update(matched_id, bbox, embedding, track_id, frame_idx)
                    match_scores.append(score)
                    obj_id = matched_id
                    event = "RE-ID ✓"
                    score_str = f"{score:.4f}"
                    reid_events.append((frame_idx, track_id, matched_id, score))
                else:
                    # New object
                    new_scores.append(score)
                    obj_id = memory.add(class_name, bbox, embedding, track_id, frame_idx)
                    reid._track_to_object[track_id] = obj_id
                    event = "new"
                    score_str = f"{score:.4f}"

            print(f"{frame_idx:>6}  {track_id:>8}  {obj_id:>9}  {event:12}  {score_str:>6}")

        # Clean up stale track mappings (mirrors ReIDManager.update logic)
        active = set(tracked_objects.keys())
        for tid in [t for t in reid._track_to_object if t not in active]:
            del reid._track_to_object[tid]

        prev_track_to_obj = {t: reid._track_to_object.get(t) for t in active}
        frame_idx += 1

    cap.release()

    # ── Summary ──────────────────────────────────────────────────────────────

    def _stats(values: list, label: str) -> None:
        if not values:
            print(f"  {label:30s}  n=0")
            return
        arr = np.array(values)
        print(
            f"  {label:30s}  n={len(arr):4d}  "
            f"min={arr.min():.4f}  mean={arr.mean():.4f}  max={arr.max():.4f}"
        )

    print(f"\n{'─'*72}")
    print("  Similarity score distributions")
    print(f"{'─'*72}")
    _stats(same_scores,  "same object (continuous)")
    _stats(match_scores, "re-id match (new track_id)")
    _stats(new_scores,   "no match (new object)")

    print(f"\n{'─'*72}")
    print("  Re-identification events")
    print(f"{'─'*72}")
    if reid_events:
        for fi, tid, oid, sc in reid_events:
            print(f"  frame={fi:5d}  track_id={tid:4d}  → object_id={oid:4d}  score={sc:.4f}")
    else:
        print("  (none — no track was lost and recovered within max_missing_frames)")

    print(f"\n{'─'*72}")
    print(f"  Total unique objects : {reid.total_object_count()}")
    print(f"  Active objects       : {reid.active_object_count()}")
    print(f"  Crops saved to       : outputs/crops/")
    print(f"{'─'*72}\n")

    if same_scores and new_scores:
        gap = np.mean(same_scores) - np.mean(new_scores)
        suggested = np.mean(same_scores) - gap * 0.4
        print(
            f"  Suggested threshold range: "
            f"{suggested - 0.05:.2f} – {suggested + 0.05:.2f}  "
            f"(current: {threshold})\n"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integration test: YOLO → Tracker → ReID on real video"
    )
    p.add_argument(
        "--source",
        required=True,
        help="Video file path or webcam index (e.g. 0)",
    )
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to pipeline config (default: configs/default.yaml)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Number of frames to process (default: 300)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for re-id matching (default: 0.75)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        source=_normalize_source(args.source),
        config_path=args.config,
        max_frames=args.max_frames,
        threshold=args.threshold,
    )
