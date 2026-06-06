# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Real-time object detection, tracking, and counting pipeline built on YOLOv8 + Ultralytics trackers (ByteTrack / BoT-SORT), with an optional CLIP-based Re-ID layer for persistent object identities. Reads from a webcam or video file, renders an annotated window, and prints metrics/counts on exit. Project docs in `docs/` are written in Russian.

## Commands

Run the main app (the only "production" entry point):
```bash
python -m src.main --source 0                              # webcam index 0
python -m src.main --source data/input/video.mp4           # video file
python -m src.main --config configs/experiments/05_bytetrack_accuracy.yaml --source 0
```
`--source` overrides `video.source` in the config. Press `q` or `ESC` to quit. A digit string is parsed as a webcam index; anything else is a file path.

ReID integration test / threshold-tuning harness (there is **no pytest suite** — this is the only test, and it requires a real video + downloads model weights):
```bash
python -m scripts.test_reid_integration --source data/input/video.mp4 --max-frames 300 --threshold 0.75
```

Dataset/training scripts (all run as modules from repo root):
```bash
python -m scripts.extract_frames data/raw_videos/x.mp4 data/frames/x --step 30
python -m scripts.split_yolo_dataset <ls_export_dir> <out_dir> --val-ratio 0.2
python -m scripts.convert_yolo_predictions_to_label_studio <imgs> <labels> <out.json>
python -m scripts.review_yolo_labels <imgs> <labels> --output-review-dir outputs/label_review
```
Custom-model training uses the Ultralytics CLI directly (`yolo detect train ...`); see `docs/dataset_preparation.md`.

Export a `.pt` model to ONNX (portable) or TensorRT (NVIDIA-only, fastest), then benchmark backends:
```bash
python -m scripts.export_model --weights models/yolov8n.pt --format onnx
python -m scripts.export_model --weights models/yolov8n.pt --format engine --half
python -m scripts.export_model --weights <pt> --format engine --int8 --data <data.yaml>
python -m scripts.benchmark_backends --source <video> --models models/yolov8n.pt models/yolov8n.onnx
```
See `docs/export_optimization.md`. `.onnx`/`.engine` are gitignored (hardware/TensorRT-version specific — re-export, never commit).

## Environments

Two **separate** pip/conda environments are required — Label Studio conflicts with the runtime deps:
- `requirements.txt` — main runtime (torch, ultralytics, opencv, open-clip-torch) + the portable ONNX export/inference path (onnx, onnxruntime). Install torch from the cu118 index. Note: ultralytics 8.0.200 forces `.onnx` onto CPU (`autobackend.py:103-106`), so plain `onnxruntime` (not -gpu) is used; GPU acceleration is the TensorRT `.engine` path.
- `requirements-annotations.txt` — annotation tooling only (Label Studio), used in a `annotations` env.
- `requirements-tensorrt.txt` — optional, NVIDIA-only TensorRT (`.engine`) path; **must** be installed with `pip install --no-build-isolation -r requirements-tensorrt.txt` (the `tensorrt` meta-package's build subprocess can't see pip under build isolation). Note: TRT 8.6.1 pulls CUDA-12 wheels alongside the cu118 torch stack — unverified, validate with a real `.engine` export.

YOLO weights (`yolov8n.pt`) and ReID/CLIP weights download automatically on first run.

## Architecture

### Per-frame pipeline (`src/main.py` → `ObjectCounterApp.run`)
`VideoSource.read()` → `UltralyticsTracker.update(frame)` → `ReIDManager.update()` (optional) → `ObjectCounter.update()` → `FrameRenderer` (composites layers) + `PerformanceMetrics`. Each module is constructed in `_initialize_components()` purely from config values.

**Detection and tracking happen in one pass.** `UltralyticsTracker.update()` calls `model.track(persist=True, ...)`, which runs YOLO detection *and* tracking together. `ObjectDetector` (`src/detector.py`) exists mainly to load and own the shared `YOLO` model object — that model instance is passed into the tracker; the detector is not invoked separately in the loop. Both return the project-wide formats:

**Frame skipping (`detector.detect_interval`, default 1 = off).** When >1, the full detect+track pass runs only every Nth frame (`run` splits into `_process_inference_frame` / `_process_skipped_frame`). On skipped frames track boxes are linearly extrapolated from per-track velocity (estimated between the last two inference frames); counting and ReID run on inference frames only, render/`imshow` every frame. Two consequences threaded through the code: `lost_track_buffer` is divided by N in `_initialize_components` (the tracker only ticks on inference frames, so this keeps the lost-track window constant in *seconds* — applied to both auto and explicit values), and `PerformanceMetrics` exposes a second FPS (`get_inference_fps`, raw model rate) shown as a second overlay line when N>1. Counting on extrapolated boxes is intentionally skipped to avoid phantom line crossings / double counts. The raw-detection layer is frozen on skipped frames (no track_id to extrapolate). See `docs/frame_skipping.md`.

**Multi-backend model loading.** `detector.model` may point at a `.pt`, `.onnx`, or `.engine` file; `ObjectDetector._detect_backend()` infers the backend from the suffix. Only `.pt` gets `.to(device)` (calling it on a TensorRT engine raises). The resolved `device` is threaded through `ObjectDetector → UltralyticsTracker → model.track(device=...)`; `.engine` is forced to CUDA (GPU-only, no CPU fallback). The same pattern applies to FP16: `detector.half` (or the `--half` CLI flag, which overrides the config) is resolved once in `ObjectDetector._resolve_half` — honoured only for `.pt` on CUDA, ignored on CPU and for `.onnx`/`.engine` (their precision is baked in at export) — then threaded as `half=` into `model.track`. Note: with ultralytics 8.0.200, `.onnx` always runs on CPU (AutoBackend forces it — `autobackend.py:103-106`), so GPU acceleration is the `.engine` path, and `.onnx` is the portable/CPU path. See `docs/export_optimization.md`.
- detections: `List[((x1,y1,x2,y2), class_name, confidence)]`
- tracked_objects: `Dict[track_id, ((x1,y1,x2,y2), class_name)]`

### Two independent Re-ID layers (do not conflate them)
1. **Tracker-level Re-ID** — `tracker.algorithm: "botsort_reid"` enables BoT-SORT's built-in OSNet appearance matching, controlled by the `tracker.*` config block. Keeps tracker `track_id`s stable through occlusions.
2. **Application-level CLIP Re-ID** — the `reid.*` config block enables a separate `ObjectCropper → ObjectEmbedder (OpenCLIP) → ObjectMemory → ReIDManager` chain that assigns each object a **persistent `object_id`** surviving full tracker resets/re-entries. Lives in `src/reid.py`, `src/cropper.py`, `src/embedder.py`, `src/object_memory.py`, `src/similarity.py`, and is configured by a *second* YAML at `reid.embeddings_config` (default `configs/embeddings/default.yaml`). This layer is lazy-imported only when `reid.enabled: true`.

`ReIDManager.update()` is the subtle part: confirmed tracks update memory directly; new tracks first try `memory.find_match()` against existing objects (instant assign on hit), otherwise wait `reid.min_track_age` frames before being registered as a genuinely new object. Matching and new-object registration are split into two passes so two simultaneously-new tracks can't match each other. `reid.update_interval` runs the (expensive) embedding pass only every N frames, reusing last-known `object_id`s in between.

### Configuration (`src/config.py`)
Dot-notation YAML loader (`config.get("tracker.algorithm")`). **Critical distinction:** `get()` treats explicit `null` as "missing" and returns the default; `get_raw()` preserves explicit `null`. Use `get_raw()` for fields where `null` is a meaningful "auto" sentinel — e.g. `video.fps` (null = read from source) and `tracker.lost_track_buffer` (null = compute from FPS × `auto_lost_track_buffer_seconds`). See `_resolve_lost_track_buffer` / `_get_configured_fps` in `main.py`.

The user-facing `tracker.*` config keys are friendly names that the tracker maps to Ultralytics' raw names (e.g. `track_activation_threshold` → both `track_high_thresh` and `new_track_thresh`; `matching_cost_threshold` → `match_thresh`). `UltralyticsTracker._write_tracker_yaml` merges `_ALGORITHM_DEFAULTS[algorithm]` with these overrides and writes a generated tracker YAML to `.runtime/trackers/<algorithm>.yaml` at startup. **Do not hand-edit files in `.runtime/`** — they are regenerated every run.

`configs/experiments/` holds ready-made presets (`01_bytetrack_fast` … `05_bytetrack_accuracy`, plus `custom_*` for the trained 3-class model: `arx`, `taar`, `the_institute`).

### Counting (`src/counter.py`)
Two modes, combinable: zone-based (count once when a centroid enters `counter.count_zone` polygon) and line-crossing (per-direction in/out counts when a centroid crosses a line in `counter.crossing_lines`). Direction is determined by `point_side_of_line` sign flips tracked per `track_id`.