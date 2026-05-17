# Project Overview: Real-time Object Counter

## Purpose

A computer vision system for real-time object detection, tracking, and counting. Accepts a video stream (webcam or video file), detects objects using YOLOv8/YOLO11, assigns each object a unique ID, and counts them — by counting zone and/or line crossing.

Target scenarios: traffic monitoring, people counting, analytics on UAVs with edge devices.

**Environment constraints:** laptop with RTX 3050 Ti (~4 GB VRAM), target — 20–30 FPS in real time.

---

## System Architecture

Linear per-frame processing pipeline:

```
VideoSource → UltralyticsTracker → Counter → Renderer → Display
```

Detection and tracking are combined into a single inference pass via `model.track()` (Ultralytics API). Each component is an independent class with a well-defined interface. Orchestration is handled by `ObjectCounterApp` in `src/main.py`. Performance is monitored through a dedicated `metrics` module.

---

## Core Components

### video_source — `src/video_source.py`

Abstraction over the video source: unifies webcam (`cv2.VideoCapture(0)`) and video file handling. Captures frames, resizes to the configured resolution, and counts frames. Supports a context manager for proper resource cleanup.

### detector — `src/detector.py`

Loads the YOLOv8/YOLO11 model (Ultralytics) and holds its instance. The model is shared with the tracker — detection happens inside `UltralyticsTracker.update()`, so there is no separate inference call in the pipeline.

### tracker — `src/tracker.py`

`UltralyticsTracker` — a wrapper around the Ultralytics tracking API (`model.track()`). Combines detection and tracking in a single inference pass.

**Supported algorithms:**
- **ByteTrack** — two-stage IoU matching (high- and low-confidence detections). Recommended for static cameras.
- **BoT-SORT** — extends ByteTrack with Global Motion Compensation (GMC) to stabilize tracks when the camera is moving.
- **BoT-SORT + Re-ID** (`botsort_reid`) — adds ID recovery via visual similarity (OSNet) during long occlusions.

On initialization, `UltralyticsTracker` generates a tracker YAML config in `.runtime/trackers/` and passes its path to `model.track()`. Supports class filtering (`allowed_classes`): class names are mapped to model class IDs and passed directly to inference.

**`update()` output format:**
- `detections`: `List[((x1,y1,x2,y2), class_name, confidence)]`
- `tracked_objects`: `Dict[track_id, ((x1,y1,x2,y2), class_name)]`

### counter — `src/counter.py`

Counts objects using two independent methods (combinable):

- **Zone counting** (`count_zone`): an object is counted exactly once when its centroid first enters the polygon. `null` — count all objects in the frame. Stores per-class and total counters.
- **Line crossing counting** (`crossing_lines`): increments the `in` or `out` direction counter each time a track centroid crosses a line (determined by the side change relative to the line vector). Supports multiple named lines.

Track side state (`_prev_sides`) is automatically cleaned up when a track disappears.

### renderer — `src/renderer.py`

Draws annotations over the frame using OpenCV: detection bounding boxes with confidence, track bounding boxes with IDs, zone counter panel, line counter panel, crossing line visualization, FPS indicator. Track colors are deterministic by `track_id % 256` — the same object always has the same color throughout a session.

### metrics — `src/metrics.py`

Measures pipeline performance: FPS (30-frame sliding window), average detector and tracker inference time in milliseconds. Does not affect processing logic — monitoring only.

---

## Supporting Components

- **`src/config.py`** — YAML configuration loader with dot-notation support (`video.frame_width`). `get_raw()` returns a value without a fallback (needed to distinguish `null` from a missing key).
- **`src/utils.py`** — geometric utilities: bbox centroid, Euclidean distance, IoU, point-in-polygon check (ray casting), point side of line (`point_side_of_line`).

---

## Utilities and Scripts

### extract_frames — `scripts/extract_frames.py`

Extracts every N-th frame from a video file and saves as JPG. Supports batch processing of multiple videos into a single directory with continued numbering. Used for dataset preparation.

```bash
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip --step 30
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip --step 15 --dry-run
```

See `docs/en/extract_frames.md` for details.

### cors_http_server — `scripts/cors_http_server.py`

Starts a local HTTP server (port 9000) with CORS headers to allow Label Studio access to images from the filesystem.

```bash
python -m scripts.cors_http_server
```

### convert_yolo_predictions_to_label_studio — `scripts/convert_yolo_predictions_to_label_studio.py`

Converts YOLO predictions (`.txt` files in `class cx cy w h` format) to Label Studio JSON format for importing and reviewing annotations.

```bash
python -m scripts.convert_yolo_predictions_to_label_studio \
  data/frames/all \
  custom_models/bootstrap_predictions/run/labels \
  outputs/predictions_all.json
```

### review_yolo_labels — `scripts/review_yolo_labels.py`

OpenCV-based interactive label viewer. Lets you quickly review images with YOLO bboxes and make a decision for each: approve, delete annotation, or flag for manual review. Logs results to CSV.

```bash
python -m scripts.review_yolo_labels \
  data/frames/all \
  custom_models/bootstrap_predictions/run/labels \
  --output-review-dir outputs/label_review \
  --dry-run
```

Keys: `n`/`p` — next/previous, `k` — OK, `d` — delete annotation, `m` — flag for review, `q` — quit.

### split_yolo_dataset — `scripts/split_yolo_dataset.py`

Splits a Label Studio export into train and validation sets in YOLO format. Generates `data.yaml` for `yolo detect train`.

```bash
python -m scripts.split_yolo_dataset \
  data/bootstrap_export/project-export-dir \
  data/yolo_output/project-output-dir \
  --val-ratio 0.2
```

### Annotation tools (separate `annotations` environment)

**Label Studio** — web interface for centralized annotation project management:
```bash
conda activate annotations
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio
```

**labelImg** — GUI for quick image annotation in YOLO/Pascal VOC format:
```bash
conda activate annotations
labelImg
```

Require a separate environment due to dependency conflicts. Installation instructions — in `docs/en/how_to_run.md`.

Full dataset preparation and model training workflow — in `docs/en/dataset_preparation.md`.

---

## Configuration

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Main default config |
| `configs/experiments/*.yaml` | Experiment presets (same structure) |
| `.runtime/trackers/*.yaml` | Auto-generated at runtime, in `.gitignore` |

Running with a custom config:
```bash
python -m src.main --config configs/experiments/01_bytetrack_fast.yaml
python -m src.main --source /path/to/video.mp4
```

Available presets:
- `01_bytetrack_fast.yaml` — maximum speed
- `02_bytetrack_stable.yaml` — stable tracking
- `03_botsort_balanced.yaml` — moving camera
- `04_botsort_reid_light.yaml` — Re-ID with minimal overhead
- `05_bytetrack_accuracy.yaml` — maximum accuracy

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv8 / YOLO11 (Ultralytics) |
| Tracking | ByteTrack / BoT-SORT (Ultralytics built-in) |
| Re-ID | OSNet (torchreid, via Ultralytics) |
| Inference | PyTorch + CUDA |
| Video capture | OpenCV |
| Configuration | YAML |
| Language | Python 3.10+ |

---

## Architectural Notes

**Current limitations:**
- Synchronous pipeline: inference blocks the main thread.
- `output.save_video` is defined in the config but video writing is not connected to the pipeline.
- `calculate_iou` in `utils.py` is implemented but unused.

**Development roadmap:**
- Async frame capture (separate thread for `VideoSource`)
- Annotated video recording (`output.save_video`)
- Model export to ONNX / TensorRT for inference optimization
- Integration with a drone simulator (PX4 / AirSim)
