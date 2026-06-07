# Project Overview: Real-time Object Counter

## Purpose

A computer vision system for real-time detection, tracking, and counting of objects. It takes a video stream (webcam or video file), recognizes objects with YOLOv8/YOLO11, assigns each a unique ID, and counts them — by a counting zone and/or by line crossings.

Target scenarios: traffic monitoring, people counting, on-board UAV analytics on edge devices.

**Environment constraints:** a laptop with an RTX 3050 Ti (~4 GB VRAM), target — 20–30 FPS in real time.

---

## System Architecture

A linear per-frame processing pipeline:

```
VideoSource → UltralyticsTracker → Counter → Renderer → Display
```

Detection and tracking are combined into a single inference pass via `model.track()` (Ultralytics API). Each component is an independent class with a clearly defined interface. Orchestration is performed by `ObjectCounterApp` in `src/main.py`. Performance is tracked via a separate `metrics` module.

---

## Core Components

### video_source — `src/video_source.py`

An abstraction over the video source: it unifies work with a webcam (`cv2.VideoCapture(0)`) and a video file. It performs frame capture, resizing to a given resolution, and frame counting. Supports a context manager for proper resource release.

### detector — `src/detector.py`

Loads the YOLOv8/YOLO11 model (Ultralytics) and holds its instance. The model is shared with the tracker — detection happens inside `UltralyticsTracker.update()`, so there is no separate inference call in the pipeline.

### tracker — `src/tracker.py`

`UltralyticsTracker` is a wrapper over the Ultralytics tracking API (`model.track()`). It combines detection and tracking in a single inference pass.

**Supported algorithms:**
- **ByteTrack** — two-stage IoU matching (high- and low-confidence detections). Recommended for a static camera.
- **BoT-SORT** — extends ByteTrack with Global Motion Compensation (GMC) to stabilize tracks when the camera moves.
- **BoT-SORT + Re-ID** (`botsort_reid`) — adds ID recovery by visual appearance similarity (OSNet) during long occlusions.

On initialization, `UltralyticsTracker` generates a tracker YAML config in `.runtime/trackers/` and passes its path to `model.track()`. It supports class filtering (`allowed_classes`): names are converted into the model's class IDs and passed directly into inference.

**Output format of `update()`:**
- `detections`: `List[((x1,y1,x2,y2), class_name, confidence)]`
- `tracked_objects`: `Dict[track_id, ((x1,y1,x2,y2), class_name)]`

### counter — `src/counter.py`

Counts objects using two independent methods (which can be combined):

- **Zone-based counting** (`count_zone`): an object is counted exactly once, the first time its centroid enters the polygon. `null` — count all objects in the frame. Maintains per-class counters and a total counter.
- **Line-crossing counting** (`crossing_lines`): on each crossing of a line by a track's centroid, it increments the `in` or `out` direction counter (determined by the side change relative to the line vector). Supports multiple named lines.

Track side state (`_prev_sides`) is automatically cleared when a track disappears.

### renderer — `src/renderer.py`

Draws annotations over the frame using OpenCV: detection boxes with confidence, track boxes with IDs, a zone counters panel, a line counters panel, visualization of crossing lines, and an FPS indicator. Track colors are deterministic by `track_id % 256` — one object always keeps the same color throughout the session.

### metrics — `src/metrics.py`

Measures pipeline performance: FPS (30-frame sliding window), average detector and tracker inference time in milliseconds. It does not affect processing logic — monitoring only.

---

## Auxiliary Components

- **`src/config.py`** — loads YAML configuration with dot-notation support (`video.frame_width`). The `get_raw()` method returns the value without a fallback (needed to distinguish `null` from a missing key).
- **`src/utils.py`** — geometric utilities: bbox centroid, Euclidean distance, IoU, point-in-polygon check (`ray casting`), determination of a point's side relative to a line (`point_side_of_line`).

---

## Utilities and Scripts

### extract_frames — `scripts/extract_frames.py`

Extracts every Nth frame from a video file and saves it as JPG. Supports batch processing of several videos into one directory with continued numbering. Used for dataset preparation.

```bash
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip --step 30
python -m scripts.extract_frames data/raw_videos/clip.mp4 data/frames/clip --step 15 --dry-run
```

See `docs/extract_frames.md` for details.

### cors_http_server — `scripts/cors_http_server.py`

Starts a local HTTP server (port 9000) with CORS headers so that Label Studio can access images from the filesystem.

```bash
python -m scripts.cors_http_server
```

### convert_yolo_predictions_to_label_studio — `scripts/convert_yolo_predictions_to_label_studio.py`

Converts YOLO predictions (`.txt` files in `class cx cy w h` format) into the Label Studio JSON format for import and label review.

```bash
python -m scripts.convert_yolo_predictions_to_label_studio \
  data/frames/all \
  custom_models/bootstrap_predictions/run/labels \
  outputs/predictions_all.json
```

### review_yolo_labels — `scripts/review_yolo_labels.py`

An interactive OpenCV-based label viewer. It lets you quickly review images with YOLO bboxes and make a decision on each: approve, delete the labels, or mark for manual review. Logs results to CSV.

```bash
python -m scripts.review_yolo_labels \
  data/frames/all \
  custom_models/bootstrap_predictions/run/labels \
  --output-review-dir outputs/label_review \
  --dry-run
```

Keys: `n`/`p` — next/previous, `k` — OK, `d` — delete labels, `m` — mark for review, `q` — quit.

### split_yolo_dataset — `scripts/split_yolo_dataset.py`

Splits a Label Studio export into training and validation sets in YOLO format. Generates a `data.yaml` for running `yolo detect train`.

```bash
python -m scripts.split_yolo_dataset \
  data/bootstrap_export/project-export-dir \
  data/yolo_output/project-output-dir \
  --val-ratio 0.2
```

### Annotation tools (separate `annotations` environment)

**Label Studio** — a web interface for centralized management of the annotation project:
```bash
conda activate annotations
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio
```

**labelImg** — a GUI for fast image annotation in YOLO/Pascal VOC format:
```bash
conda activate annotations
labelImg
```

They require a separate environment due to dependency conflicts. Installation instructions are in `docs/how_to_run.md`.

The full dataset-preparation and model-training workflow is in `docs/dataset_preparation.md`.

---

## Configuration

| File | Purpose |
|------|-----------|
| `configs/default.yaml` | Main default config |
| `configs/experiments/*.yaml` | Presets for experiments (same structure) |
| `.runtime/trackers/*.yaml` | Generated automatically at startup, in `.gitignore` |

Run with a custom config:
```bash
python -m src.main --config configs/experiments/01_bytetrack_fast.yaml
python -m src.main --source /path/to/video.mp4
```

Available presets:
- `01_bytetrack_fast.yaml` — maximum speed
- `02_bytetrack_stable.yaml` — stable tracking
- `03_botsort_balanced.yaml` — moving camera
- `04_botsort_reid_light.yaml` — Re-ID with minimal load
- `05_bytetrack_accuracy.yaml` — maximum accuracy

---

## Technology Stack

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
- `output.save_video` is provided in the config, but video recording is not wired into the pipeline.
- `calculate_iou` in `utils.py` is implemented but not used anywhere.

**Direction of development:**
- Asynchronous frame capture (a separate thread for `VideoSource`)
- Recording of the annotated video (`output.save_video`)
- Model export to ONNX / TensorRT for inference optimization
- Integration with a drone simulator (PX4 / AirSim)
