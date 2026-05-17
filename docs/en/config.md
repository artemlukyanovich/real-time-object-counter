# Configuration

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Main default config |
| `configs/experiments/*.yaml` | Experiment configs (same structure as `default.yaml`) |

Loaded by the `Config` class (`src/config.py`). Parameters are accessed via dot-notation: `config.get("detector.confidence_threshold")`.

**Running with a custom config:**
```bash
python -m src.main --config configs/experiments/my_experiment.yaml
```

---

## File Structure

```yaml
video:
  source: 0
  fps: null
  fallback_fps: 30
  frame_width: 1280
  frame_height: 720

detector:
  model: "models/yolov8n.pt"
  confidence_threshold: 0.5
  device: "cuda"
  allowed_classes: null

tracker:
  algorithm: "bytetrack"
  track_activation_threshold: 0.5
  track_low_threshold: 0.1
  matching_cost_threshold: 0.8
  lost_track_buffer: null
  auto_lost_track_buffer_seconds: 3.0
  fuse_score: true          # BoT-SORT only
  gmc_method: "sparseOptFlow"  # BoT-SORT only
  reid_weights: "osnet_x0_25_market.pt"  # botsort_reid only
  proximity_threshold: 0.5    # botsort_reid only
  appearance_threshold: 0.25  # botsort_reid only

counter:
  enable: true
  count_zone: null
  crossing_lines: null

output:
  save_video: false
  output_dir: "outputs"
  video_name: "output.mp4"
  fps: 30

display:
  show_detections: true
  show_tracking_ids: true
  show_counts: true
  font_size: 1.0
```

---

## `video` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | int / str | `0` | Source: `0` — first camera, `1` — second, file path — video file |
| `fps` | int / null | `null` | `null` — read FPS from source; explicit value overrides it |
| `fallback_fps` | int | `30` | Default FPS if the source reports `0` or nothing |
| `frame_width` | int | `1280` | Frame width after resize (pixels) |
| `frame_height` | int | `720` | Frame height after resize (pixels) |

**Performance impact:**
- Reducing resolution to 640×480 can double FPS on CPU inference
- Increasing resolution improves detection of small objects but raises inference and resize time

---

## `detector` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"yolov8n.pt"` | Name or path to YOLOv8 weight file |
| `confidence_threshold` | float | `0.5` | Minimum detection confidence [0.0–1.0] |
| `device` | str | `"cuda"` | Inference device: `"cuda"` or `"cpu"` |
| `allowed_classes` | list / null | `null` | Classes to detect/track. `null` = all classes. Example: `["person"]` or `["person", "car", "truck"]` |

**Class filtering (`allowed_classes`):**

Passed directly to Ultralytics as `classes=[id, ...]` — filtering happens at the model level before the tracker, so it does not affect tracking speed. Class names correspond to COCO dataset labels (for standard YOLOv8/YOLO11 models).

```yaml
# Count only people:
detector:
  allowed_classes: ["person"]

# Count vehicles:
detector:
  allowed_classes: ["car", "truck", "bus", "motorcycle"]
```

If a specified class is not found in the model — a warning is printed with the list of available classes, and the class is ignored.

**Model selection:**

Ultralytics automatically downloads weights on first run if the file is not found locally.

| Model | Speed (GPU) | Accuracy | Notes |
|-------|------------|---------|-------|
| `yolov8n.pt` | ~5–10 ms | Baseline | Recommended for real-time |
| `yolov8s.pt` | ~10–20 ms | Medium | Speed/accuracy trade-off |
| `yolov8m.pt` | ~20–40 ms | High | Requires a powerful GPU |
| `yolov8l.pt` | ~30–60 ms | Very high | For server inference |
| `yolov8x.pt` | ~50–100 ms | Maximum | Most accurate YOLOv8 model |
| `yolo11n.pt` | ~4–8 ms | Baseline | Newer architecture, faster than v8n |
| `yolo11s.pt` | ~8–15 ms | Medium | New architecture, balanced |
| `yolo11m.pt` | ~15–30 ms | High | New architecture, high accuracy |

> `yolo11*` models (YOLO11) are the current generation from Ultralytics; all else being equal, prefer them over `yolov8*`.

**`confidence_threshold` impact:**
- `0.3–0.4` — more detections, more false positives → pollutes the tracker
- `0.5` — balanced (default)
- `0.7+` — only confident detections; possible misses when objects are partially occluded

**`device` impact:**
- `"cuda"` — GPU inference (~5–15 ms for yolov8n)
- `"cpu"` — CPU inference (~50–200 ms), FPS drops to 5–10

---

## `tracker` Section

All tracker settings come from the selected config (`default.yaml` or `configs/experiments/*.yaml`). At runtime, `tracker.py` generates a YAML for Ultralytics in `.runtime/trackers/` (directory is in `.gitignore`) — these files do not need to be edited manually.

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | str | `"bytetrack"` | Tracking algorithm: `"bytetrack"`, `"botsort"`, or `"botsort_reid"` |
| `track_activation_threshold` | float | `0.25` | Minimum confidence to activate a new track |
| `track_low_threshold` | float | `0.1` | Minimum confidence for the second matching stage (low-confidence detections) |
| `matching_cost_threshold` | float | `0.8` | IoU cost threshold for matching detections to tracks |
| `lost_track_buffer` | int / null | `null` | Frames before a lost track is deleted; `null` = auto-calculated |
| `auto_lost_track_buffer_seconds` | float | `3.0` | Seconds for auto-calculation: `round(fps × seconds)` |

**Mapping to Ultralytics YAML fields** (auto-generated in `.runtime/trackers/`):

| Config parameter | Ultralytics YAML field |
|-----------------|----------------------|
| `track_activation_threshold` | `track_high_thresh`, `new_track_thresh` |
| `track_low_threshold` | `track_low_thresh` |
| `lost_track_buffer` | `track_buffer` |
| `matching_cost_threshold` | `match_thresh` |

**`track_activation_threshold` impact:**
- Small value (0.1–0.2): new tracks created even on weak confidence → more false tracks
- Large value (0.5+): only confident detections activate a track → misses on partial occlusion

**`lost_track_buffer` impact:**
- Small value (10–20 frames): tracks deleted quickly on brief occlusions → frequent ID changes → inaccurate counting
- Large value (50–100 frames): tracks survive long occlusions → stable ID
- With `null`, the buffer is auto-calculated: `round(fps × auto_lost_track_buffer_seconds)`

**`matching_cost_threshold` impact:**
- High value (0.8+): accepts matching with low IoU → object keeps its ID after brief occlusion (e.g., passing behind a lamp post)
- Low value (0.4–0.5): requires high overlap for matching → brief occlusion causes ID change and new track creation

### ByteTrack (`algorithm: "bytetrack"`)

ByteTrack uses two-stage matching: first against high-confidence detections, then against low-confidence ones to "rescue" existing tracks.

All parameters are controlled by the common settings above — no additional parameters.

### BoT-SORT (`algorithm: "botsort"`)

BoT-SORT extends ByteTrack with two components: **Global Motion Compensation (GMC)** — corrects track positions when the camera moves — and optional **Re-ID** for re-identifying objects by appearance.

Additional parameters (beyond common ones):

| Config parameter | Default | Description |
|-----------------|---------|-------------|
| `fuse_score` | `true` | Include detection confidence in the IoU cost matrix |
| `gmc_method` | `"sparseOptFlow"` | GMC method: `sparseOptFlow` / `orb` / `ecc` / `none` |

**When to use BoT-SORT instead of ByteTrack:**
- Video shot with a moving camera (drone, PTZ) → GMC stabilizes tracks
- With a static camera, ByteTrack is faster and accurate enough

### BoT-SORT with Re-ID (`algorithm: "botsort_reid"`)

Same as BoT-SORT but with Re-ID enabled (`with_reid: true`). Allows recovering an object's ID by visual similarity even after a long absence from the frame.

Additional parameters compared to `botsort`:

| Config parameter | Default | Description |
|-----------------|---------|-------------|
| `reid_weights` | `"osnet_x0_25_market.pt"` | Re-ID model weights (downloaded automatically) |
| `proximity_threshold` | `0.5` | IoU threshold below which Re-ID features are used |
| `appearance_threshold` | `0.25` | Cosine distance threshold for Re-ID matching |

**Available Re-ID models:**

| Model | Size | Dataset | Notes |
|-------|------|---------|-------|
| `osnet_x0_25_market.pt` | ~1 MB | Market-1501 | Default; minimal FPS overhead |
| `osnet_x0_5_market.pt` | ~3 MB | Market-1501 | Speed/accuracy balance |
| `osnet_x1_0_market.pt` | ~11 MB | Market-1501 | High Re-ID accuracy |
| `osnet_x0_25_msmt17.pt` | ~1 MB | MSMT17 | More diverse dataset |
| `osnet_x1_0_msmt17.pt` | ~11 MB | MSMT17 | Maximum Re-ID accuracy |

> All models are downloaded automatically by Ultralytics on first run. Change the model via the `reid_weights` parameter in the config.

**Performance impact:**
- `osnet_x0_25_*` adds ~2–5 ms per frame on GPU; on CPU — ~10–30 ms
- If FPS is critical — use `"botsort"` without Re-ID

**When to use `botsort_reid`:**
- Objects frequently leave and re-enter the frame
- ID preservation is required during long occlusions
- Compute budget allows it

---

## `counter` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `true` | Enable/disable counting |
| `count_zone` | list / null | `null` | Counting zone polygon |
| `crossing_lines` | list / null | `null` | Lines for IN/OUT crossing counts |

**`count_zone` format:**
```yaml
count_zone:
  - [100, 200]   # x1, y1
  - [500, 200]   # x2, y2
  - [500, 400]   # x3, y3
  - [100, 400]   # x4, y4
```
Defines a rectangle or arbitrary polygon. Objects are counted only when their centroid is inside the zone. `null` — count all objects in the frame.

**`crossing_lines` format:**
```yaml
crossing_lines:
  - name: "Centre"
    points: [[640, 0], [640, 720]]
```
Direction **in** — object crossed the line from left to right (relative to the vector from `points[0]` to `points[1]`). Direction **out** — opposite.

---

## `output` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_video` | bool | `false` | Save annotated video |
| `output_dir` | str | `"outputs"` | Output directory |
| `video_name` | str | `"output.mp4"` | Output file name |
| `fps` | int | `30` | Output video FPS |

---

## `display` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_detections` | bool | `true` | Show raw detection bboxes |
| `show_tracking_ids` | bool | `true` | Show track bboxes with IDs |
| `show_counts` | bool | `true` | Show counter panel |
| `font_size` | float | `1.0` | Annotation font scale |

---

## Recommended Configurations

### Maximum performance (edge / low-end hardware)
```yaml
video:
  frame_width: 640
  frame_height: 480
detector:
  model: "models/yolov8n.pt"
  confidence_threshold: 0.5
  device: "cpu"
tracker:
  algorithm: "bytetrack"
display:
  show_detections: false
  show_tracking_ids: true
  show_counts: true
```

### Maximum accuracy (GPU server, static camera)
```yaml
video:
  frame_width: 1920
  frame_height: 1080
detector:
  model: "models/yolov8s.pt"
  confidence_threshold: 0.4
  device: "cuda"
tracker:
  algorithm: "bytetrack"
  track_activation_threshold: 0.3
  auto_lost_track_buffer_seconds: 5.0
  matching_cost_threshold: 0.7
```

### Moving camera (drone / PTZ)
```yaml
tracker:
  algorithm: "botsort"
  track_activation_threshold: 0.4
  auto_lost_track_buffer_seconds: 4.0
  matching_cost_threshold: 0.7
```
BoT-SORT with GMC (`sparseOptFlow`) compensates for frame shift and reduces track loss when the camera is moving.

### Robust tracking with Re-ID (occlusions, re-appearances)
```yaml
tracker:
  algorithm: "botsort_reid"
  track_activation_threshold: 0.4
  auto_lost_track_buffer_seconds: 5.0
  matching_cost_threshold: 0.7
```
BoT-SORT with Re-ID recovers an object's ID by appearance even after a long absence from the frame. The Re-ID model is downloaded automatically (`osnet_x0_25_market.pt`).

### Debug (all detections and tracks visible)
```yaml
detector:
  confidence_threshold: 0.3
tracker:
  track_activation_threshold: 0.2
display:
  show_detections: true
  show_tracking_ids: true
  show_counts: true
```
