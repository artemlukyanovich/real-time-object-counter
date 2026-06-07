# Configuration

## Configuration Files

| File | Purpose |
|------|-----------|
| `configs/default.yaml` | Main default config |
| `configs/experiments/*.yaml` | Experiment configs (same structure as `default.yaml`) |
| `configs/embeddings/default.yaml` | CLIP embedding and ReID memory parameters (see [`reid` section](#reid-section)) |
| `configs/embeddings/experiments/*.yaml` | Experiments with embedding models |

Loaded by the `Config` class (`src/config.py`). Parameters are accessed via dot-notation: `config.get("detector.confidence_threshold")`.

**Run with a custom config:**
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
  async_pipeline: true
  drop_policy: "auto"

detector:
  model: "models/yolov8n.pt"
  confidence_threshold: 0.5
  device: "cuda"
  half: false
  detect_interval: 1
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

reid:
  enabled: false
  embeddings_config: "configs/embeddings/default.yaml"
  update_interval: 3
  min_track_age: 1

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
  show_object_ids: true   # only when reid.enabled: true
  show_reid_stats: true   # only when reid.enabled: true
```

---

## `video` Section

| Parameter | Type | Default | Description |
|----------|-----|-------------|----------|
| `source` | int / str | `0` | Source: `0` — first camera, `1` — second, a file path — video |
| `fps` | int / null | `null` | `null` — read FPS from the source; an explicit value overrides it |
| `fallback_fps` | int | `30` | Default FPS if the source reports `0` or nothing |
| `frame_width` | int | `1280` | Frame width after resizing (pixels) |
| `frame_height` | int | `720` | Frame height after resizing (pixels) |
| `async_pipeline` | bool | `true` | Decode frames in a background thread in parallel with inference. `false` = synchronous read. See `docs/async_pipeline.md` |
| `drop_policy` | str | `"auto"` | Frame-drop policy in async mode: `auto` \| `block` \| `drop` |

**Performance impact:**
- Lowering the resolution to 640×480 can double the FPS with CPU inference
- Increasing the resolution improves detection of small objects but increases inference and resize time

**Effect of `async_pipeline` (asynchronous video pipeline):**

Decoding (`cap.read` + resize) is moved into a background producer thread that pre-fills a frame queue while the main loop is busy with inference. The frame time drops from `decode + inference` to `max(decode, inference)`. Especially useful together with `detector.detect_interval` (on skipped frames, decoding easily becomes the bottleneck).

- `true` — recommended (default). `false` — synchronous single-threaded read, as before.
- The flag is kept intentionally: for A/B-measuring the gain on a single video and as a fallback during debugging.
- At startup the mode is visible in the log: `Video: async pipeline ON | drop_policy=auto (block/no-drop)`.

**Effect of `drop_policy` (only when `async_pipeline: true`):**

What to do when inference can't keep up with the frame stream:

| Value | Behavior | When |
|----------|-----------|-------|
| `block` | the producer waits for space in the queue (backpressure); no frame is lost, deterministic | video file |
| `drop` | queue size 1, when full the old frame is discarded → the freshest is processed; low latency | live stream (camera/drone) |
| `auto` | `block` for files, `drop` for live sources | default |

> On a file, `drop` silently loses frames and breaks reproducibility of counting — that's why `auto` uses `block`. Explicit `drop` is needed for load testing: running a file "as a live stream" on weak hardware. See `docs/async_pipeline.md` for details.

**Estimating the gain:** the `avg_io_wait_ms` metric in the final summary — the main loop's idle time waiting for a frame. In sync ≈ decode time (on the critical path); in async it drops toward ~0 if decoding is faster than inference.

---

## `detector` Section

| Parameter | Type | Default | Description |
|----------|-----|-------------|----------|
| `model` | str | `"yolov8n.pt"` | Name or path to the YOLOv8 weights file |
| `confidence_threshold` | float | `0.5` | Minimum detection confidence [0.0–1.0] |
| `device` | str | `"cuda"` | Inference device: `"cuda"` or `"cpu"` |
| `half` | bool | `false` | FP16 (half precision) inference. Applied only to `.pt` models on CUDA |
| `detect_interval` | int | `1` | Frame skipping: the full detect+track pass runs once every N frames. `1` = off. See below and `docs/frame_skipping.md` |
| `allowed_classes` | list / null | `null` | List of classes for detection/tracking. `null` = all classes. Example: `["person"]` or `["person", "car", "truck"]` |

**Class filtering (`allowed_classes`):**

Passed directly to Ultralytics as `classes=[id, ...]` — filtering happens at the model level before the tracker, so it does not affect tracking speed. Class names correspond to the COCO dataset labels (for standard YOLOv8/YOLO11 models).

```yaml
# Count only people:
detector:
  allowed_classes: ["person"]

# Count vehicles:
detector:
  allowed_classes: ["car", "truck", "bus", "motorcycle"]
```

If a specified class is not found in the model — a warning with the list of available classes is printed, and that class is ignored.

**Model selection:**

Ultralytics automatically downloads the weights on the first run if the file is not found locally.

| Model | Speed (GPU) | Accuracy | Note |
|--------|---------------|---------|-----------|
| `yolov8n.pt` | ~5–10 ms | Baseline | Recommended for real time |
| `yolov8s.pt` | ~10–20 ms | Medium | Speed/accuracy trade-off |
| `yolov8m.pt` | ~20–40 ms | High | Requires a powerful GPU |
| `yolov8l.pt` | ~30–60 ms | Very high | For server-side inference |
| `yolov8x.pt` | ~50–100 ms | Maximum | The most accurate YOLOv8 model |
| `yolo11n.pt` | ~4–8 ms | Baseline | Newer architecture, faster than v8n |
| `yolo11s.pt` | ~8–15 ms | Medium | New architecture, trade-off |
| `yolo11m.pt` | ~15–30 ms | High | New architecture, high accuracy |

> The `yolo11*` models (YOLO11) are Ultralytics' current generation; all else being equal, preferable over `yolov8*`.

**Effect of `confidence_threshold`:**
- `0.3–0.4` — more detections, more false positives → clutters the tracker
- `0.5` — balanced (default)
- `0.7+` — only confident detections; possible misses with partial occlusion of objects

**Effect of `device`:**
- `"cuda"` — GPU inference (~5–15 ms for yolov8n)
- `"cpu"` — CPU inference (~50–200 ms), FPS drops to 5–10

**Effect of `half` (FP16):**

Running the network in 16-bit precision instead of 32-bit. On a GPU with Tensor Cores it gives ~1.5–2× inference speedup and half the VRAM usage with negligible detection accuracy loss.

- Applied **only** to `.pt` models on CUDA. The value is resolved in `ObjectDetector._resolve_half` and passed as a single flag into `model.track(half=...)`.
- On CPU the flag is ignored (FP16 on CPU is unsupported / slower); a warning is printed.
- For `.onnx` / `.engine` the flag is also ignored: their precision is fixed at export time (`export_model.py --half`) — see `docs/export_optimization.md`.
- Overridden at launch by the `--half` flag (takes priority over `detector.half` in the config):

```bash
python -m src.main --source data/input/video.mp4 --half
```

At startup the actually applied mode is visible in the log: `Detector backend: pytorch | device: cuda | half: True`.

> **The gain depends on model size.** FP16 speeds up compute-bound load (large models / large `imgsz`). Small models bottleneck on overhead (preprocessing, NMS, kernel launches) which FP16 does not touch. Measurements on this project (full pipeline, `data/input/movie.mp4`, RTX, ~300 frames):
>
> | Model | Detection FP32 | Detection FP16 | Speedup | FPS FP32 → FP16 |
> |--------|--------------|--------------|-----------|------------------|
> | `yolov8n` | ~7.0 ms | ~6.8 ms | ~1.0× (within noise) | 55 → 56 |
> | `yolov8m` | ~16.5 ms | ~10.8 ms | ~1.5× | 39 → 51 |
>
> Conclusion: on `yolov8n` keep `half: false` (no gain); enable FP16 when switching to `yolov8m`/`l`/`x` or increasing `imgsz`. If detection is not the bottleneck (ReID/render/IO take a significant share of the frame), the effect on the final FPS will be smaller than the speedup of detection itself.

**Effect of `detect_interval` (frame skipping):**

The heavy detect+track pass (`model.track`) runs only on every Nth frame; on intermediate frames the track boxes are **linearly extrapolated** from the velocity estimated between the last two inference frames. Counting (`counter.update`) and ReID run only on inference frames; render and display — on every frame.

- `1` — off, detection on every frame (default behavior, full backward compatibility).
- `2` — inference load ~−50%, `3` — ~−66%.
- We pay with tracking accuracy: at large N there's a higher risk of ID switching on fast motion/occlusions (Kalman predicts over a larger step, less overlap of neighboring observations).
- `lost_track_buffer` is automatically divided by N (see the `tracker` section) so the lost-track lifetime window stays constant **in seconds**.
- When `detect_interval > 1`, a second line `infer:` appears in the FPS overlay — the actual model rate (inferences/sec), separate from throughput (displayed frames/sec).

A full description of the logic and decisions is in `docs/frame_skipping.md`.

---

## `tracker` Section

All tracker settings are taken from the selected config (`default.yaml` or `configs/experiments/*.yaml`). At startup `tracker.py` generates a YAML for Ultralytics in `.runtime/trackers/` (folder in `.gitignore`) — there's no need to edit those files.

### Common parameters

| Parameter | Type | Default | Description |
|----------|-----|-------------|----------|
| `algorithm` | str | `"bytetrack"` | Tracking algorithm: `"bytetrack"`, `"botsort"`, or `"botsort_reid"` |
| `track_activation_threshold` | float | `0.25` | Minimum confidence to activate a new track |
| `track_low_threshold` | float | `0.1` | Minimum confidence for the second matching stage (low-confidence detections) |
| `matching_cost_threshold` | float | `0.8` | IoU cost threshold for matching detections to tracks |
| `lost_track_buffer` | int / null | `null` | Frames until a lost track is removed; `null` = auto-computed |
| `auto_lost_track_buffer_seconds` | float | `3.0` | Seconds for auto-computing the buffer: `round(fps × seconds)` |

**Mapping to Ultralytics YAML fields** (generated automatically in `.runtime/trackers/`):

| Config parameter | Ultralytics YAML field |
|-----------------|-------------------------|
| `track_activation_threshold` | `track_high_thresh`, `new_track_thresh` |
| `track_low_threshold` | `track_low_thresh` |
| `lost_track_buffer` | `track_buffer` |
| `matching_cost_threshold` | `match_thresh` |

**Effect of `track_activation_threshold`:**
- Low value (0.1–0.2): new tracks are created even at weak confidence → more false tracks
- High value (0.5+): only confident detections activate a track → misses with partial occlusion

**Effect of `lost_track_buffer`:**
- Low value (10–20 frames): tracks are removed quickly on brief occlusions → frequent ID switching → inaccurate counting
- High value (50–100 frames): tracks are retained through long occlusions → stable ID
- With `null` the buffer is computed automatically: `round(fps × auto_lost_track_buffer_seconds)`
- The value is interpreted in frames of the source video. With `detector.detect_interval > 1` it is divided by N (`max(1, round(buffer / N))`) — both auto- and explicitly set. The tracker ticks only on inference frames, so the division keeps the buffer's actual duration in seconds constant. Without it, the lost-track lifetime window would inflate by N (e.g., a set 3 s would become 9 s at N=3). See `docs/frame_skipping.md`

**Effect of `matching_cost_threshold`:**
- High value (0.8+): a match with low IoU is accepted → an object keeps its ID after brief overlap (e.g., passing behind a lamppost)
- Low value (0.4–0.5): a high overlap is required for a match → a brief occlusion leads to an ID switch and a new track

### ByteTrack (`algorithm: "bytetrack"`)

ByteTrack uses two-stage matching: first by high-confidence detections, then by low-confidence ones to "re-pick up" already existing tracks.

All parameters are controlled via the common settings above — there are no additional parameters.

### BoT-SORT (`algorithm: "botsort"`)

BoT-SORT extends ByteTrack with two components: **Global Motion Compensation (GMC)** — correction of track positions when the camera moves — and an optional **Re-ID** for re-identifying objects by appearance.

Additional parameters (beyond the common ones):

| Config parameter | Default | Description |
|-----------------|-------------|----------|
| `fuse_score` | `true` | Account for detection confidence in the IoU cost matrix |
| `gmc_method` | `"sparseOptFlow"` | GMC method: `sparseOptFlow` / `orb` / `ecc` / `none` |

**When to use BoT-SORT instead of ByteTrack:**
- The video is shot with a moving camera (drone, PTZ) → GMC stabilizes the tracks
- With a static camera, ByteTrack is faster and accurate enough

### BoT-SORT with Re-ID (`algorithm: "botsort_reid"`)

The same BoT-SORT, but with the Re-ID model enabled (`with_reid: true`). It allows recovering an object's ID by visual similarity even after a long disappearance from the frame.

Additional parameters compared to `botsort`:

| Config parameter | Default | Description |
|-----------------|-------------|----------|
| `reid_weights` | `"osnet_x0_25_market.pt"` | Re-ID model weights (downloaded automatically) |
| `proximity_threshold` | `0.5` | IoU threshold below which Re-ID features are engaged |
| `appearance_threshold` | `0.25` | Cosine distance threshold for Re-ID matching |

**Available Re-ID models:**

| Model | Size | Dataset | Note |
|--------|--------|---------|-----------|
| `osnet_x0_25_market.pt` | ~1 MB | Market-1501 | Default; minimal FPS load |
| `osnet_x0_5_market.pt` | ~3 MB | Market-1501 | Speed/accuracy balance |
| `osnet_x1_0_market.pt` | ~11 MB | Market-1501 | High Re-ID accuracy |
| `osnet_x0_25_msmt17.pt` | ~1 MB | MSMT17 | A more diverse dataset |
| `osnet_x1_0_msmt17.pt` | ~11 MB | MSMT17 | Maximum Re-ID accuracy |

> All models are downloaded by Ultralytics automatically on the first run. You can change the model via the `reid_weights` parameter in the config.

**Performance impact:**
- `osnet_x0_25_*` adds ~2–5 ms per frame on GPU; on CPU — ~10–30 ms
- If FPS is critical — use `"botsort"` without Re-ID

**When to use `botsort_reid`:**
- Objects frequently leave the frame for a long time and return
- ID preservation through long occlusions is required
- There is spare compute headroom

---

## `counter` Section

| Parameter | Type | Default | Description |
|----------|-----|-------------|----------|
| `enable` | bool | `true` | Enable/disable counting |
| `count_zone` | list / null | `null` | Counting-zone polygon |
| `crossing_lines` | list / null | `null` | Lines for counting IN/OUT crossings |

**`count_zone` format:**
```yaml
count_zone:
  - [100, 200]   # x1, y1
  - [500, 200]   # x2, y2
  - [500, 400]   # x3, y3
  - [100, 400]   # x4, y4
```
Defines a rectangle or an arbitrary polygon. Objects are counted only when their centroid is inside the zone. `null` — count all objects in the frame.

**`crossing_lines` format:**
```yaml
crossing_lines:
  - name: "Centre"
    points: [[640, 0], [640, 720]]
```
Direction **in** — the object crossed the line left to right (relative to the vector from `points[0]` to `points[1]`). Direction **out** — the opposite.

---

## `reid` Section

CLIP-based Re-Identification pipeline — assigns each object a persistent `object_id` that survives track loss, occlusion, and reappearance. Works on top of the standard tracker.

| Parameter | Type | Default | Description |
|----------|-----|-------------|----------|
| `enabled` | bool | `false` | Enable the CLIP ReID pipeline |
| `embeddings_config` | str | `"configs/embeddings/default.yaml"` | Path to the embeddings config (model, memory, cropper) |
| `update_interval` | int | `3` | Run the embedding pipeline every N frames. `1` = every frame |
| `min_track_age` | int | `1` | Minimum track frames before registering a new object in memory. `1` = instant |

**`update_interval`** — reduces GPU/CPU load. On frames between updates the last known `object_id`s are reused. At 30 FPS a value of `3` gives ~10 embedding passes per second — enough for stable ReID.

**`min_track_age`** — filters out "flickering" objects (false detections lasting 1–5 frames). Counted in real frames by `frame_idx` **independently of `update_interval`**. A track matched to an already-known object by embedding is matched instantly regardless of this threshold.

| `min_track_age` | Behavior |
|---|---|
| `1` | Disabled — every track is registered immediately |
| `8` | At 30 FPS ≈ 0.25 sec — cuts off instantaneous false detections |
| `30` | 1 second — only stable objects |

**The embeddings config** (`configs/embeddings/default.yaml`) contains the model and memory parameters:

| Parameter | Default | Description |
|----------|-------------|----------|
| `embedder.model_name` | `"ViT-B-32"` | OpenCLIP model. Options: `ViT-B-32`, `ViT-B-16`, `ViT-L-14`, `ViT-H-14` |
| `embedder.pretrained` | `"laion2b_s34b_b79k"` | Weights tag (must match the model) |
| `embedder.device` | `"cuda"` | Device for embeddings: `"cuda"` or `"cpu"` |
| `cropper.padding` | `8` | Padding in pixels around the bbox when cropping |
| `cropper.save_crops` | `false` | Save crops to `outputs/crops/` for visual inspection |
| `memory.similarity_threshold` | `0.75` | Cosine similarity threshold for a Re-ID match [0.0–1.0] |
| `memory.max_missing_frames` | `90` | Frames without detection until an object is deactivated (~3 sec at 30 FPS) |
| `memory.max_embeddings_per_object` | `5` | Rolling buffer of embeddings per object |

See [docs/embeddings_reid.md](embeddings_reid.md) for details.

---

## `output` Section

| Parameter | Type | Default | Description |
|----------|-----|-------------|----------|
| `save_video` | bool | `false` | Save the annotated video |
| `output_dir` | str | `"outputs"` | Directory for output files |
| `video_name` | str | `"output.mp4"` | Output file name |
| `fps` | int | `30` | Output video FPS |

---

## `display` Section

| Parameter | Type | Default | Description |
|----------|-----|-------------|----------|
| `show_detections` | bool | `true` | Display raw detection bboxes |
| `show_tracking_ids` | bool | `true` | Display track bboxes with IDs |
| `show_counts` | bool | `true` | Display the counters panel |
| `font_size` | float | `1.0` | Annotation font scale |
| `show_object_ids` | bool | `true` | Show the persistent `object_id` on the bbox instead of `track_id` (only when `reid.enabled: true`) |
| `show_reid_stats` | bool | `true` | "ReID unique / active" panel in the top-right corner (only when `reid.enabled: true`) |

With `show_object_ids: true` the label format on the bounding box is: `#N class [tM]`, where `N` is the persistent `object_id`, `class` is the class, and `tM` is the tracker's current `track_id`. The bbox color is determined by `object_id` and does not change when `track_id` changes.

---

## Recommended Configurations

### Maximum performance (edge/weak hardware)
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
BoT-SORT with GMC (`sparseOptFlow`) compensates for frame shift and reduces the number of lost tracks when the camera moves.

### Stable tracking with Re-ID (occlusions, reappearances)
```yaml
tracker:
  algorithm: "botsort_reid"
  track_activation_threshold: 0.4
  auto_lost_track_buffer_seconds: 5.0
  matching_cost_threshold: 0.7
```
BoT-SORT with Re-ID recovers an object's ID by appearance even after a long absence from the frame. The ReID model downloads automatically (`osnet_x0_25_market.pt`).

### Stable counting with CLIP ReID (persistent IDs, flicker filter)
```yaml
reid:
  enabled: true
  embeddings_config: "configs/embeddings/default.yaml"
  update_interval: 3   # every 3 frames at 30 FPS
  min_track_age: 8     # ≈0.25 sec — don't count objects that flickered for 1–7 frames
display:
  show_object_ids: true
  show_reid_stats: true
```
CLIP ReID assigns an object a persistent `#N` that doesn't change on track loss. `min_track_age: 8` excludes false detections that didn't have time to stabilize.

### Debugging (all detections and tracks visible)
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
