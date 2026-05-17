# Data Flow

---

## Pipeline Overview

```
┌─────────────┐    np.ndarray     ┌──────────────┐    List[Detection]  ┌─────────────┐
│ VideoSource │ ────────────────► │   Detector   │ ──────────────────► │   Tracker   │
└─────────────┘   (BGR, HxWx3)    └──────────────┘                     └─────────────┘
                                                                               │
                                                                    Dict[int, (bbox, class)]
                                                                               │
┌─────────────┐    np.ndarray     ┌──────────────┐    Dict[str, int]   ┌─────▼───────┐
│   Display   │ ◄──────────────── │   Renderer   │ ◄────────────────── │   Counter   │
└─────────────┘   (annotated)     └──────────────┘                     └─────────────┘
```

---

## Stage 1: VideoSource → frame

**What happens:**
- `cv2.VideoCapture.read()` captures a raw frame
- If source resolution ≠ configured resolution — `cv2.resize()` is applied
- `frame_count` counter is incremented

**Output format:**
```python
frame: np.ndarray  # shape: (height, width, 3), dtype: uint8, BGR
# Example: (720, 1280, 3)
```

**Critical points:**
- When `read()` returns `(False, None)` — end of video file or camera loss; loop terminates
- Resize runs every frame — on weak CPUs this can cause FPS drops at high resolutions

---

## Stage 2: Detector → list of detections

**What happens:**
- YOLOv8 runs inference on the frame
- Filtering by `confidence_threshold`
- For each detected object: bbox in `xyxy` format, class name, confidence

**Output format:**
```python
detections: List[Tuple[Tuple[int, int, int, int], str, float]]
# [
#   ((120, 45, 280, 310), "person", 0.87),
#   ((400, 200, 650, 450), "car", 0.92),
# ]
```

- `(x1, y1, x2, y2)` — absolute pixel coordinates (left-top, right-bottom)
- `class_name` — string from COCO dictionary (80 classes)
- `confidence` — float [0.0, 1.0]

**Critical points:**
- If an empty frame (`None`) arrives — `detect()` may crash; guarded in `main.py`
- Too high `confidence_threshold` → missed real objects; too low → noisy detections pollute the tracker
- YOLOv8n inference time on GPU: ~5–15 ms; on CPU: ~50–200 ms

---

## Stage 3: Tracker → object tracks

**What happens:**
1. Centroids of all new detections are computed: `cx = (x1+x2)/2`, `cy = (y1+y2)/2`
2. For each existing track, the nearest detection is found (Euclidean distance)
3. Matching: if `distance < max_distance` — track is updated
4. Unmatched detections are registered as new tracks with a new `track_id`
5. Tracks without matches: `disappeared[id] += 1`; if `disappeared[id] > max_disappeared` — track is deleted

**Output format:**
```python
tracked_objects: Dict[int, Tuple[Tuple[int, int, int, int], str]]
# {
#   0: ((120, 45, 280, 310), "person"),
#   1: ((400, 200, 650, 450), "car"),
#   3: ((50, 100, 180, 250), "person"),
# }
```

- Key — unique `track_id` (monotonically increasing int)
- Value — current bbox and object class

**Critical points:**

- **Track loss:** if an object moves sharply more than `max_distance` pixels — a new ID is assigned; the old one is deleted after `max_disappeared` frames. Affects counting accuracy.
- **ID swap during occlusion:** two objects crossing each other in the frame may swap IDs — the algorithm cannot distinguish them until they separate.
- **Object class:** taken from the latest detection; if the detector starts classifying an object differently (unstable predictions), the class in the track will change.
- **`max_distance`** should be calibrated for the actual object speed and FPS. At 30 FPS, an object moving at 100 px/s shifts ~3 px/frame — `max_distance=50` is comfortable. At 10 FPS — already ~10 px/frame.

---

## Stage 4: Counter → per-class counts

**What happens:**
1. For each `track_id` in `tracked_objects`, check: has it already been counted (`id in counted_ids`)?
2. If not — compute the bbox centroid
3. If `count_zone` is set — check whether the centroid is inside the polygon (ray casting)
4. On pass: `counted_ids.add(track_id)`, `class_counts[class_name] += 1`, `total_count += 1`

**Output format:**
```python
counts: Dict[str, int]
# {"person": 5, "car": 2, "bicycle": 1}
```

**Critical points:**

- **Double counting is impossible** — `counted_ids` is a `Set`; each ID enters exactly once.
- **Track loss → missed count:** if the tracker loses an object before it enters the zone, it will not be counted.
- **ID reassignment:** if an object leaves the frame and returns with a new `track_id` — it will be counted again. This is a known limitation of a centroid tracker without Re-ID.
- **count_zone = None:** without a zone, every new track is counted immediately on its first appearance.

---

## Stage 5: Renderer → annotated frame

**What happens:**
- Conditional rendering based on config flags (`show_detections`, `show_tracking_ids`, `show_counts`)
- Frame is modified in-place (all operations write to the passed `np.ndarray`)
- Final `cv2.imshow()` — display in window

**Rendering order:**
1. `render_detections` — thin detection boxes (if enabled)
2. `render_tracks` — thick boxes with IDs (if enabled)
3. `render_counts` — panel in the top-left corner (if enabled)
4. `render_fps` — FPS in the top-right corner (always)

**Critical points:**
- Renderer does not copy the frame — writes over the original. If a clean frame is needed for recording/analysis, copy it before calling Renderer.
- `cv2.addWeighted` for the semi-transparent counter panel background — lightweight operation, but runs every frame.

---

## Performance Impact

| Factor | FPS impact | Accuracy impact |
|--------|-----------|----------------|
| `detector.confidence_threshold` | None | High → missed detections; low → false tracks |
| `detector.model` (yolov8n vs yolov8s) | Significant | n is faster, s is more accurate |
| `video.frame_width/height` | Moderate | Higher resolution → better detection of small objects |
| `tracker.max_distance` | None | Large → track mixing; small → frequent losses |
| `tracker.max_disappeared` | None | Large → stable tracks; small → premature deletion |
| `display.show_detections` | Negligible | None |
| `display.show_tracking_ids` | Negligible | None |

---

## Stability

**Main sources of instability:**
1. **Unstable YOLO inference** — object is detected intermittently → flickering bbox, potential ID change
2. **High object speed** relative to FPS → exceeds `max_distance`, track lost
3. **Occlusions** — objects overlap each other → incorrect centroid matching
4. **Object class change** — YOLO unstably classifies an object at the confidence boundary → `class_counts` may accumulate counters for different classes for the same physical object

**Recommendations for improved stability:**
- Use `max_disappeared = 30–50` frames (buffer for temporary occlusions)
- Do not lower `confidence_threshold` below 0.4 without reason
- For fast-moving objects, increase `max_distance` or reduce resolution to improve FPS
