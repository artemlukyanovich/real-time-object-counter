# Data Flow

---

## Overall Pipeline Diagram

```
┌─────────────┐    np.ndarray     ┌──────────────┐    List[Detection]  ┌─────────────┐
│ VideoSource │ ────────────────► │   Detector   │ ──────────────────► │   Tracker   │
└─────────────┘   (BGR, HxWx3)    └──────────────┘                     └─────────────┘
                                                                               │
                                                                    Dict[int, (bbox, class)]
                                                                               │
┌─────────────┐    np.ndarray     ┌──────────────┐    Dict[str, int]   ┌─────▼───────┐
│   Display   │ ◄──────────────── │   Renderer   │ ◄────────────────── │   Counter   │
└─────────────┘   (annotations)   └──────────────┘                     └─────────────┘
```

---

## Stage 1: VideoSource → frame

**What happens:**
- `cv2.VideoCapture.read()` captures a raw frame, `cv2.resize()` normalizes the resolution
- The `frame_count` counter is incremented
- With `async_pipeline: true` (default) the capture+resize is done by a background producer thread filling a queue; `read()` pulls a ready frame from the queue (decoding runs in parallel with inference). The main loop measures the frame-wait time (`record_io_wait`)

**Output format:**
```python
frame: np.ndarray  # shape: (height, width, 3), dtype: uint8, BGR
# Example: (720, 1280, 3)
```

**Critical points:**
- On `read()` → `(False, None)` — end of the video file or camera loss; the loop terminates
- Resizing runs every frame — on weak CPUs it can cause an FPS drop at high resolutions
- In async mode, `drop_policy` determines behavior under overload: `block` (lossless — files) or `drop` (freshest frame — live stream). See `docs/async_pipeline.md`

---

## Stage 2: Detector → list of detections

**What happens:**
- YOLOv8 runs inference on the frame
- Filtering by `confidence_threshold`
- For each detected object: extraction of the bbox in `xyxy` format, the class name, and confidence

**Output format:**
```python
detections: List[Tuple[Tuple[int, int, int, int], str, float]]
# [
#   ((120, 45, 280, 310), "person", 0.87),
#   ((400, 200, 650, 450), "car", 0.92),
# ]
```

- `(x1, y1, x2, y2)` — absolute pixel coordinates (left-top, right-bottom)
- `class_name` — a string from the COCO dictionary (80 classes)
- `confidence` — float [0.0, 1.0]

**Critical points:**
- If the frame came in empty (`None`) — `detect()` may crash; the guard is on the `main.py` side
- If `confidence_threshold` is too high → real objects are missed; too low → noise detections clutter the tracker
- YOLOv8n inference time on GPU: ~5–15 ms; on CPU: ~50–200 ms

---

## Stage 3: Tracker → object tracks

**What happens:**
1. Centroids of all new detections are computed: `cx = (x1+x2)/2`, `cy = (y1+y2)/2`
2. For each existing track, the nearest detection is found (Euclidean distance)
3. Matching: if `distance < max_distance` — the track is updated
4. Unmatched detections are registered as new tracks with a new `track_id`
5. Tracks without matches: `disappeared[id] += 1`; if `disappeared[id] > max_disappeared` — the track is removed

**Output format:**
```python
tracked_objects: Dict[int, Tuple[Tuple[int, int, int, int], str]]
# {
#   0: ((120, 45, 280, 310), "person"),
#   1: ((400, 200, 650, 450), "car"),
#   3: ((50, 100, 180, 250), "person"),
# }
```

- Key — the unique `track_id` (a monotonically increasing int)
- Value — the object's current bbox and class

**Critical points:**

- **Track loss:** on a sharp object movement the centroid shifts by more than `max_distance` pixels → a new ID; the old one is removed after `max_disappeared` frames. Affects counting accuracy.
- **ID switch on occlusion:** two objects intersecting in the frame may swap IDs — the algorithm doesn't distinguish them until they separate.
- **Object class:** taken from the last detection; if the detector starts classifying the object differently (unstable predictions), the class in the track will change.
- **The `max_distance` parameter** must be calibrated to the real speed of objects and the FPS. At 30 FPS an object moving at 100 px/s shifts ~3 px/frame — `max_distance=50` is plenty. At 10 FPS — already ~10 px/frame.

---

## Stage 4: Counter → per-class counters

**What happens:**
1. For each `track_id` in `tracked_objects` it checks: has it already been counted (`id in counted_ids`)?
2. If not — the bbox centroid is computed
3. If `count_zone` is set — the centroid's membership in the polygon is checked (ray casting)
4. On passing the check: `counted_ids.add(track_id)`, `class_counts[class_name] += 1`, `total_count += 1`

**Output format:**
```python
counts: Dict[str, int]
# {"person": 5, "car": 2, "bicycle": 1}
```

**Critical points:**

- **Double counting is impossible** — `counted_ids` is a `Set`, each ID enters exactly once.
- **Track loss → lost count:** if the tracker lost the object before it entered the zone, no count happens.
- **ID reassignment:** if an object left the frame and returned with a new `track_id` — it will be counted again. This is a known limitation of a centroid tracker without ReID.
- **count_zone = None:** without a zone, every new track is counted immediately on first appearance.

---

## Stage 5: Renderer → annotated frame

**What happens:**
- Conditional rendering based on config flags (`show_detections`, `show_tracking_ids`, `show_counts`)
- In-place frame modification (all operations write into the passed `np.ndarray`)
- Final `cv2.imshow()` — display in a window

**Drawing order:**
1. `render_detections` — thin detection boxes (if enabled)
2. `render_tracks` — bold boxes with IDs (if enabled)
3. `render_counts` — panel in the top-left corner (if enabled)
4. `render_fps` — FPS in the top-right corner (always)

**Critical points:**
- The Renderer does not copy the frame — it writes over the original. If a clean frame is needed for recording/analysis, the copy must be made before calling the Renderer.
- `cv2.addWeighted` for the semi-transparent counters panel background is a light operation, but runs every frame.

---

## Performance Impact

| Factor | Effect on FPS | Effect on accuracy |
|--------|---------------|-------------------|
| `detector.confidence_threshold` | None | High → misses; low → false tracks |
| `detector.model` (yolov8n vs yolov8s) | Significant | n is faster, s is more accurate |
| `video.frame_width/height` | Moderate | High resolution → better detection of small objects |
| `tracker.max_distance` | None | Large → object mixing; small → frequent losses |
| `tracker.max_disappeared` | None | Large → stable tracks; small → premature removal |
| `display.show_detections` | Negligible | None |
| `display.show_tracking_ids` | Negligible | None |

---

## Stability Impact

**Main sources of instability:**
1. **Unstable YOLO inference** — an object is detected, then not → bbox flicker, potential ID switch
2. **High object speed** relative to FPS → exceeding `max_distance`, track loss
3. **Occlusions** — objects overlap each other → incorrect centroid matching
4. **Object class change** — YOLO classifies an object unstably near the confidence boundary → `class_counts` may end up with counters of different classes for one physical object

**Recommendations to improve stability:**
- Use `max_disappeared = 30–50` frames (a buffer for temporary occlusions)
- Don't lower `confidence_threshold` below 0.4 without need
- For fast objects, increase `max_distance` or lower the resolution to raise FPS
