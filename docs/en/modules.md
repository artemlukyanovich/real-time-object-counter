# Module Reference

---

## `src/main.py` — Orchestrator

**Responsibility:** application entry point, coordination of all pipeline components, main frame processing loop.

**Main classes:**

### `ObjectCounterApp`

| Method | Description |
|--------|-------------|
| `__init__(config_path, source_override)` | Loads config, creates all components |
| `_initialize_components()` | Instantiates VideoSource, Detector, Tracker, Counter, Renderer, Metrics |
| `run()` | Main loop: read frame → process → display |
| `_render_frame(frame, detections, tracked_objects, counts, fps)` | Conditional rendering based on config flags |
| `cleanup()` | Releases resources, prints final metrics |

**Inputs:** `config_path: str`, `source: int | str` (from CLI)

**Outputs:** OpenCV window display, final metrics printed to stdout

**Role in pipeline:** the only module that knows about all others. Passes data between components sequentially on each loop iteration.

---

## `src/video_source.py` — Video Source

**Responsibility:** frame capture from webcam or video file, resolution normalization.

**Main classes:**

### `VideoSource`

| Method | Description |
|--------|-------------|
| `__init__(source, width, height)` | Opens cv2.VideoCapture, sets resolution |
| `read()` | Returns `(bool, frame)`, resizes to configured resolution |
| `release()` | Closes capture |
| `get_fps()` | Source FPS (from video metadata or camera) |
| `get_frame_count()` | Current frame number |

**Inputs:**
- `source`: `int` (camera index) or `str` (file path)
- `width`, `height`: target resolution

**Outputs:** `Tuple[bool, Optional[np.ndarray]]` — success flag and frame in BGR format

**Role in pipeline:** first stage. Supplies raw frames to the detector.

---

## `src/detector.py` — Object Detector

**Responsibility:** YOLOv8 inference on a frame, conversion of raw model output to structured format.

**Main classes:**

### `ObjectDetector`

| Method | Description |
|--------|-------------|
| `__init__(model_name, confidence_threshold, device)` | Loads YOLO weights, configures device |
| `detect(frame)` | Runs inference, returns list of detections |
| `get_class_names()` | `{id: class_name}` dictionary from the model |

**Inputs:** `frame: np.ndarray` (BGR, HxWx3)

**Outputs:**
```python
List[Tuple[Tuple[int, int, int, int], str, float]]
# [((x1, y1, x2, y2), class_name, confidence), ...]
```

**Role in pipeline:** second stage. Converts a raw frame into a list of detected objects with coordinates and class.

**Note:** uses `box.xyxy[0].int().tolist()` — absolute pixel coordinates in `(left, top, right, bottom)` format.

---

## `src/tracker.py` — Object Tracker

**Responsibility:** matching detections across frames, assigning and maintaining unique object IDs.

**Main classes:**

### `CentroidTracker`

| Method | Description |
|--------|-------------|
| `__init__(max_disappeared, max_distance)` | Initializes tracker state |
| `update(detections)` | Matches new detections against existing tracks |
| `get_tracks()` | Current track positions `{id: (cx, cy)}` |
| `reset()` | Resets all state |

**`update` algorithm:**
1. Compute the centroid of each detection
2. For each existing track, find the nearest detection (Euclidean distance)
3. If distance < `max_distance` — update the track
4. Unmatched detections → new tracks with incremented IDs
5. Tracks without matches → increment `disappeared`; if `disappeared > max_disappeared` — delete

**Inputs:** `List[((x1,y1,x2,y2), class_name, confidence)]`

**Outputs:**
```python
Dict[int, Tuple[Tuple[int, int, int, int], str]]
# {track_id: ((x1, y1, x2, y2), class_name)}
```

**Role in pipeline:** third stage. Turns independent detections into continuous object tracks.

**Architectural limitation:** no occlusion handling — objects overlapping each other may lose their ID.

---

## `src/counter.py` — Object Counter

**Responsibility:** unique object accounting, optional filtering by counting zone.

**Main classes:**

### `ObjectCounter`

| Method | Description |
|--------|-------------|
| `__init__(count_zone=None)` | Initializes counters; `count_zone` — polygon `List[Tuple[int, int]]` |
| `update(tracked_objects)` | Updates counters based on current tracks |
| `get_counts()` | `{class_name: count}` |
| `get_total_count()` | Total count |
| `get_class_count(class_name)` | Counter for a specific class |
| `set_count_zone(zone)` | Dynamically set counting zone |
| `reset()` | Reset all counters |

**Counting logic:** each `track_id` is added to `counted_ids` exactly once. Re-appearing in the frame is not recounted. If `count_zone` is set, an object is only counted when its centroid is inside the polygon.

**Inputs:** `Dict[int, Tuple[Tuple[int,int,int,int], str]]`

**Outputs:** `Dict[str, int]` — `{"person": 5, "car": 2}`

**Role in pipeline:** fourth stage. Aggregates object statistics by class.

---

## `src/renderer.py` — Renderer

**Responsibility:** visualization of all pipeline data over the frame.

**Main classes:**

### `FrameRenderer`

| Method | Description |
|--------|-------------|
| `__init__(font_size=0.7)` | Initializes font parameters, generates 256 colors |
| `render_detections(frame, detections, show_confidence)` | Draws detection bboxes |
| `render_tracks(frame, tracked_objects, show_ids)` | Draws track bboxes with IDs |
| `render_counts(frame, counts, total_count)` | Semi-transparent counter panel |
| `render_fps(frame, fps)` | FPS in the top-right corner |
| `_draw_label(frame, text, x, y, bg_color)` | Label with background |
| `_generate_colors(n=256)` | HSV → BGR, 256 colors |

**Inputs:** `np.ndarray` + data from previous modules

**Outputs:** `np.ndarray` with annotations (same buffer, modified in-place)

**Role in pipeline:** final stage before display. Does not affect processing logic.

**Color scheme:**
- White text on colored background (track bboxes)
- Green text for FPS metric
- Semi-transparent black background for the counter panel

---

## `src/metrics.py` — Performance Metrics

**Responsibility:** measurement and aggregation of pipeline timing characteristics.

**Main classes:**

### `PerformanceMetrics`

| Method | Description |
|--------|-------------|
| `__init__(window_size=30)` | Creates deque buffers for sliding window |
| `record_frame_time(elapsed)` | Records frame processing time (sec) |
| `record_detection_time(elapsed)` | Records inference time (sec) |
| `record_tracking_time(elapsed)` | Records tracking time (sec) |
| `get_fps()` | `1.0 / avg_frame_time` over the last 30 frames |
| `get_average_detection_time()` | Average detection time (ms) |
| `get_average_tracking_time()` | Average tracking time (ms) |
| `get_summary()` | Summary dictionary of all metrics |
| `reset()` | Clear buffers |

**Inputs:** `float` (time in seconds)

**Outputs:** `Dict[str, float]` — fps, avg_detection_time_ms, avg_tracking_time_ms, total_frames, elapsed_seconds

**Role in pipeline:** cross-cutting monitoring, not embedded in the data processing chain.

---

## `src/config.py` — Configuration

**Responsibility:** loading and accessing parameters from a YAML file.

### `Config`

| Method | Description |
|--------|-------------|
| `__init__(config_path)` | Path to YAML |
| `load()` | Reads file via `yaml.safe_load` |
| `get(key, default=None)` | Supports dot-notation: `"detector.confidence_threshold"` |

**Inputs:** path to a `.yaml` file

**Outputs:** config values of any type (`str`, `int`, `float`, `list`, `None`)

---

## `src/utils.py` — Utilities

**Responsibility:** geometric computations used by the tracker and counter.

| Function | Description |
|----------|-------------|
| `get_centroid(bbox)` | `(x1,y1,x2,y2)` → `(cx, cy)` |
| `distance(p1, p2)` | Euclidean distance between two points |
| `is_point_in_polygon(point, polygon)` | Ray casting, `True` if point is inside polygon |
| `calculate_iou(box1, box2)` | IoU of two bboxes (unused in current pipeline) |

**Note:** `calculate_iou` is implemented but not called by any module — potential for a future IoU-based tracker.
