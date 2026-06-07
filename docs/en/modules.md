# Module Descriptions

---

## `src/main.py` — Orchestrator

**Responsibility:** application entry point, coordination of all pipeline components, the main frame-processing loop.

**Main classes:**

### `ObjectCounterApp`

| Method | Description |
|-------|----------|
| `__init__(config_path, source_override)` | Loads the config, creates all components |
| `_initialize_components()` | Instantiates VideoSource, Detector, Tracker, Counter, Renderer, Metrics |
| `run()` | Main loop: read a frame → process → display |
| `_render_frame(frame, detections, tracked_objects, counts, fps)` | Conditional rendering based on config flags |
| `cleanup()` | Releases resources, prints the final metrics |

**Input:** `config_path: str`, `source: int | str` (from CLI)

**Output:** display in an OpenCV window, printing of final metrics to stdout

**Role in the pipeline:** the only module that knows about all the others. It passes data between components sequentially on each loop iteration.

---

## `src/video_source.py` — Video Source

**Responsibility:** capturing frames from a webcam or video file, resolution normalization, optional asynchronous (threaded) decoding.

**Main classes:**

### `VideoSource`

| Method | Description |
|-------|----------|
| `__init__(source, width, height, threaded, drop_policy)` | Opens cv2.VideoCapture, sets the resolution; with `threaded=True` starts a background producer thread |
| `read()` | Returns `(bool, frame)`, resizes to the target resolution; in threaded mode pulls a frame from the queue and re-raises decode errors |
| `release()` | Stops the producer thread (if any) and closes the capture |
| `get_fps()` | Source FPS (from video metadata or the camera) |
| `get_frame_count()` | Current frame number |
| `drops_frames` | Property: whether the drop-oldest policy is used |

**Asynchronous mode (`threaded=True`).** A background thread decodes and resizes frames into a `queue.Queue` while the main loop is busy with inference (decoding leaves the critical path). The `drop_policy`: `block` (backpressure, no losses — files), `drop` (only the freshest frame — live stream), `auto` (block for files, drop for live sources). See `docs/async_pipeline.md` for details.

**Input:**
- `source`: `int` (camera index) or `str` (file path)
- `width`, `height`: target resolution

**Output:** `Tuple[bool, Optional[np.ndarray]]` — a success flag and a frame in BGR format

**Role in the pipeline:** the first link. It supplies raw frames to the detector.

---

## `src/detector.py` — Object Detector

**Responsibility:** YOLOv8 inference on a frame, conversion of the raw model output into a structured format.

**Main classes:**

### `ObjectDetector`

| Method | Description |
|-------|----------|
| `__init__(model_name, confidence_threshold, device)` | Loads YOLO weights, configures the device |
| `detect(frame)` | Runs inference, returns a list of detections |
| `get_class_names()` | Dictionary `{id: class_name}` from the model |

**Input:** `frame: np.ndarray` (BGR, HxWx3)

**Output:**
```python
List[Tuple[Tuple[int, int, int, int], str, float]]
# [((x1, y1, x2, y2), class_name, confidence), ...]
```

**Role in the pipeline:** the second link. It converts a raw frame into a list of detected objects with coordinates and class.

**Note:** it uses `box.xyxy[0].int().tolist()` — absolute pixel coordinates in `(left, top, right, bottom)` format.

---

## `src/tracker.py` — Object Tracker

**Responsibility:** matching detections between frames, assigning and maintaining unique object IDs.

**Main classes:**

### `CentroidTracker`

| Method | Description |
|-------|----------|
| `__init__(max_disappeared, max_distance)` | Initializes the tracker state |
| `update(detections)` | Matches new detections with existing tracks |
| `get_tracks()` | Current track positions `{id: (cx, cy)}` |
| `reset()` | Resets all state |

**`update` algorithm:**
1. Compute the centroid of each detection
2. For each existing track, find the nearest detection (Euclidean distance)
3. If the distance < `max_distance` — update the track
4. Unmatched detections → new tracks with an incremental ID
5. Tracks without matches → increment `disappeared`; if `disappeared > max_disappeared` — remove it

**Input:** `List[((x1,y1,x2,y2), class_name, confidence)]`

**Output:**
```python
Dict[int, Tuple[Tuple[int, int, int, int], str]]
# {track_id: ((x1, y1, x2, y2), class_name)}
```

**Role in the pipeline:** the third link. It turns independent detections into continuous object tracks.

**Architectural limitation:** no occlusion handling — objects overlapping each other may lose their ID.

---

## `src/counter.py` — Object Counter

**Responsibility:** counting of unique objects, optional filtering by counting zone.

**Main classes:**

### `ObjectCounter`

| Method | Description |
|-------|----------|
| `__init__(count_zone=None)` | Initializes the counters; `count_zone` — polygon `List[Tuple[int, int]]` |
| `update(tracked_objects)` | Updates the counters based on the current tracks |
| `get_counts()` | `{class_name: count}` |
| `get_total_count()` | Total amount |
| `get_class_count(class_name)` | Counter for a specific class |
| `set_count_zone(zone)` | Dynamically set the counting zone |
| `reset()` | Reset all counters |

**Counting logic:** each `track_id` enters `counted_ids` exactly once. A repeated appearance in the frame is not counted. If `count_zone` is set, an object is counted only when its centroid is inside the polygon.

**Input:** `Dict[int, Tuple[Tuple[int,int,int,int], str]]`

**Output:** `Dict[str, int]` — `{"person": 5, "car": 2}`

**Role in the pipeline:** the fourth link. It aggregates object statistics by class.

---

## `src/renderer.py` — Renderer

**Responsibility:** visualization of all pipeline data over the frame.

**Main classes:**

### `FrameRenderer`

| Method | Description |
|-------|----------|
| `__init__(font_size=0.7)` | Initializes font parameters, generates 256 colors |
| `render_detections(frame, detections, show_confidence)` | Draws detection bboxes |
| `render_tracks(frame, tracked_objects, show_ids)` | Draws track bboxes with IDs |
| `render_counts(frame, counts, total_count)` | Semi-transparent panel with counters |
| `render_fps(frame, fps)` | FPS in the top-right corner |
| `_draw_label(frame, text, x, y, bg_color)` | Label with a background |
| `_generate_colors(n=256)` | HSV → BGR, 256 colors |

**Input:** `np.ndarray` + data from previous modules

**Output:** `np.ndarray` with annotations (the same buffer, modified in-place)

**Role in the pipeline:** the final link before display. It does not affect processing logic.

**Color scheme:**
- White text on a colored background (track bboxes)
- Green text for FPS metrics
- Semi-transparent black background for the counters panel

---

## `src/metrics.py` — Performance Metrics

**Responsibility:** measurement and aggregation of the pipeline's timing characteristics.

**Main classes:**

### `PerformanceMetrics`

| Method | Description |
|-------|----------|
| `__init__(window_size=30)` | Creates deque buffers for the sliding window |
| `record_frame_time(elapsed)` | Records frame processing time (sec) |
| `record_detection_time(elapsed)` | Records inference time (sec) |
| `record_tracking_time(elapsed)` | Records tracking time (sec) |
| `record_io_wait(elapsed)` | Records the main loop's frame-wait time (sec) |
| `get_fps()` | Throughput: `1.0 / avg_frame_time` over the last 30 frames |
| `get_inference_fps()` | Actual model rate (inferences/sec) by detection time |
| `get_average_detection_time()` | Average detection time (ms) |
| `get_average_tracking_time()` | Average tracking time (ms) |
| `get_average_io_wait()` | Average frame-wait time (ms) — the async-gain indicator |
| `get_summary()` | Summary dictionary of all metrics |
| `reset()` | Clears the buffers |

**Input:** `float` (time in seconds)

**Output:** `Dict[str, float]` — fps, inference_fps, avg_detection_time_ms, avg_tracking_time_ms, avg_io_wait_ms, total_frames, elapsed_seconds

**Role in the pipeline:** cross-cutting monitoring, not embedded in the data-processing chain.

---

## `src/config.py` — Configuration

**Responsibility:** loading and accessing parameters from a YAML file.

### `Config`

| Method | Description |
|-------|----------|
| `__init__(config_path)` | Path to the YAML |
| `load()` | Reads the file via `yaml.safe_load` |
| `get(key, default=None)` | Supports dot-notation: `"detector.confidence_threshold"` |

**Input:** path to a `.yaml` file

**Output:** config values of any type (`str`, `int`, `float`, `list`, `None`)

---

## `src/utils.py` — Utilities

**Responsibility:** geometric computations used by the tracker and counter.

| Function | Description |
|---------|----------|
| `get_centroid(bbox)` | `(x1,y1,x2,y2)` → `(cx, cy)` |
| `distance(p1, p2)` | Euclidean distance between two points |
| `is_point_in_polygon(point, polygon)` | Ray casting, `True` if the point is inside the polygon |
| `calculate_iou(box1, box2)` | IoU of two bboxes (not used in the current pipeline) |

**Note:** `calculate_iou` is implemented but not called from any module — potential for a future IoU-based tracker.
