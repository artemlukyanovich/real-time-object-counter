# Asynchronous Video Pipeline (Async pipeline)

Moving frame decoding into a background thread so that decoding runs **in
parallel** with inference instead of sitting on the critical path of the main
loop. It removes the IO-bound limitation and, for a live source, controls latency
by dropping stale frames.

Controlled by two parameters: `video.async_pipeline` (on/off, default `true`) and
`video.drop_policy` (the frame-drop policy).

---

## Why It's Needed

In a synchronous loop, one frame costs `decode + inference` — while the next
frame is decoded, the GPU sits idle, and vice versa. A background producer thread
decodes frames in advance, so the frame time drops toward `max(decode,
inference)`:

```
sync:   [decode][   inference   ][decode][   inference   ] ...
async:  [decode][decode][decode] ...   (in background)
                 [   inference   ][   inference   ] ...
```

Especially useful together with frame skipping (`detect_interval`, see
`docs/frame_skipping.md`): on skipped frames the loop is cheap
(extrapolation+render), and decoding easily becomes the bottleneck — async
removes it.

> **Why a thread, not a process (GIL).** `cap.read()`, `cv2.resize`, and
> torch/CUDA inference release the GIL during C/CUDA operations, so an ordinary
> `threading.Thread` producer gives a real overlap of decoding with inference.
> `multiprocessing` is not needed and would be more expensive due to frame
> serialization.

---

## How It Works

```
[producer thread]  cap.read() → resize → queue.put()
        │  (sentinel at end of stream / on error)
        ▼
[main loop]  queue.get() → tracker → counter → render → imshow
```

Implemented in `src/video_source.py`. The signature `read() → (ret, frame)` has
**not changed**, so the main loop (`ObjectCounterApp.run`) is barely affected — it
simply reads frames as before, unaware whether the mode is synchronous or not.

- **Producer** (`_reader_loop`): in a loop decodes and resizes frames, puts them
  into a `queue.Queue`. At end of stream (or on a decode exception) it puts a
  sentinel.
- **Consumer** (`read`): pulls a frame from the queue; on a sentinel returns
  `(False, None)`, and if there was an exception in the thread — re-raises it in
  the main loop.
- **Shutdown** (`release`): sets a stop event, unblocks a possibly blocked `put`,
  `join`s the thread, and releases the capture.

---

## Frame-Drop Policy (`drop_policy`)

The main question of async mode is **what to do when inference can't keep up with
the frame stream**:

| Policy | Behavior | When |
|----------|-----------|-------|
| `block` | the producer waits until space frees up in the queue (backpressure). No frame is lost, the result is deterministic, but latency grows | **video file** |
| `drop` | queue size 1; when full the old frame is discarded, the freshest is processed. Low latency, the load sheds itself, but some frames are skipped | **live stream** (camera/drone) |
| `auto` | `block` for video files, `drop` for live sources | default |

### Why the policy differs for a file and a stream

- **A video file** is essentially an offline task: *every* frame needs to be
  processed. Dropping would change the count and make the result non-deterministic
  (dependent on CPU load/timings). So a file → `block` (lossless,
  reproducible). The gain here is the overlap of decoding with inference, without
  frame loss.
- **A live stream (camera/drone)** is a real-time task: if we can't keep up, we
  must discard stale frames and keep the fresh one. For a drone this is critical —
  an old frame means the object has already moved. On weak hardware `drop` is the
  very mechanism for "not choking": the system automatically thins the stream
  exactly as much as it can't handle.

### Override

`drop_policy` can be set explicitly (`block` / `drop`), overriding the auto
choice. The main scenario is **load-testing edge cases**: run a recorded file in
`drop` mode, simulating a live stream from a drone, right on the target weak
device — without a real drone. This shows how many frames the system actually
manages to process and how it keeps freshness under load.

---

## Decisions

1. **A toggle instead of "always on".** Asynchronous reading is almost always
   beneficial, but the `async_pipeline` flag is kept intentionally: (a) to
   **A/B-measure** the gain on the same video (sync vs async), (b) as a fallback
   when debugging thread races. The default is `true`.
2. **Drop policy by source type, not "one drop for all".** A single drop would
   silently lose frames on a file → a counting regression and loss of
   reproducibility. So `auto` differentiates the behavior, and the override is
   kept for load tests.
3. **A thread, not a process** — see the GIL note above.
4. **The `read()` signature is preserved** — the main loop doesn't know about the
   mode; async is enabled with a single flag without rewriting the pipeline.
5. **Correct shutdown and error propagation** — a sentinel at end of stream,
   propagation of decode exceptions into the main loop, stopping with a `join` in
   `release()`.

---

## Configuration and Running

```yaml
video:
  async_pipeline: true   # false = synchronous single-threaded mode
  drop_policy: "auto"    # auto | block | drop
```

```bash
python -m src.main --source data/input/video.mp4
```

At startup the actual mode is visible in the log:
`Video: async pipeline ON | drop_policy=auto (block/no-drop)`.

---

## How to Assess the Result

Unlike ONNX/FP16/skipping, where the gain is visible from a backend change, here
the comparison is sync vs async on the **same** video. The main indicator is
**I/O-wait**: the time the main loop sits idle waiting for a frame. It's printed
in the final summary as `avg_io_wait_ms` (the `record_io_wait` metric in
`PerformanceMetrics`).

| Metric | sync | async (expected) |
|---------|------|------------------|
| `avg_io_wait_ms` | ≈ decode time (on the critical path) | → ~0 if decoding is faster than inference |
| `fps` (throughput) | baseline | higher; the ceiling is the frame time `max(decode, inference)` |

### A/B methodology (on a file — `block`, determinism)

```bash
# async ON (default)
python -m src.main --source data/input/video.mp4

# async OFF: video.async_pipeline: false, then the same run
python -m src.main --source data/input/video.mp4
```

Compare `avg_io_wait_ms` and `fps` in the final summary. A drop of
`avg_io_wait_ms` toward ~0 is direct proof that decoding left the critical path.

### Live-stream load test

```yaml
video:
  drop_policy: "drop"   # a file as a "live stream"
```

Shows how many frames the system actually manages to process on the given
hardware and how fresh the frame stays (relevant for a drone on a weak edge
device).

---

## "Before / After" Measurement Examples

Config `custom_1.yaml` (yolov8m, CUDA, `half: true`, `detect_interval: 2`), an
RTX-class GPU. This is an illustration of **when async helps and when it
bottlenecks on the source** — the absolute numbers depend on the hardware.

### File (`drop_policy: auto` → `block`)

Source: `street_camera_2x_speed.mp4`, ~1473 frames.

| Metric | sync | async |
|---------|------|-------|
| `fps` (throughput) | 85.3 | **109.1** (+28%) |
| `inference_fps` | 76.5 | 72.6 (≈ unchanged) |
| `avg_io_wait_ms` | 3.11 | **0.03** |
| `elapsed_seconds` | 23.6 | **19.8** |
| `total_frames` / Total counts | 1473 / 27 | 1472 / 27 (identical) |

**Conclusion:** a clean win. Decoding left the critical path — `io_wait` dropped
almost to zero, throughput grew by ~28%. The gain ≈ exactly the decode that sat
on the critical path: the frame time 11.7 → 9.2 ms, a difference of ~2.5 ms ≈ the
removed `io_wait`. `inference_fps` didn't change (async doesn't speed up the
model, it overlaps decoding). The counts matched → `block` loses no frames,
determinism is preserved.

### Webcam (`drop_policy: auto` → `drop`)

| Metric | sync | async |
|---------|------|-------|
| `fps` (throughput) | 10.46 | 10.49 |
| `inference_fps` | 48.3 | 45.6 |
| `avg_io_wait_ms` | 81.8 | 80.7 |

**Conclusion:** async gave nothing — and that's correct. The bottleneck here is
not compute but the camera itself: `inference_fps` ≈ 48 (the model could do 48
frames/s), but throughput is only ~10.5. The large `io_wait` of ~81 ms is the
wait until the camera delivers the next frame (typical for auto-exposure in low
light). Async doesn't create frames the source doesn't deliver; the wait merely
moves to the producer thread. Inference (~20 ms) is much faster than the camera
interval (~80 ms), so the queue is almost always empty and **`drop` never
triggers** — this test doesn't stress the drop.

### What follows from this

- **Async speeds things up when the bottleneck is decode/IO and frames are
  abundant** (files, high-rate streams).
- **Async is neutral when the bottleneck is the source itself** (the camera
  delivers ~10 fps). Not a bug — there's nothing to optimize.
- **The benefit of `drop`** (latency control + load shedding) shows up only when
  compute is *slower* than the frame inflow. To reproduce a drone/edge scenario:
  run on the target weak device, or simulate it — a high-fps file +
  `drop_policy: "drop"` + a heavy model (`yolov8l/x`, `detect_interval: 1`,
  `half: false`); then inference starts to lag, the queue overflows, and `drop`
  will cut `total_frames` but keep `io_wait`/latency low (unlike `block`, where
  latency would grow).

---

## Effect on the Result

| Aspect | Effect |
|--------|--------|
| Throughput (FPS) | grows if decoding was on the critical path; the ceiling is `max(decode, inference)` |
| I/O-wait | drops toward ~0 with `block` when decoding is faster than inference |
| Latency (camera, `drop`) | low and stable — the freshest frame is always processed |
| Counting on a file (`block`/`auto`) | unchanged — all frames are processed, deterministically |
| Counting with `drop` | some frames are skipped → events may be missed; this mode is for live/load tests only |

> **Interaction with `detect_interval`.** With `drop`, the real interval between
> processed frames becomes irregular (depends on how many frames were dropped),
> which means the velocity estimate for box extrapolation may be slightly
> distorted. With `block`/`auto` on a file there's none of this — frames come in
> sequence.

---

## Where in the Code

| What | File |
|-----|------|
| Producer thread, queue, drop policy, shutdown | `src/video_source.py` (`_reader_loop`, `_resolve_drop`, `read`, `release`, `drops_frames`) |
| I/O-wait metric | `src/metrics.py` (`record_io_wait`, `get_average_io_wait`, `get_summary`) |
| Config reading, I/O-wait measurement, mode log | `src/main.py` (`_initialize_components`, `run`) |
| Parameters and comments | `configs/default.yaml`, `docs/config.md` |

---

## Related Optimizations

- **Frame skipping** (`detect_interval`) — `docs/frame_skipping.md`. Reduces the
  *compute* load (inference frequency); async removes the *IO-bound* limitation.
  They complement each other.
- **FP16 / ONNX/TensorRT export** — `docs/export_optimization.md`. They speed up
  inference itself.
