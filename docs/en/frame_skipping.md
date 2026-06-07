# Frame Skipping

Reducing the inference load by running the heavy detect+track pass not on every
frame, but once every N frames. On intermediate frames the track boxes are
extrapolated, so the output stays smooth and the counting stays correct.

Controlled by a single parameter — `detector.detect_interval` (default `1` =
off, full backward compatibility).

---

## Why It's Needed

Inference (`model.track`) is the dominant cost per frame. If you run it once
every N frames, the GPU/CPU load drops roughly by a factor of N:

| `detect_interval` | Inferences per 100 frames | Inference load |
|-------------------|--------------------------|----------------------|
| `1` (off)         | 100                      | baseline             |
| `2`               | 50                       | ~−50%                |
| `3`               | 34                       | ~−66%                |
| `4`               | 25                       | ~−75%                |

This is the main speedup lever where the model is the bottleneck: a weak GPU, the
CPU path (`.onnx`), large models (`yolov8m`/`l`/`x`). It complements other
optimizations — FP16 (`half`) and export to ONNX/TensorRT (see
`docs/export_optimization.md`).

> This is **not** a real-time mechanism for dropping stale frames from a live
> camera (frame dropping for latency control) — that's a separate task at the
> level of the asynchronous pipeline. Here we mean deterministic skipping: every
> frame is read and displayed, but only every Nth is detected.

---

## How It Works

### Per-frame logic (`src/main.py` → `ObjectCounterApp.run`)

The frame is read **every** time; then it branches on `frame_idx % detect_interval`:

**Inference frame** (`_process_inference_frame`, every Nth):
1. `tracker.update(frame)` — the full detect+track pass.
2. For each track, the centroid velocity is estimated (displacement since the
   previous inference frame divided by `detect_interval` → pixels/frame).
3. ReID (if enabled) — by the same `reid.update_interval` rules.
4. `counter.update(tracked_objects)` — counting on the real boxes.
5. Results (detections, tracks, counts, velocities, object_id) are cached.

**Skipped frame** (`_process_skipped_frame`, the rest):
1. Cached track boxes are shifted by `velocity × k`, where `k = frame_idx % N`
   (linear forward extrapolation).
2. Counting and ReID are **not** recomputed — the cached values are reused.
3. The raw-detection layer is taken from the cache as-is (frozen).

Render and `cv2.imshow` run on **every** frame — the picture stays smooth in
terms of output rate.

---

## Decisions and Their Rationale

### 1. Skipping the input stream, not a separate lightweight tracker

The classic trick "heavy detector once every N frames + a light tracker in
between" is not directly applicable here: in this project **detection and
tracking are fused into a single call** `model.track(persist=True)`
(`tracker.py`). Ultralytics provides no "track without detection" mode. Setting
up a separate Kalman/optical-flow tracker for intermediate frames would be a
large separate rework. So we implemented skipping of the whole pass with box
extrapolation on the application side.

### 2. Linear extrapolation instead of freezing the boxes

On skipped frames the track boxes move by the estimated velocity rather than
standing still. This:
- gives a smooth picture (no "jerk" once every N frames);
- is more geometrically accurate for line/zone crossing during visual inspection.

The velocity is estimated from the last two inference frames and normalized by
`detect_interval` (displacement over N real frames → pixels per frame).
Trade-off: on a sharp direction change the extrapolated box may briefly "drift
away" from the object — this is corrected on the next inference frame.

### 3. Counting and ReID — only on inference frames

`counter.update()` is called only when there are **observed** (not predicted)
boxes. Reasons:
- predicted centroids must not produce phantom line crossings;
- double counting is excluded (on frozen/extrapolated boxes the object does not
  "re-enter" the zone).

This is safe for crossing counting: the counter records a line-side change
between samples (`counter.py`), and as long as the object is sampled on both
sides of the line, the crossing is counted once. The only risk is missing an
event if the object manages to fully cross the line *between* two inference
frames (grows with N and object speed). ReID similarly runs embeddings only on
fresh crops; between frames the last `object_id`s are reused.

### 4. The raw-detection layer is frozen

Raw detections (`show_detections`) have no `track_id`, so a velocity cannot be
attached to them — they are reused from the cache without a shift. The track
boxes (with IDs, used for counting) do move. At large N the frozen detection
layer may visually diverge from the tracks; it's a debugging layer, and for a
clean picture it can be disabled (`display.show_detections: false`).

### 5. `lost_track_buffer` is divided by N

`lost_track_buffer` maps to Ultralytics' `track_buffer` and is measured in
**tracker ticks**, and the tracker ticks only on inference frames. One tick now =
N real frames. If the value isn't scaled, the buffer's actual duration grows by N
(a set `auto_lost_track_buffer_seconds: 3.0` at N=3 would become ~9 s of real
time → dead tracks linger too long, higher risk of ID hijacking).

Solution: treat the config value as "frames of the source video" and divide by N
— `max(1, round(buffer / detect_interval))` — for **both auto- and explicitly
set** values (for consistency: the same buffer in seconds must not behave
differently depending on how it was specified). The buffer duration in seconds
stays constant for any N. Implemented in one place in `_initialize_components`
after `_resolve_lost_track_buffer`.

> Important: the association *quality* itself still degrades as N grows (Kalman
> predicts over a larger step dt, less overlap of neighboring observations →
> higher ID-switch risk). This is not cured by the buffer — it's the price of
> skipping.

### 6. Two FPS metrics

When `detect_interval > 1`, showing a single FPS is misleading. The overlay
shows:
- **`FPS`** — throughput, displayed frames per second (including the cheap
  skipped frames);
- **`infer`** — the actual model rate, inferences per second (by the time spent
  inside `model.track`), reflecting the true inference cost regardless of
  skipping.

The second line appears only when `detect_interval > 1`. In the final summary
(`metrics.get_summary`) an `inference_fps` key is added.

---

## Configuration and Running

```yaml
detector:
  detect_interval: 3   # detection every 3rd frame; 1 = off
```

```bash
# via config
python -m src.main --config configs/experiments/01_bytetrack_fast.yaml --source data/input/video.mp4
```

At startup the actual value is visible in the tracker log:
`Tracker: bytetrack (Ultralytics) | fps=30, ..., detect_interval=3`.

---

## Effect on the Result

| Aspect | Effect |
|--------|--------|
| Inference load | drops ~N× — the main gain |
| Throughput (output FPS) | grows if inference is the bottleneck; limited by IO/decode/render |
| Tracking | higher ID-switch risk on fast motion/occlusions at large N |
| Line-crossing counting | correct as long as the object is sampled on both sides of the line; risk of missing fast crossings between inference frames |
| Zone counting | robust (recorded once on entry) |
| Visually | track boxes are smooth (extrapolation); the raw-detection layer may lag |

**Recommendations for choosing N:**
- `2–3` — a safe range for most scenes, a noticeable gain with little quality
  loss.
- `4+` — only for slow scenes / when inference is critically expensive; verify
  counting and ID stability on your own video.
- If inference is not the bottleneck (ReID/render/IO take a significant share of
  the frame) — the gain in final FPS will be smaller; consider FP16/export first.

---

## Where in the Code

| What | File |
|-----|------|
| Reading `detect_interval`, dividing the buffer, the loop branch, extrapolation | `src/main.py` (`_initialize_components`, `run`, `_process_inference_frame`, `_process_skipped_frame`) |
| `inference_fps` (raw model rate) | `src/metrics.py` (`get_inference_fps`, `get_summary`) |
| Two-line FPS overlay | `src/renderer.py` (`render_fps`) |
| Parameter and comments | `configs/default.yaml`, `docs/config.md` |

---

## Related Optimizations and Future Tasks

- **FP16 / export to ONNX/TensorRT** — `docs/export_optimization.md`. Combine
  with skipping (they speed up inference itself, not its frequency).
- **Asynchronous video pipeline** — a separate task: move frame
  reading/decoding into a producer thread and add real-time dropping of stale
  frames for a live camera. Skipping reduces the *compute* load; the async
  pipeline removes the *IO-bound* limitation and controls latency.
