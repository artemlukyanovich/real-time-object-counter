# Embeddings and Re-Identification (Stages 4–5)

---

## Overview

A new pipeline on top of the existing one:

```
YOLO → Tracker → tracked_objects
                      │
                  ObjectCropper   ← crop from bbox
                      │
                  ObjectEmbedder  ← feature vector (OpenCLIP)
                      │
                  ObjectMemory    ← similarity search
                      │
                  ReIDManager     → Dict[track_id, object_id]
```

**Key difference from ordinary tracking:**
- `track_id` — a temporary ID issued by ByteTrack/BoT-SORT. Reset when the object is lost.
- `object_id` — a persistent ID that survives leaving the frame, occlusion, and a `track_id` change.

---

## Installing Dependencies

`open-clip-torch` is already included in `requirements.txt`. On the first access to
`ObjectEmbedder`, the model weights (`ViT-B-32`, ~350 MB) are downloaded
automatically into the Torch Hub cache.

```bash
pip install -r requirements.txt
```

For GPU acceleration, make sure the CUDA version of torch is installed and set
`device: "cuda"` in `configs/embeddings/default.yaml`.

---

## Quick Test: crop → embedding → similarity

The following script verifies that all the new modules work without a video source.

```python
import cv2
import numpy as np
from src.cropper import ObjectCropper
from src.embedder import ObjectEmbedder
from src.similarity import cosine_similarity, find_best_match

# A synthetic frame and two bboxes
frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
bbox_a = (100, 100, 300, 400)
bbox_b = (600, 200, 900, 500)

cropper = ObjectCropper(padding=8)
embedder = ObjectEmbedder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")

crop_a = cropper.crop(frame, bbox_a)
crop_b = cropper.crop(frame, bbox_b)

emb_a = embedder.embed(crop_a)
emb_b = embedder.embed(crop_b)

score = cosine_similarity(emb_a, emb_b)
print(f"Similarity between two random crops: {score:.4f}")  # ~0.8–0.95 for random patches

# find_best_match test
gallery = np.stack([emb_a, emb_b])
matched_id, best_score = find_best_match(emb_a, gallery, object_ids=[1, 2], threshold=0.8)
print(f"Best match: object_id={matched_id}, score={best_score:.4f}")  # expected id=1, score≈1.0
```

---

## Quick Test: ReIDManager on a Video File

```python
import cv2
from src.cropper import ObjectCropper
from src.embedder import ObjectEmbedder
from src.object_memory import ObjectMemory
from src.reid import ReIDManager

cropper = ObjectCropper(padding=8, save_crops=True)          # saves crops to outputs/crops/
embedder = ObjectEmbedder(device="cpu")
memory = ObjectMemory(similarity_threshold=0.75, max_missing_frames=90)
reid = ReIDManager(cropper, embedder, memory)

cap = cv2.VideoCapture("data/input/your_video.mp4")
frame_idx = 0

# Assume tracked_objects is already obtained from UltralyticsTracker
# Here we simulate with a single object
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # In real usage: detections, tracked_objects = tracker.update(frame)
    fake_tracked = {42: ((200, 100, 400, 350), "person")}

    track_to_obj = reid.update(frame, fake_tracked, frame_idx)
    print(f"Frame {frame_idx}: track 42 → object_id {track_to_obj.get(42)}")

    frame_idx += 1

cap.release()
print(f"Total unique objects seen: {reid.total_object_count()}")
```

---

## Integration Test on Real Video

A smoke test with synthetic data verifies that the code runs without errors, but
says nothing about ReID quality on real objects. The next step is to run the full
pipeline `YOLO → Tracker → Crop → Embedding → ReID` on a real video and analyze
the distribution of similarity scores.

### Running

```bash
python -m scripts.test_reid_integration \
    --source data/input/your_video.mp4 \
    --config configs/default.yaml \
    --max-frames 300 \
    --threshold 0.75
```

Arguments:

| Argument | Default | Description |
|---|---|---|
| `--source` | required | Path to a video or a webcam index (`0`) |
| `--config` | `configs/default.yaml` | Pipeline config (model, tracker, resolution) |
| `--max-frames` | `300` | Number of frames to process |
| `--threshold` | `0.75` | Cosine similarity threshold for matching |

### What the Script Does

1. Opens the video, runs the detector + tracker on every frame.
2. For each track it crops, computes the embedding, and compares it against memory.
3. Logs each event to the console:
   - **`continuous`** — the track is already bound to an object_id, we update it.
   - **`RE-ID ✓`** — a new track_id matched a known object (a real re-id).
   - **`new`** — no match above the threshold, a new object_id is created.
4. Saves all crops to `outputs/crops/` for visual inspection.
5. Prints a final table of the score distribution and a hint on choosing the threshold.

### Example Output

```
 frame  track_id  object_id  event         score
────────────────────────────────────────────────────────────────────────
     0         1          1  new           0.0000
     0         2          2  new           0.0000
     1         1          1  continuous    0.9821
     1         2          2  continuous    0.9743
    47         3          1  RE-ID ✓       0.8612   ← track lost, but the object is recognized
    47         4          3  new           0.4231
...

Similarity score distributions
────────────────────────────────────────────────────────────────────────
  same object (continuous)        n= 284  min=0.8901  mean=0.9612  max=0.9981
  re-id match (new track_id)      n=   3  min=0.8401  score=0.8612  max=0.8901
  no match (new object)           n=   4  min=0.3120  mean=0.4105  max=0.5230

Suggested threshold range: 0.76 – 0.86  (current: 0.75)
```

### What to Check in the Console

| What to look at | Good sign | Warning sign |
|---|---|---|
| `continuous` scores | `mean > 0.90` — the object is recognized stably | `mean < 0.80` — the model is unstable on this object type |
| Gap between `continuous` and `no match` mean | `> 0.20` — the threshold is easy to choose | `< 0.10` — object classes are poorly separated by CLIP |
| Number of `RE-ID ✓` events | Appear on track loss/change | `0` on a long video — objects aren't lost (OK) or the threshold is too high |
| Number of unique objects | Matches the real number of people/cars in the frame | Much higher — false negatives (threshold too high); much lower — false positives (threshold too low) |

### What to Check in `outputs/crops/`

Open the folder and make sure the crops:
- contain the whole object, not the background or the frame edge;
- are not empty or too small (< 20×20 px);
- show the same object across all frames for one `track_id`;
- crops with the same `object_id` (suffix `_objN`) really depict one physical object, even if the `track_id` differs — this confirms re-id correctness.

If the crops are bad — increase `padding` in `configs/embeddings/default.yaml` or lower `detector.confidence_threshold`.

### Tuning the Threshold

After the run, the script prints a hint:
```
Suggested threshold range: 0.76 – 0.86  (current: 0.75)
```

Formula: `mean(same_scores) - 0.4 * gap`, where `gap = mean(same) - mean(no_match)`.
This is a point roughly 40% of the way from "no match" to "match". Check the values at both ends of the range:

```bash
# Strict threshold — fewer false re-ids
python -m scripts.test_reid_integration --source ... --threshold 0.85

# Lenient threshold — more re-id events
python -m scripts.test_reid_integration --source ... --threshold 0.70
```

Enter the found threshold into `configs/embeddings/default.yaml`:
```yaml
memory:
  similarity_threshold: 0.82  # tuned experimentally
```

---

## Integration into the Main Pipeline (`src/main.py`)

The ReID pipeline is **fully integrated** into `ObjectCounterApp` and is enabled
with a single line in the config.

### Enabling

In `configs/default.yaml` set:

```yaml
reid:
  enabled: true
  embeddings_config: "configs/embeddings/default.yaml"  # model and memory parameters
  update_interval: 3   # run the pipeline every N frames
  min_track_age: 1     # frames before registering a new object (1 = instant)
```

The model, cropper, and memory parameters are taken from
`configs/embeddings/default.yaml` — threshold and device tuning is done there.

### What Happens at Startup

With `reid.enabled: true` the application:

1. Loads `configs/embeddings/default.yaml` and initializes `ObjectCropper`, `ObjectEmbedder`, `ObjectMemory`, and `ReIDManager` (`_initialize_reid()`).
2. On every frame, after `tracker.update()`, it calls `reid_manager.update(frame, tracked_objects, frame_idx)`, obtaining `Dict[track_id, object_id]`.
3. Passes the mapping to `FrameRenderer` for display.

### Visual Assessment

Enable the corresponding options in `configs/default.yaml`:

```yaml
display:
  show_object_ids: true   # show the OBJ ID instead of track_id on the bounding box
  show_reid_stats: true   # "ReID unique / active" panel in the top-right corner
```

**Bounding box label format:**

```
#N class [tM]
```

- `#N` — the persistent `object_id` (doesn't change on object loss and reappearance)
- `class` — the object's class
- `[tM]` — the tracker's current temporary `track_id`

**The bounding box color** is determined by `object_id`, not `track_id`: the same
physical object is always highlighted in one color, even if the tracker changed
its `track_id`.

**The ReID stats panel** (top-right corner, under FPS):

```
ReID unique: 5
ReID active: 3
```

- `unique` — the total number of unique objects over the session
- `active` — objects currently active (not expired by `max_missing_frames`)

### How to Read the Visualization

| What you see on screen | Interpretation |
|---|---|
| The bounding box color doesn't change on a brief disappearance | ReID correctly re-identified the object |
| `#N` stays the same when `tM` changes | Successful re-id: the new track_id was matched to the old object_id |
| `unique` grows faster than the real number of objects | The threshold is too high — objects aren't recognized |
| `unique` is lower than the real number of objects | The threshold is too low — different objects merge into one |

### Example Output When Running with ReID

```
Tracker: bytetrack (Ultralytics) | fps=30, activation_threshold=0.5, ...
ReID: enabled | model=ViT-B-32 device=cpu threshold=0.75 update_interval=3 min_track_age=1
Starting object counter. Press 'q' to exit.
```

---

## Configuration

The parameters are in `configs/embeddings/default.yaml`:

```yaml
embedder:
  model_name: "ViT-B-32"          # OpenCLIP model
  pretrained: "laion2b_s34b_b79k" # weights
  device: "cpu"                   # "cuda" for GPU
  normalize: true                 # L2 normalization (recommended)

cropper:
  padding: 8          # padding pixels around the bbox
  save_crops: false   # save crops to outputs/crops/

memory:
  similarity_threshold: 0.75   # threshold for a re-id match
  max_missing_frames: 90       # ~3 sec at 30 FPS
  max_embeddings_per_object: 5 # rolling buffer

output:
  save_embeddings: false  # save .npy to outputs/embeddings/
  save_reid_log: false    # save the event log to outputs/reid/
```

**Tuning `update_interval` and `min_track_age`** (in `configs/default.yaml`, the
`reid` section):

`update_interval` — how often to run the embedding pipeline. `min_track_age` —
filters out "flickering" objects: until a track "survives" the given number of
**video frames** (not pipeline calls), it doesn't enter `unique` / `active`. If a
track matches an already-known object by embedding — the threshold is not applied,
the object is matched instantly.

`min_track_age` is measured in real frames by `frame_idx`, independently of
`update_interval`. With `update_interval: 10` and `min_track_age: 8` an object is
confirmed after 8 frames (≈0.25 sec at 30 FPS), not after 80.

| `min_track_age` | Behavior |
|---|---|
| `1` | Disabled — every track becomes an object immediately (former behavior) |
| `8` | At 30 FPS ≈ 0.25 sec — cuts off instantaneous false detections |
| `30` | 1 second — only stable objects |

**Threshold tuning:**

| `similarity_threshold` | Behavior |
|---|---|
| `0.90+` | Strict — almost no false matches, but objects are more often counted as new |
| `0.75` | Balanced (recommended to start) |
| `0.60–` | Lenient — more re-id matches, higher risk of merging different objects |

---

## Embedding Aggregation Methods

By default, `ObjectMemory` represents each object as the **mean** of all
accumulated embeddings (`aggregation_method: "mean"`). This is a safe option for an
MVP, but it has a weak point: simple averaging **blurs the identity** if:

- the object changes appearance significantly between frames,
- "bad" crops got into the buffer (partial occlusion, glare, very small size),
- the embeddings of early frames turned out to be noisy.

For such cases, three alternative methods are implemented.

### Comparison Table

| Method | How it works | Pros | Cons | When to use |
|---|---|---|---|---|
| `mean` | Mean of all stored embeddings | Stability, robustness to noise | Adapts slowly, noisy embeddings weigh the same as fresh ones | The object doesn't change; short scenes; start |
| `ema` | Exponential moving average: a new embedding weighs `alpha`, the old EMA `1-alpha` | Adapts to appearance changes, doesn't sharply "forget" history | A bad crop quickly spoils the representation; needs calibration of `ema_alpha` | The object gradually changes appearance (different angle, lighting); long scenes |
| `weighted` | Weighted mean with exponential decay: old embeddings are multiplied by `weighted_decay` each step | Smooth adaptation; old data keeps influencing, but more weakly | Harder to tune `weighted_decay` for a specific scenario | An intermediate option between `mean` and `ema` |
| `recent` | Mean of only the last `recent_n` embeddings | Fast reaction to appearance change; ignores "stale" history | Vulnerable to bad crops in recent frames | The object changes a lot; long tracks with viewpoint change |

### Configuration

All parameters are in `configs/embeddings/default.yaml`, the `memory` section:

```yaml
memory:
  aggregation_method: "mean"   # mean | ema | weighted | recent
  ema_alpha: 0.3               # ema only:       0.2 (slow) – 0.5 (fast)
  recent_n: 3                  # recent only:    <= max_embeddings_per_object
  weighted_decay: 0.7          # weighted only:  0.5 (steep decay) – 0.95 (almost flat)
```

#### Example: EMA

```yaml
memory:
  aggregation_method: "ema"
  ema_alpha: 0.3
```

`ema_alpha = 0.3` means: a new embedding contributes 30% to the representation, the
accumulated history — 70%. If the object changes appearance quickly, increase to
0.4–0.5.

#### Example: recent

```yaml
memory:
  aggregation_method: "recent"
  recent_n: 3
```

Only the 3 most recent embeddings from the rolling buffer are used. If the buffer
is `max_embeddings_per_object: 5`, then the 2 oldest are ignored entirely.

#### Example: weighted

```yaml
memory:
  aggregation_method: "weighted"
  weighted_decay: 0.7
```

For a buffer of 5 embeddings `[e0, e1, e2, e3, e4]` (e0 — oldest, e4 — newest) the
weights will be proportional to `[0.7³, 0.7², 0.7¹, 0.7⁰ · 0.7, 1.0]`, normalized
to sum to 1.

### How to Choose a Method

1. **Start with `mean`** — it's the most predictable option for most scenes.
2. If in the `test_reid_integration` logs you see that `continuous` scores are unstable (large spread) or RE-ID stops working after a few minutes — try `ema` with `ema_alpha: 0.3`.
3. If the object frequently changes its angle (e.g., a person turns around) and the threshold has to be dropped below 0.70 — try `recent` with `recent_n: 3`.
4. With poor crop quality (small objects, frequent occlusions), `mean` or `weighted` is preferable to `recent` — the latter is insensitive to history and "breaks" more easily from noisy frames.

> After changing the aggregation method, **recalibrate `similarity_threshold`**
> with `test_reid_integration` — different methods give different ranges of
> similarity scores.

---

## Choosing the OpenCLIP Model

All models use the same `ObjectEmbedder` interface and are swapped via just two
config fields: `model_name` and `pretrained`.

### Comparison Table

| Model | Patch size | Embedding dimension | Weights size | Speed (CPU) | ReID accuracy |
|---|---|---|---|---|---|
| `ViT-B-32` | 32×32 | 512 | ~350 MB | fast | baseline |
| `ViT-B-16` | 16×16 | 512 | ~350 MB | medium | higher than B-32 |
| `ViT-L-14` | 14×14 | 768 | ~890 MB | slow | high |
| `ViT-H-14` | 14×14 | 1024 | ~2.5 GB | very slow | highest |

> Smaller patch size → more tokens per image → higher feature detail, but also
> higher compute load.

### ViT-B-32 (default)

The optimal choice for real time. The 32×32 patches give coarse spatial features,
sufficient for ReID of large objects (people, cars). Runs on CPU with acceptable
latency (~15–40 ms/frame depending on the number of objects).

```yaml
embedder:
  model_name: "ViT-B-32"
  pretrained: "laion2b_s34b_b79k"
  device: "cpu"
```

**When to use:** prototyping, no GPU, objects are large enough (> 80×80 px in the crop).

### ViT-B-16

The same Base architecture, but 16×16 patches — twice as many tokens. The
embedding dimension is the same (512), but the features capture finer texture
details. Recommended as the first upgrade from B-32 without a memory increase.

```yaml
embedder:
  model_name: "ViT-B-16"
  pretrained: "laion2b_s34b_b88k"
  device: "cuda"
```

**When to use:** a GPU is available, medium-sized objects, improved accuracy is
needed without an increased dimension.

### ViT-L-14

The Large model: 24 transformer layers vs 12 in Base. The embedding dimension is
768. Noticeably better at separating similar objects (e.g., people in identical
clothing). On CPU it gives 100–300 ms/frame latency, on a modern GPU — 5–15 ms.

```yaml
embedder:
  model_name: "ViT-L-14"
  pretrained: "laion2b_s32b_b82k"
  device: "cuda"
```

**When to use:** a GPU is mandatory, objects are hard to distinguish visually, high
ReID accuracy is required.

> Dimension 768 instead of 512: if you save embeddings to `.npy` or pass them to an
> external system, account for the format change.

### ViT-H-14

The Huge model: 32 transformer layers, embedding dimension 1024. The weights take
~2.5 GB of VRAM. Practically not used in real time — applied for offline analysis
or when accuracy matters more than speed.

```yaml
embedder:
  model_name: "ViT-H-14"
  pretrained: "laion2b_s32b_b79k"
  device: "cuda"
```

**When to use:** offline processing of recorded video, maximum accuracy, a GPU with
≥ 8 GB of VRAM.

### How to Choose `pretrained`

The weights tag indicates on which dataset and with what batch the model was
trained. **The tag is tied to a specific architecture** — you can't use a tag from
one model with another.

Recommended tags for ReID:

| Model | `pretrained` | Dataset |
|---|---|---|
| `ViT-B-32` | `laion2b_s34b_b79k` | LAION-2B |
| `ViT-B-16` | `laion2b_s34b_b88k` | LAION-2B |
| `ViT-L-14` | `laion2b_s32b_b82k` | LAION-2B |
| `ViT-H-14` | `laion2b_s32b_b79k` | LAION-2B |

To view all available `(model_name, pretrained)` pairs:

```python
import open_clip
open_clip.list_pretrained()
```

For ReID the `laion2b_*` weights are recommended — they are trained on 2 billion
images and give better visual features than `openai` (trained on only 400M
text–image pairs).

### Effect on `similarity_threshold`

Different models give different ranges of cosine similarity for the same object.
After changing the model, **be sure** to recalibrate the threshold with:

```bash
python -m scripts.test_reid_integration --source data/input/your_video.mp4 --threshold 0.75
```

Approximate starting values:

| Model | Recommended `similarity_threshold` |
|---|---|
| `ViT-B-32` | 0.75 |
| `ViT-B-16` | 0.78 |
| `ViT-L-14` | 0.80 |
| `ViT-H-14` | 0.82 |

---

## API Reference

### `ObjectCropper` (`src/cropper.py`)

| Method | Description |
|---|---|
| `__init__(padding, save_crops, output_dir)` | Creates a cropper; `padding` — padding in pixels |
| `crop(frame, bbox, track_id, frame_idx, object_id)` | Crops one crop; returns a `np.ndarray` BGR |
| `crop_all(frame, tracked_objects, frame_idx)` | Batch: `Dict[track_id, crop]` |

Returns an empty array `shape=(0,0,3)` if the bbox goes out of the frame bounds.

**Naming of saved files:**

| Passed parameters | File name |
|---|---|
| `frame_idx=31, track_id=2` | `frame000031_id2.jpg` |
| `frame_idx=31, track_id=2, object_id=1` | `frame000031_id2_obj1.jpg` |
| `track_id=2` only | `id2.jpg` |

The `_obj{N}` suffix is added only when `object_id` is explicitly passed. This lets
you match crops with different (temporary) `track_id`s by a common (persistent)
`object_id`: for example, `frame000031_id2_obj1.jpg` and
`frame000035_id5_obj1.jpg` — the same physical object.

**If the `_obj` suffix is absent** (e.g. `frame000031_id2.jpg`), it means that at
the moment the crop was saved, the `object_id` was not yet known. This happens only
on a track's first appearance: the object has just been detected, and ReIDManager
hasn't yet matched it against memory. Starting from the next frame, the same track
already has an `object_id`, and all subsequent crops are saved with the suffix.

---

### `ObjectEmbedder` (`src/embedder.py`)

| Method | Description |
|---|---|
| `__init__(model_name, pretrained, device, normalize)` | Loads the OpenCLIP model |
| `embed(crop)` | One crop → `np.ndarray (512,)` float32 |
| `embed_batch(crops)` | List of crops → `np.ndarray (N, 512)` float32 |

The dimension depends on the model: `ViT-B-32` → 512, `ViT-L-14` → 768.

---

### `similarity.py`

| Function | Description |
|---|---|
| `cosine_similarity(a, b)` | Scalar similarity of two vectors `[-1, 1]` |
| `cosine_similarity_batch(query, gallery)` | `query` against a matrix → `(N,)` float32 |
| `find_best_match(query, gallery, object_ids, threshold)` | → `(object_id \| None, score)` |

---

### `ObjectMemory` (`src/object_memory.py`)

| Method | Description |
|---|---|
| `add(class_name, bbox, embedding, track_id, frame_idx)` | New object → returns `object_id` |
| `update(object_id, bbox, embedding, track_id, frame_idx)` | Update a known object |
| `find_match(embedding, current_frame)` | → `(object_id \| None, score)` |
| `expire_old(current_frame)` | Deactivate old objects → list of deactivated IDs |
| `get(object_id)` | `ObjectRecord` by ID |
| `active_count()` | Number of active objects |
| `total_count()` | Total objects over the session |

---

### `ReIDManager` (`src/reid.py`)

| Method | Description |
|---|---|
| `__init__(cropper, embedder, memory, min_track_age)` | Takes ready module instances; `min_track_age` — minimum frames before registering a new object |
| `update(frame, tracked_objects, frame_idx)` | → `Dict[track_id, object_id]` |
| `get_object_id(track_id)` | Fast lookup without processing the frame |
| `active_object_count()` | Active objects in memory |
| `total_object_count()` | Total unique objects over the session |

---

### `PipelineBenchmark` (`src/benchmark.py`)

```python
from src.benchmark import PipelineBenchmark

bench = PipelineBenchmark()

bench.start_frame()

bench.start("detection")
detections, tracks = tracker.update(frame)
bench.stop("detection")

bench.start("reid")
reid_result = reid.update(frame, tracks, frame_idx)
bench.stop("reid")

bench.end_frame()

# When done:
print(bench.summary())
# {'frames': 300, 'fps': 18.4, 'detection_mean_ms': 32.1, 'reid_mean_ms': 45.7, ...}

bench.save("run_001.json")  # → outputs/benchmarks/run_001.json
```

---

## Output Folders

| Folder | Contents | When used |
|---|---|---|
| `outputs/crops/` | JPEG images of cropped objects | `ObjectCropper(save_crops=True)` |
| `outputs/embeddings/` | `.npy` embedding files | future `save_embeddings: true` option |
| `outputs/reid/` | Re-id event logs, video with persistent IDs | Stage 5 |
| `outputs/benchmarks/` | JSON performance reports | `PipelineBenchmark.save()` |

---

## Limitations of the Current Implementation

- **In-memory only** — object memory is not persisted between runs.
- **One embedding per call** — `embed()` is not batched inside `ReIDManager`. With a large number of objects in the frame this is a bottleneck. Optimization: use `embed_batch()` for all crops in a single GPU pass.
- **No conflict resolution** — if two tracks claim the same object_id, the first one wins. More sophisticated logic is a Stage 5 task.
- **CLIP is not trained on specific objects** — when working with non-standard classes (drones, specific vehicles), the re-id accuracy will be lower than that of a specialized ReID model.
