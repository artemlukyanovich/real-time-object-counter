# Model Export and Optimization (ONNX / TensorRT)

Speeding up inference and enabling portability to other hardware by exporting a
trained YOLO model into optimized formats.

---

## Why It's Needed

| Format | Where it runs | Speed | Purpose |
|--------|--------------|----------|------------|
| `.pt` (PyTorch) | CPU / any CUDA GPU | baseline | development, training |
| `.onnx` | **any hardware** via ONNX Runtime (CPU, NVIDIA, AMD, Intel, Apple) | medium/high | **portability** + an intermediate step toward TensorRT |
| `.engine` (TensorRT) | NVIDIA GPU only | maximum | production on a specific NVIDIA card |

> **How the roles are split in this project (ultralytics 8.0.200):**
> - **GPU on NVIDIA → `.engine` (TensorRT)** — the fastest path, validated.
> - **Portability / CPU / non-NVIDIA → `.onnx`** — runs on CPU.
>
> Important: **in the pipeline `.onnx` runs on CPU**, even with `device: "cuda"`.
> This is an Ultralytics 8.0.200 limitation: its `AutoBackend` forces ONNX (and
> other non-`pt`/`engine` formats) onto CPU
> (`ultralytics/nn/autobackend.py:103-106`). So GPU acceleration here comes only
> from TensorRT, and ONNX is used as the portable format. ONNX inference on GPU
> would require an Ultralytics upgrade — a separate task.

### ONNX vs TensorRT — what's the difference

- **ONNX** — an open, vendor-neutral exchange format. The same `.onnx` runs via
  different ONNX Runtime *execution providers*: `CPUExecutionProvider`,
  `CUDAExecutionProvider` (NVIDIA), `DmlExecutionProvider` (DirectML — AMD/Intel on
  Windows), `CoreMLExecutionProvider` (Apple), `OpenVINOExecutionProvider` (Intel).
  This is the portability layer — if the target machine has no NVIDIA, choose ONNX.
- **TensorRT** — NVIDIA's proprietary optimization. It runs only on their GPUs and
  gives maximum FPS. Ultralytics builds `.engine` **through an intermediate ONNX**,
  so understanding ONNX is the foundation for TensorRT too.

> **Important:** `.engine` is bound to the specific GPU model and TensorRT version
> it was built with. It cannot be transferred between machines and must not be
> committed — only rebuilt. `.onnx` and `.engine` are added to `.gitignore`.

---

## Installing Dependencies

The dependencies are split across two files: the portable ONNX path — in the main
`requirements.txt`, the optional NVIDIA-only TensorRT — in `requirements-tensorrt.txt`.

### ONNX path — in the main `requirements.txt` (installs on any hardware)

```
onnx==1.15.0
onnxruntime==1.16.3      # CPU runtime; this is exactly what the pipeline uses for .onnx
```

`onnxruntime` (CPU) is installed, **not** `onnxruntime-gpu`, because Ultralytics
8.0.200 runs ONNX on CPU anyway (see the note above). The GPU package gives no
gain here, and installing it also overwrites the shared `onnxruntime` package
(they share one namespace) — on NVIDIA, GPU is served by `.engine` (TensorRT).

- **On a non-NVIDIA machine** you can, if desired, install an EP for your hardware
  (`onnxruntime-directml` for AMD/Intel on Windows, `onnxruntime-openvino` for
  Intel, etc.) — the `.onnx` format itself doesn't change.

### TensorRT path — `requirements-tensorrt.txt` (NVIDIA GPU only, opt-in)

```bash
pip install --no-build-isolation -r requirements-tensorrt.txt
# Check:
python -c "import tensorrt as trt; print(trt.__version__)"
```

The `--no-build-isolation` flag is **required**. The PyPI `tensorrt` package is a
meta-package whose `setup.py` (a custom `InstallCommand`) launches a subprocess
`python -m pip install tensorrt_libs ...`. Under build isolation pip hides the
main environment's `site-packages` (along with `pip`) from this subprocess → `No
module named pip` → the build fails. `--no-build-isolation` builds with the
environment visible, and the subprocess finds `pip`. The meta-package itself is
genuinely needed: the importable `tensorrt` module is its 2-line wrapper (`from
tensorrt_bindings import *`); `tensorrt-libs`/`tensorrt-bindings` alone do **not**
provide the `tensorrt` module.

> **CUDA version mismatch — verified, works.** `tensorrt-libs 8.6.1` pulls a
> CUDA-12 runtime (`nvidia-*-cu12`, ~2.5 GB) alongside the CUDA-11.8 torch stack
> (cu118). A real `.pt → .engine` (FP16) export and `.engine` inference work — the
> two runtimes coexist thanks to different sonames. The TRT bindings need
> `libcudnn.so.8`, and **torch itself carries it** (`torch/lib/libcudnn.so.8`,
> cuDNN 8.7): it's found as long as torch is imported before tensorrt — and the
> pipeline always does so (ultralytics/torch load first). Check: `python -c
> "import torch, tensorrt"`.
>
> Do not install on machines without an NVIDIA GPU.

---

## Export: `scripts/export_model.py`

A wrapper over `YOLO(...).export(...)`. The exported file is placed next to the
source weights (Ultralytics behavior); `--output-dir` moves it to the desired
folder (e.g. `models/`).

### ONNX (portable format)

```bash
python -m scripts.export_model --weights models/yolov8n.pt --format onnx
python -m scripts.export_model --weights models/yolov8n.pt --format onnx --half
```

### TensorRT FP16 (NVIDIA GPU)

```bash
python -m scripts.export_model --weights models/yolov8n.pt --format engine --half
```

### TensorRT INT8 (maximum speed)

INT8 requires a **calibration dataset** — use the same `data.yaml` as for
training:

```bash
python -m scripts.export_model \
  --weights custom_models/final/custom_final_1/weights/best.pt \
  --format engine --int8 \
  --data data/yolo_final/<project>/data.yaml
```

### Arguments

| Argument | Default | Description |
|----------|--------------|----------|
| `--weights` | `models/yolov8n.pt` | Source `.pt` weights |
| `--format` | — (required) | `onnx` or `engine` |
| `--imgsz` | `640` | Input size baked into the model |
| `--half` | off | FP16 (faster, ~no accuracy loss) |
| `--int8` | off | INT8 quantization (`engine` only, requires `--data`) |
| `--data` | — | `data.yaml` for INT8 calibration |
| `--dynamic` | off | Dynamic input sizes (ONNX; incompatible with `--int8`) |
| `--batch` | `1` | Maximum batch |
| `--device` | `0` | Build device (`0` = first GPU, `cpu`) |
| `--output-dir` | — | Where to move the result (e.g. `models/`) |

The constraints are validated: `--int8` requires `--data` and `--format engine`;
`--half` and `--int8` are mutually exclusive; `--dynamic` is incompatible with `--int8`.

---

## Wiring into the Pipeline

It's enough to specify the path to the exported model in the config — the format
is determined by the extension automatically:

```yaml
detector:
  model: "models/yolov8n.onnx"     # or models/yolov8n.engine
  device: "cuda"                    # for .engine always GPU
```

```bash
python -m src.main --config configs/default.yaml --source data/input/video.mp4
```

### How it works internally

`ObjectDetector` determines the backend by the file suffix
(`_detect_backend`: `.pt` → `pytorch`, `.onnx` → `onnx`, `.engine` → `tensorrt`):

- `.pt` — the model is loaded and moved to the device via `.to(device)`.
- `.onnx` / `.engine` — `.to()` is **not** called (on an engine it crashes; the
  format is already bound to its runtime). The device is passed into
  `model.track(device=...)` via the tracker.
- For `.engine` the device is forced to `cuda` (there's no CPU fallback, the GPU
  is mandatory).
- For `.onnx` Ultralytics 8.0.200 forces CPU (`autobackend.py:103-106`) — `.onnx`
  always runs on CPU regardless of `device`. This is the portable path; for GPU on
  NVIDIA use `.engine`.

The model is loaded once in `ObjectDetector` and reused by the tracker
(`model.track()`), so the exported format works for both detection and tracking
without a separate conversion.

---

## Benchmark: `scripts/benchmark_backends.py`

Comparing inference speed between backends on one video (it measures the detection
forward-pass — exactly what export speeds up; the tracking overhead is the same
for all formats):

```bash
python -m scripts.benchmark_backends \
  --source data/input/video.mp4 \
  --models models/yolov8n.pt models/yolov8n.onnx models/yolov8n.engine \
  --max-frames 300 --warmup 20
```

Outputs a table: avg/p50/p95 latency (ms), FPS, and speedup relative to the first
model in the list (usually `.pt` as the baseline). The first `--warmup` frames are
not counted (lazy initialization, engine loading, CUDA warmup). Missing models are
skipped — you can compare only what's been built.

A measured result on this project (yolov8n, 640, RTX 3050 Ti Laptop):

| Backend | FPS | Avg ms | Speedup |
|---------|-----|--------|---------|
| `.pt` (pytorch, GPU) | ~147 | 6.8 | 1.00× |
| `.onnx` (onnxruntime, **CPU**) | ~28 | 36 | 0.19× |
| `.engine` (TensorRT FP16, GPU) | ~189 | 5.3 | **1.28×** |

`.onnx` is slower precisely because Ultralytics runs it on CPU (see above) — this
is expected, not an error. The fast path on NVIDIA is `.engine`.

---

## Manual Functionality Check

Step by step, what to check and how, manually. Run the commands from the project
root in the main environment. Before runs that use ONNX through Ultralytics, set
`YOLO_AUTOINSTALL=false` — otherwise Ultralytics may install the CPU `onnxruntime`
package and overwrite the installed one (see "Common Issues").

### 1. ONNX export

```bash
python -m scripts.export_model --weights models/yolov8n.pt --format onnx
ls -lh models/yolov8n.onnx
```
**Expected:** `Export complete: models/yolov8n.onnx`, the file is created (~12 MB).

### 2. TensorRT export (FP16)

```bash
python -m scripts.export_model --weights models/yolov8n.pt --format engine --half
ls -lh models/yolov8n.engine
```
**Expected:** `building FP16 engine ...`, at the end `export success ✅`,
`models/yolov8n.engine` (~9 MB). The build takes a few minutes — this is normal. A
crash with `libcudnn.so.8` means tensorrt is imported before torch — our scripts
don't do this, but in manual checks import `torch` first.

### 3. TensorRT runs on GPU (import)

```bash
python -c "import torch, tensorrt as trt; print(trt.__version__)"
```
**Expected:** prints `8.6.1` without errors (torch first — it provides `libcudnn.so.8`).

### 4. Pipeline run on each backend

Specify the model in the config (`detector.model: "models/yolov8n.engine"` or
`.onnx`, or `.pt`) or check quickly on a video. At startup the line `Detector
backend: ...` is printed in the output:

```bash
# .pt  -> backend pytorch (GPU)
# .onnx -> backend onnx (CPU, expected)
# .engine -> backend tensorrt (GPU)
python -m src.main --source data/input/street_camera_2x_speed.mp4 \
  --config configs/default.yaml
```
**Expected:** a window with detections/tracking; for `.engine` and `.pt` — high
FPS, for `.onnx` — noticeably lower (CPU). `track_id`s are stable on all backends
(a compatibility check of `model.track()` with the exported formats).

### 5. Speed comparison (benchmark)

```bash
YOLO_AUTOINSTALL=false python -m scripts.benchmark_backends \
  --source data/input/street_camera_2x_speed.mp4 \
  --models models/yolov8n.pt models/yolov8n.onnx models/yolov8n.engine \
  --max-frames 120 --warmup 15
```
**Expected:** a table; `.engine` faster than `.pt` (~1.3×), `.onnx` slower (CPU).

### 6. ONNX portability (CPU)

The fact that `.onnx` runs on CPU is itself a demonstration of portability — the
same file will run on a machine without NVIDIA. Check that `onnxruntime` really
uses CPU:
```bash
python -c "import onnxruntime as o; print(o.get_available_providers())"
```
**Expected:** the list contains `CPUExecutionProvider` (on plain `onnxruntime` —
only it; that's what we need).

### 7. INT8 engine (optional, requires a dataset)

```bash
python -m scripts.export_model \
  --weights custom_models/final/custom_final_1/weights/best.pt \
  --format engine --int8 --data data/yolo_final/<project>/data.yaml
```
**Expected:** a successful INT8 engine build. Compare detections with FP16/`.pt`
on a test set — make sure the accuracy drop is acceptable.

---

## Common Issues

**`.engine` won't run / TensorRT version error** → the engine was built for a
different TensorRT version or a different GPU. Rebuild it with
`scripts/export_model.py` on the current machine.

**ONNX runs on CPU with `device: "cuda"`** → this is expected: Ultralytics 8.0.200
forces ONNX onto CPU (`autobackend.py:103-106`). For GPU on NVIDIA use `.engine`
(TensorRT). ONNX-on-GPU would require an Ultralytics upgrade.

**After an ONNX run the environment got an `onnxruntime` package (and only the
needed one was there)** → during onnx inference Ultralytics calls
`check_requirements('onnxruntime')` and installs the CPU package. This is normal
for Path 1 (we use exactly the CPU `onnxruntime`). To stop Ultralytics from
installing anything in prod, run with the environment variable
`YOLO_AUTOINSTALL=false`.

**`Failed building wheel for tensorrt` (`No module named pip`)** → the
`--no-build-isolation` flag was forgotten. Install strictly like this:
`pip install --no-build-isolation -r requirements-tensorrt.txt`. If it still fails
with the flag — make sure the environment has `setuptools` and `wheel`
(`python -m pip install -U pip setuptools wheel`), and retry.

**INT8: fails with a data requirement** → `--int8` mandatorily needs `--data
<data.yaml>` with calibration images.
