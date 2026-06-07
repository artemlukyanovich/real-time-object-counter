# Running the Project

---

## Installation

### Main environment (object-counter)

#### Conda (recommended)

```bash
conda create -n object-counter python=3.10
conda activate object-counter
pip install -r requirements.txt
```

#### pip (without conda)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Separate annotation environment (annotations)

Label Studio and labelImg require a separate environment due to dependency conflicts.

#### Conda

```bash
conda create -n annotations python=3.10
conda activate annotations
pip install -r requirements-annotations.txt
```

#### pip

```bash
python -m venv .venv-annotations
source .venv-annotations/bin/activate   # Linux/macOS
# .venv-annotations\Scripts\activate    # Windows

pip install -r requirements-annotations.txt
```

**Main dependencies:**
- `ultralytics` — YOLOv8 + automatic weight download
- `opencv-python` — video capture and drawing
- `torch` — inference (CPU or CUDA)
- `pyyaml` — config parsing

On the first run, `yolov8n.pt` is downloaded automatically (~6 MB). Alternatively, place the file in the project root manually.

---

## Annotation (Label Studio + labelImg)

After installation in the `annotations` environment:

### Label Studio

```bash
conda activate annotations  # or source .venv-annotations/bin/activate
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
label-studio
```

A web interface opens at `http://localhost:8080`. To access local images from the browser, also run in a separate terminal:

```bash
python -m scripts.cors_http_server
```

The server is available at `http://localhost:9000`.

### labelImg

```bash
conda activate annotations
labelImg
```

Opens a GUI for fast annotation of individual images. Used to create YOLO-format datasets.

---

## Running

### Webcam (index 0)

```bash
python -m src.main --source 0
```

### Video file

```bash
python -m src.main --source path/to/video.mp4
```

### With a specified config

```bash
python -m src.main --config configs/default.yaml --source 0
```

### Video file with a custom config

```bash
python -m src.main --config configs/default.yaml --source path/to/video.mp4
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|----------|-----|-------------|----------|
| `--source` | int or str | from config (`video.source`) | Video source: `0` — camera, a path — file |
| `--config` | str | `configs/default.yaml` | Path to the YAML configuration |
| `--half` | flag | from config (`detector.half`) | Enables FP16 inference. Effective only for `.pt` models on CUDA |

`--source` overrides `video.source` from the config. `--half` overrides `detector.half` (if the flag is not specified — the config value is used).

```bash
# FP16 inference (~1.5-2x speedup on a GPU with Tensor Cores):
python -m src.main --source data/input/video.mp4 --half
```

---

## Controls During Runtime

| Key | Action |
|---------|----------|
| `q` | Quit |
| `ESC` | Quit |

---

## Configuration via config.yaml

All main parameters are set in `configs/default.yaml`:

```yaml
detector:
  model: "models/yolov8n.pt"          # Switch to yolov8s.pt for higher accuracy
  confidence_threshold: 0.5
  device: "cuda"               # "cpu" if there's no GPU

display:
  show_detections: true
  show_tracking_ids: true
  show_counts: true
```

See `docs/config.md` for details.

---

## Running with a Custom Model

A model trained on custom data is specified in the config:

```yaml
detector:
  model: "custom_models/final/custom_final_1/weights/best.pt"
  allowed_classes: ["arx", "taar", "the_institute"]
```

```bash
python -m src.main --config configs/experiments/custom_objects.yaml
python -m src.main --source data/input/video.mp4 --config configs/experiments/custom_objects.yaml
```

Dataset preparation and model training are described in `docs/dataset_preparation.md`.

---

## Output

Upon completion, a summary is printed to stdout:

```
=== Final Metrics ===
FPS: 28.4
Total frames: 1250
Avg detection time: 12.3 ms
Avg tracking time: 1.1 ms
```

Also the final per-class counts (if the Counter is enabled).

---

## Common Issues

**Camera won't open:**
```
VideoSource: failed to open source 0
```
→ Check the camera index. Try `--source 1` or `--source 2`.

**CUDA is unavailable, crashes with an error:**
→ In the config set `detector.device: "cpu"`. The detector falls back automatically, but an explicit setting is more reliable.

**Low FPS:**
→ Lower the resolution (`video.frame_width: 640`, `video.frame_height: 480`), switch the model to `yolov8n.pt`, disable extra rendering layers (`show_detections: false`).

**Model not found:**
```
FileNotFoundError: yolov8n.pt
```
→ Ultralytics downloads the model automatically on the first run. If there's no internet — copy the `.pt` file into the project root manually.
