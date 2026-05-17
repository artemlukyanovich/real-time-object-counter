# Dataset Preparation and Model Training

Full cycle description: from raw videos to a trained YOLO model for custom object detection.

---

## Pipeline Overview

```
Videos → Frame extraction → Manual annotation (Label Studio)
       → Bootstrap model → Auto-labeling → Label review
       → Final dataset → Final model training
```

---

## 1. Frame Extraction from Video

`scripts/extract_frames.py` — saves every N-th frame as JPG.

```bash
python -m scripts.extract_frames data/raw_videos/taar_1.mp4 data/frames/taar --step 30
python -m scripts.extract_frames data/raw_videos/the_institute_3.mp4 data/frames/the_institute --step 30 --prefix the_institute

# Preview without saving
python -m scripts.extract_frames data/raw_videos/taar_1.mp4 data/frames/taar --step 30 --dry-run
```

Numbering continues from the last existing file — multiple videos can be processed into the same directory without overwriting.

See `docs/en/extract_frames.md` for details.

---

## 2. Manual Annotation in Label Studio

Label Studio requires the separate `annotations` environment.

### Starting Label Studio

```bash
conda activate annotations
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
label-studio
```

Web interface available at `http://localhost:8080`.

### CORS Server for Local Images

Label Studio cannot read files directly from disk in the browser. Run in a separate terminal:

```bash
python -m scripts.cors_http_server
```

The server serves the project root at `http://localhost:9000` with CORS headers. Images are referenced with URLs like `http://localhost:9000/data/frames/...`.

---

## 3. Bootstrap Model for Semi-Automatic Annotation

### Dataset Split (bootstrap)

Export annotations from Label Studio in YOLO format, then split into train/val:

```bash
python -m scripts.split_yolo_dataset \
  data/bootstrap_export/project-6-at-2026-05-15-05-52-0a94b509 \
  data/yolo_bootstrap/project-6-at-2026-05-15-05-52-0a94b509 \
  --val-ratio 0.2
```

Result: directory with `train/` and `val/` subdirectories and a `data.yaml` file.

### Bootstrap Model Training

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/yolo_bootstrap/project-6-at-2026-05-15-05-52-0a94b509/data.yaml \
  epochs=20 \
  imgsz=640 \
  batch=8 \
  project=custom_models/bootstrap \
  name=custom_bootstrap_1
```

Weights are saved to `custom_models/bootstrap/custom_bootstrap_1/weights/best.pt`.

---

## 4. Auto-labeling: Generating Bboxes for the Full Dataset

```bash
yolo detect predict \
  model=custom_models/bootstrap/custom_bootstrap_1/weights/best.pt \
  source=data/frames/all \
  conf=0.25 \
  save=True \
  save_txt=True \
  project=custom_models/bootstrap_predictions \
  name=custom_bootstrap_1_frames_all
```

Predictions (`.txt`) are saved to `custom_models/bootstrap_predictions/custom_bootstrap_1_frames_all/labels/`.

---

## 5. Review of Auto-Generated Annotations

### Local Review

`scripts/review_yolo_labels.py` — OpenCV-based interactive viewer:

```bash
python -m scripts.review_yolo_labels \
  data/frames/all \
  custom_models/bootstrap_predictions/custom_bootstrap_1_frames_all/labels \
  --output-review-dir outputs/label_review \
  --dry-run
```

Remove `--dry-run` for active mode (with saved decisions).

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `n` | Next image |
| `p` | Previous image |
| `k` | OK — annotation is correct |
| `d` | Delete annotation for this image |
| `m` | Flag for manual review |
| `q` | Quit |

Results are logged to CSV; `ok.txt` and `needs_manual_review.txt` files are generated.

### Import to Label Studio for Correction

Convert YOLO predictions to Label Studio format:

```bash
python -m scripts.convert_yolo_predictions_to_label_studio \
  data/frames/all \
  custom_models/bootstrap_predictions/custom_bootstrap_1_frames_all/labels \
  outputs/predictions_all.json
```

Import `outputs/predictions_all.json` into Label Studio as pre-annotations, review and correct errors, then export the final dataset.

---

## 6. Final Model

### Final Dataset Split

```bash
python -m scripts.split_yolo_dataset \
  data/bootstrap_export/project-7-at-2026-05-15-14-03-f712bcc8 \
  data/yolo_final/project-7-at-2026-05-15-14-03-f712bcc8 \
  --val-ratio 0.2
```

### Final Model Training

```bash
yolo detect train \
  model=yolov8n.pt \
  data=data/yolo_final/project-7-at-2026-05-15-14-03-f712bcc8/data.yaml \
  epochs=80 \
  imgsz=640 \
  batch=8 \
  project=custom_models/final \
  name=custom_final_1
```

Weights are saved to `custom_models/final/custom_final_1/weights/best.pt`.

---

## 7. Running with a Custom Model

In the config (`configs/experiments/custom_objects.yaml`), set the path to the weights:

```yaml
detector:
  model: "custom_models/final/custom_final_1/weights/best.pt"
  allowed_classes: ["arx", "taar", "the_institute"]
```

Run:

```bash
python -m src.main --config configs/experiments/custom_objects.yaml
python -m src.main --source data/input/video.mp4 --config configs/experiments/custom_objects.yaml
```

---

## Classes

The model is trained on 3 custom classes:

| ID | Class name |
|----|-----------|
| 0 | arx |
| 1 | taar |
| 2 | the_institute |
