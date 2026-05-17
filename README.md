# Real-time Object Counter

## Goal
Real-time object detection, tracking and counting pipeline.

## Tech Stack
Python, PyTorch, OpenCV, YOLO, ByteTrack.

## Features
- Webcam/video input
- Real-time detection
- Object tracking
- Line crossing counter
- FPS and latency metrics

## What was implemented

### CV Pipeline
- Tracking algorithm integration: ByteTrack, BoT-SORT, BoT-SORT + Re-ID (OSNet)
- Fully configurable pipeline via YAML: model, tracking algorithm, confidence thresholds, counting zones, visualization parameters
- 5 experiment configuration presets for different scenarios

### Dataset Preparation
- Video data collection, frame extraction (`scripts/extract_frames.py`) with batch support for multiple videos into a single directory
- Manual annotation of a dataset subset in Label Studio (with a CORS server for local file access)
- Bootstrap YOLO model training for semi-automatic annotation (auto-labeling)
- Automatic bbox generation using the trained model (`yolo detect predict`)
- Conversion of YOLO predictions to Label Studio format (`scripts/convert_yolo_predictions_to_label_studio.py`)
- Local tool for fast review and validation of auto-generated labels (`scripts/review_yolo_labels.py`)
- Annotation validation and correction in Label Studio
- Dataset splitting into train/val sets in YOLO format (`scripts/split_yolo_dataset.py`)

### Model Training
- Bootstrap YOLO model training (YOLOv8n, 20 epochs) for initial label generation
- Final YOLO model training (YOLOv8n, 80 epochs) on the full dataset (3 classes: arx, taar, the_institute)
- Custom model integration into the main pipeline via config (`detector.model`)

See `docs/dataset_preparation.md` for dataset preparation and training details.

## Demo
GIF/video/screenshot

## How to run

### Main counter app

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m src.main --source 0
```

### Annotation tools (separate env)

```bash
# Label Studio or labelImg (in annotations environment)
pip install -r requirements-annotations.txt
label-studio   # or labelImg
```

See `docs/how_to_run.md` for detailed setup.

## Results
- FPS: ...
- GPU: RTX 3050 Ti
- Model: YOLOv8n / YOLO11n