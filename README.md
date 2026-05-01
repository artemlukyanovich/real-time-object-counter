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

## Demo
GIF/video/screenshot

## How to run
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m src.main --source 0

## Results
- FPS: ...
- GPU: RTX 3050 Ti
- Model: YOLOv8n / YOLO11n