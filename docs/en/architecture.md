# Architecture — Real-time Object Counter

## Overview

The project implements a real-time computer vision pipeline for object detection, tracking, and counting from a video stream (webcam or video file).

The system is designed as a modular pipeline with clear separation of concerns to support scalability, performance tuning, and future extensions (edge deployment, drone integration).

---

## High-Level Pipeline

VideoSource → Detector → Tracker → Counter → Renderer → Output

---

## Data Flow

1. VideoSource reads frames from:
   - webcam
   - video file

2. Detector processes frames:
   - runs YOLO model inference
   - outputs bounding boxes, classes, confidence scores

3. Tracker assigns IDs:
   - tracks objects across frames
   - ensures temporal consistency

4. Counter processes tracked objects:
   - detects line crossing events
   - increments counters (IN / OUT)

5. Renderer overlays:
   - bounding boxes
   - object IDs
   - counters
   - metrics (FPS, latency)

6. Output:
   - displayed in real-time window
   - optionally saved as video

---

## Modules

### video_source.py
Responsible for:
- reading frames
- handling input source (webcam/video)
- basic buffering

---

### detector.py
Responsible for:
- loading YOLO model
- running inference
- returning detections in structured format

---

### tracker.py
Responsible for:
- assigning unique IDs to objects
- maintaining tracking state across frames

Uses:
- ByteTrack (or similar algorithm)

---

### counter.py
Responsible for:
- detecting line crossing
- counting objects

Core logic:
- track previous vs current position
- detect intersection with counting line

---

### renderer.py
Responsible for:
- drawing bounding boxes
- drawing IDs
- rendering counters and metrics

---

### metrics.py
Responsible for:
- FPS calculation
- latency measurement
- optional performance logs

---

### config.py
Responsible for:
- loading config (YAML)
- centralizing parameters

---

## Key Design Principles

### 1. Modular Architecture
Each component is independent and replaceable.

---

### 2. Real-time Constraints
The system prioritizes:
- low latency
- stable FPS
- minimal frame drops

---

### 3. Observability
Metrics are first-class citizens:
- FPS
- inference time
- queue size (future)

---

### 4. Extensibility
Future upgrades:
- ONNX / TensorRT
- async pipeline
- drone integration (PX4, AirSim)
- edge deployment

---

## Future Improvements

- Multi-threaded pipeline
- Frame queue buffering
- Model optimization
- Edge deployment support
- Integration with drone control systems