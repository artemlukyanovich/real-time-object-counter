# Project Overview — Real-time Object Counter

## Project Purpose

This project is a real-time computer vision system designed to:

- detect objects in a video stream
- track them across frames
- count objects crossing a virtual line
- measure performance (FPS, latency)

The project is intended as a learning and engineering exercise in:

- real-time CV pipelines
- performance optimization
- edge AI systems

---

## Core Features

- Real-time object detection (YOLO)
- Object tracking (ByteTrack)
- Line-crossing object counting
- FPS and latency monitoring
- Support for webcam and video input

---

## Tech Stack

- Python
- PyTorch
- OpenCV
- YOLO (Ultralytics)
- ByteTrack
- NumPy

---

## Architecture Summary

Pipeline:

VideoSource → Detector → Tracker → Counter → Renderer

Each component is modular and can be replaced independently.

---

## Key Concepts

### Real-time Processing
The system must maintain stable FPS (target: 20–30 FPS).

---

### Latency Awareness
Each frame must be processed quickly to avoid delays.

---

### Tracking Consistency
Objects must retain consistent IDs across frames.

---

### Event Detection
Counting is based on detecting line crossing events.

---

## Constraints

- Runs locally on a laptop (RTX 3050 Ti GPU)
- Limited VRAM (~4GB)
- Must avoid high memory usage
- Must maintain real-time performance

---

## Development Guidelines (for AI Agents)

When modifying or generating code:

1. Preserve modular structure:
   - do not merge components into a single file

2. Avoid blocking operations:
   - keep pipeline responsive

3. Optimize for performance:
   - avoid unnecessary copies of frames
   - minimize CPU-GPU transfers

4. Keep code simple and readable:
   - prefer clarity over premature optimization

5. Use structured outputs:
   - detections should be consistent (dict or dataclass)

---

## Non-Goals

- High-accuracy detection is NOT the priority
- Training models from scratch is NOT required
- Cloud deployment is NOT required

---

## Future Goals

- Export model to ONNX
- Optimize inference (TensorRT)
- Add async processing
- Integrate with drone simulation (PX4 / AirSim)

---

## Example Use Case

A camera observes a street:

- detects cars and people
- tracks each object
- counts how many cross a virtual line

---

## Expected Output

- Video with bounding boxes
- Object IDs
- Counter (IN / OUT)
- FPS and latency displayed on screen