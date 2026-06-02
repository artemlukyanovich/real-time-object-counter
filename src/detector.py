"""Object detection module using YOLOv8."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from ultralytics import YOLO


# Detection format used across the project:
# ((x1, y1, x2, y2), class_name, confidence)
Detection = Tuple[Tuple[int, int, int, int], str, float]


class ObjectDetector:
    """YOLO-based object detector."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        """Initialize detector.

        Args:
            model_name: YOLO model file or model name, e.g. "yolov8n" or "yolov8n.pt".
            confidence_threshold: Minimum confidence score for detections.
            device: Device used for inference ("cuda" or "cpu").
        """
        self.model_name = self._normalize_model_name(model_name)
        self.backend = self._detect_backend(self.model_name)
        self.confidence_threshold = confidence_threshold
        self.device = self._resolve_device(device, self.backend)

        print(f"Detector backend: {self.backend} | device: {self.device}")

        # Load YOLO model. Exported backends (.onnx / .engine) are already bound
        # to their own runtime/device, so only native PyTorch (.pt) models need
        # an explicit .to(device) move — calling it on an engine raises. They also
        # can't auto-guess the task, so declare it explicitly (this project is
        # detection-only) to silence the "Unable to guess model task" warning.
        if self.backend == "pytorch":
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
        else:
            self.model = YOLO(self.model_name, task="detect")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a frame.

        Args:
            frame: Input image/frame in OpenCV BGR format.

        Returns:
            List of detections in format:
            [((x1, y1, x2, y2), class_name, confidence), ...]
        """
        # Run YOLO inference on a single frame.
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )

        detections: List[Detection] = []

        # Convert YOLO result objects to a simple project-level format.
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                detections.append(((x1, y1, x2, y2), class_name, confidence))

        return detections

    def get_class_names(self) -> Dict[int, str]:
        """Get mapping of class IDs to class names."""
        return self.model.names

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Normalize model name.

        Supports both:
        - "yolov8n"
        - "yolov8n.pt"

        This prevents accidental names like "yolov8n.pt.pt".
        """
        model_path = Path(model_name)

        if model_path.suffix:
            return model_name

        return f"{model_name}.pt"

    @staticmethod
    def _detect_backend(model_name: str) -> str:
        """Infer the inference backend from the model file suffix.

        Returns one of:
        - "pytorch"  for .pt   (native, CPU or CUDA)
        - "onnx"     for .onnx  (portable, run via ONNX Runtime)
        - "tensorrt" for .engine (NVIDIA GPU only)
        """
        suffix = Path(model_name).suffix.lower()
        if suffix == ".engine":
            return "tensorrt"
        if suffix == ".onnx":
            return "onnx"
        return "pytorch"

    @staticmethod
    def _resolve_device(device: str, backend: str = "pytorch") -> str:
        """Resolve inference device, accounting for the model backend.

        - TensorRT engines are GPU-only and bound to the GPU they were built on,
          so CUDA is required (no CPU fallback).
        - PyTorch / ONNX fall back to CPU when CUDA is requested but unavailable.
          For ONNX, GPU execution additionally needs onnxruntime-gpu installed;
          otherwise ONNX Runtime silently runs on CPU regardless of this value.
        """
        cuda_available = torch.cuda.is_available()

        if backend == "tensorrt":
            if not cuda_available:
                raise RuntimeError(
                    "TensorRT engine requires an NVIDIA GPU, but CUDA is unavailable."
                )
            if device == "cpu":
                print("TensorRT engine is GPU-only; ignoring device='cpu'.")
            return "cuda"

        if device == "cuda" and not cuda_available:
            print("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"

        return device
