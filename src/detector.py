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
        self.confidence_threshold = confidence_threshold
        self.device = self._resolve_device(device)

        print(f"Device: {self.device}")

        # Load YOLO model and move it to the selected device.
        self.model = YOLO(self.model_name)
        self.model.to(self.device)

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
    def _resolve_device(device: str) -> str:
        """Resolve inference device.

        If CUDA is requested but not available, fallback to CPU instead of failing.
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"

        return device
