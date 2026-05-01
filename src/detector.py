"""Object detection module using YOLOv8."""

import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO


class ObjectDetector:
    """YOLO-based object detector."""

    def __init__(self, model_name: str = "yolov8n", confidence_threshold: float = 0.5,
                 device: str = "cuda"):
        """Initialize detector.

        Args:
            model_name: YOLOv8 model variant (n, s, m, l, x)
            confidence_threshold: Detection confidence threshold
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = YOLO(f"{model_name}.pt")
        self.model.to(device)

    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int],
                                                        str, float]]:
        """Detect objects in frame.

        Args:
            frame: Input image

        Returns:
            List of (bbox, class_name, confidence) tuples
        """
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                detections.append(((x1, y1, x2, y2), class_name, confidence))

        return detections

    def detect_with_ids(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int],
                                                                  str, float]]:
        """Detect objects with class IDs.

        Args:
            frame: Input image

        Returns:
            List of (class_id, bbox, class_name, confidence) tuples
        """
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                detections.append((class_id, (x1, y1, x2, y2), class_name, confidence))

        return detections

    def get_class_names(self) -> dict:
        """Get mapping of class IDs to names."""
        return self.model.names
