"""Utility functions."""

import cv2
import numpy as np
from typing import Tuple, List


def get_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Calculate centroid of bounding box."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy


def calculate_iou(box1: Tuple[int, int, int, int],
                  box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def distance(point1: Tuple[int, int],
             point2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def point_side_of_line(
    point: Tuple[int, int],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
) -> float:
    """Return the signed area of the cross product for a point relative to a directed line.

    Positive means the point is on the left side of the line (from line_start to line_end),
    negative means right side, zero means the point is on the line.
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    return float((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))


def is_point_in_polygon(point: Tuple[int, int],
                        polygon: List[Tuple[int, int]]) -> bool:
    """Check if point is inside polygon using ray casting algorithm."""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside
