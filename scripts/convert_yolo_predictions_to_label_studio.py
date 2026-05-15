"""Convert YOLO predictions to Label Studio format."""

import json
import argparse
import uuid
from pathlib import Path

from PIL import Image


CLASS_NAMES = {
    0: "arx",
    1: "taar",
    2: "the_institute",
}

# Base URL of the CORS HTTP server (scripts/cors_http_server.py)
IMAGE_SERVER_URL = "http://localhost:9000"


def _clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp value to [min_val, max_val] and round to 6 decimal places."""
    return round(max(min_val, min(max_val, value)), 6)


def convert_yolo_to_label_studio(
    images_dir: Path,
    labels_dir: Path,
    output_path: Path,
) -> None:
    """Convert YOLO predictions to Label Studio format."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_path = Path(output_path)

    project_root = Path(__file__).resolve().parent.parent
    tasks = []

    for img_file in sorted(images_dir.glob("*.jpg")) + sorted(
        images_dir.glob("*.png")
    ):
        label_file = labels_dir / f"{img_file.stem}.txt"

        # Get image dimensions
        with Image.open(img_file) as img:
            img_width, img_height = img.size

        # Read YOLO labels and convert to predictions
        predictions = []
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert YOLO (center, normalized) to Label Studio (top-left, percent)
                    # Clamp to [0, 100] to avoid out-of-bounds values from edge detections
                    x = _clamp((x_center - width / 2) * 100)
                    y = _clamp((y_center - height / 2) * 100)
                    w = _clamp(width * 100)
                    h = _clamp(height * 100)

                    predictions.append(
                        {
                            "id": str(uuid.uuid4()),
                            "type": "rectanglelabels",
                            "value": {
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "rotation": 0,
                                "rectanglelabels": [CLASS_NAMES[class_id]],
                            },
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": img_width,
                            "original_height": img_height,
                        }
                    )

        image_url = f"{IMAGE_SERVER_URL}/{img_file.resolve().relative_to(project_root)}"
        task = {
            "data": {"image": image_url},
        }

        if predictions:
            task["predictions"] = [
                {
                    "model_version": "yolo_bootstrap",
                    "score": 0.9,
                    "result": predictions,
                }
            ]

        tasks.append(task)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Converted {len(tasks)} tasks → {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO predictions to Label Studio format."
    )
    parser.add_argument(
        "images",
        type=Path,
        help="Directory with images",
    )
    parser.add_argument(
        "labels",
        type=Path,
        help="Directory with YOLO labels (.txt files)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output JSON path for Label Studio import",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_yolo_to_label_studio(args.images, args.labels, args.output)