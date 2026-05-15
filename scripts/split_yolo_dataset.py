"""Split YOLO dataset from Label Studio export into train/val sets."""

import argparse
import random
import shutil
from pathlib import Path

import yaml


def split_yolo_dataset(
    input_path: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
) -> None:
    """
    Split YOLO dataset into train/val sets.

    Args:
        input_path: Path to exported dataset (with 'images' and 'labels' subdirs)
        output_dir: Output directory (will create images/train, images/val, labels/train, labels/val)
        val_ratio: Validation set ratio (default: 0.2)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    images_dir = input_path / "images"
    labels_dir = input_path / "labels"

    if not images_dir.exists():
        raise RuntimeError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise RuntimeError(f"Labels directory not found: {labels_dir}")

    # Find all image files (jpg, png)
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [
        f
        for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")

    # Verify corresponding label files exist
    pairs = []
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            raise RuntimeError(f"Label file missing for {img_file.name}: {label_file}")
        pairs.append((img_file, label_file))

    # Shuffle with fixed seed
    random.seed(42)
    random.shuffle(pairs)

    # Split into train/val
    split_idx = int(len(pairs) * (1 - val_ratio))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Create output directories
    train_images_dir = output_dir / "images" / "train"
    val_images_dir = output_dir / "images" / "val"
    train_labels_dir = output_dir / "labels" / "train"
    val_labels_dir = output_dir / "labels" / "val"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # Copy train files
    for img_file, label_file in train_pairs:
        shutil.copy2(img_file, train_images_dir / img_file.name)
        shutil.copy2(label_file, train_labels_dir / label_file.name)

    # Copy val files
    for img_file, label_file in val_pairs:
        shutil.copy2(img_file, val_images_dir / img_file.name)
        shutil.copy2(label_file, val_labels_dir / label_file.name)

    # Create data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": ["arx", "taar", "the_institute"],
    }

    yaml_file = output_dir / "data.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    # Print statistics
    print(f"Train images: {len(train_pairs)}")
    print(f"Val images  : {len(val_pairs)}")
    print(f"Total       : {len(pairs)}")
    print(f"Val ratio   : {val_ratio:.2%}")
    print(f"\nOutput: {output_dir}")
    print(f"Config: {yaml_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset from Label Studio export into train/val sets."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to exported dataset (with 'images' and 'labels' subdirs)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for split dataset",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_yolo_dataset(args.input, args.output, val_ratio=args.val_ratio)
