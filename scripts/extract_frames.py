"""Extract every N-th frame from a video file and save as JPG images."""

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, step: int, dry_run: bool = False) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("frame_*.jpg")) if output_dir.exists() else []
    start_index = max(
        (int(p.stem.split("_")[1]) for p in existing), default=0
    )

    total_frames = 0
    saved_frames = start_index

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if total_frames % step == 0:
            saved_frames += 1
            if not dry_run:
                filename = output_dir / f"frame_{saved_frames:04d}.jpg"
                cv2.imwrite(str(filename), frame)

        total_frames += 1

    cap.release()

    if dry_run:
        print(f"Dry run   : {total_frames} frames in video, step={step}")
        print(f"Would save: {saved_frames} frames → {output_dir}")
    else:
        print(f"Processed : {total_frames} frames")
        print(f"Saved     : {saved_frames} frames → {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract every N-th frame from a video.")
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the input video file (default location: data/raw_videos/)",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Directory to save extracted frames (default location: data/frames/<name>/)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=15,
        help="Save every N-th frame (default: 15)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only: show how many frames would be saved without writing files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(args.video, args.output, args.step, dry_run=args.dry_run)
