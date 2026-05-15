"""Quick local viewer and cleaner for YOLO bbox labels.

Usage:
    python -m scripts.review_yolo_labels <images_dir> <labels_dir> [options]

Keys:
    n / Right Arrow  — next image
    p / Left Arrow   — previous image
    d                — delete label file (moves to deleted_labels/)
    m                — mark as needs manual review
    k                — mark as ok
    s                — skip (no action)
    q / Esc          — quit
    h                — print key hints to console
"""

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = {
    0: "arx",
    1: "taar",
    2: "the_institute",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MAX_WIDTH = 1280
MAX_HEIGHT = 800

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.6
TEXT_COLOR = (255, 255, 255)
DARK_TEXT_COLOR = (0, 0, 0)
PANEL_BG_COLOR = (0, 0, 0)
STATUS_COLORS = {
    "OK": (0, 200, 0),
    "NEEDS REVIEW": (0, 165, 255),
    "DELETED": (0, 0, 220),
}

KEY_HINTS = (
    "n/--> next | p/<-- prev | d delete | m review | k ok | s skip | q quit | h help"
)


def _generate_colors(n: int = 256):
    colors = []
    for i in range(n):
        hue = (i * 180 // n) % 180
        hsv = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr))
    return colors


_COLORS = _generate_colors()


def _load_class_names(classes_file: Path | None) -> dict[int, str]:
    if classes_file is None or not classes_file.exists():
        return CLASS_NAMES.copy()
    names = {}
    with open(classes_file) as f:
        for i, line in enumerate(f):
            name = line.strip()
            if name:
                names[i] = name
    return names


def _find_images(images_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    images = [
        p for p in sorted(images_dir.glob(pattern))
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images


def _read_labels(label_path: Path, min_conf: float | None) -> list[dict]:
    bboxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            conf = float(parts[5]) if len(parts) >= 6 else None
            if min_conf is not None and conf is not None and conf < min_conf:
                continue
            bboxes.append(
                {
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "conf": conf,
                }
            )
    return bboxes


def _render_scale(img_w: int) -> tuple[float, int]:
    """Return (font_scale, thickness) proportional to image width.

    Calibrated so that at 1280 px wide the result equals FONT_SIZE / thickness=1.
    """
    s = img_w / 1280
    return FONT_SIZE * s, max(1, round(s))


def _draw_label(img, text: str, x: int, y: int, bg_color: tuple,
                font_scale: float, thickness: int) -> None:
    """Colored background rectangle + dark text (same as renderer.py _draw_label)."""
    text_size, _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    tw, th = text_size
    pad = max(4, round(font_scale * 6))
    label_y = max(y, th + pad * 2)
    cv2.rectangle(img, (x, label_y - th - pad * 2), (x + tw + pad * 2, label_y), bg_color, -1)
    cv2.putText(img, text, (x + pad, label_y - pad // 2 - 1),
                FONT, font_scale, DARK_TEXT_COLOR, thickness, cv2.LINE_AA)


def _draw_bboxes(img, bboxes: list[dict], class_names: dict[int, str],
                 font_scale: float, thickness: int):
    h, w = img.shape[:2]
    for bbox in bboxes:
        cid = bbox["class_id"]
        color = _COLORS[hash(class_names.get(cid, str(cid))) % len(_COLORS)]
        xc, yc = bbox["x_center"], bbox["y_center"]
        bw, bh = bbox["width"], bbox["height"]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness + 1)
        label = class_names.get(cid, str(cid))
        if bbox["conf"] is not None:
            label += f" {bbox['conf']:.2f}"
        _draw_label(img, label, x1, y1, bg_color=color,
                    font_scale=font_scale, thickness=thickness)


def _draw_overlay(img, index: int, total: int, filename: str,
                  bbox_count: int, has_label: bool, status: str | None,
                  font_scale: float, thickness: int):
    h, w = img.shape[:2]

    bbox_text = f"bboxes: {bbox_count}" if has_label else "NO LABEL"
    lines = [
        f"{index + 1}/{total}  {filename}",
        bbox_text,
        KEY_HINTS,
    ]
    if status:
        lines.append(f"[{status}]")

    row_h = max(22, round(font_scale * 36))
    panel_h = len(lines) * row_h + 10
    panel_w = w

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), PANEL_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    y = row_h - 4
    for i, line in enumerate(lines):
        color = TEXT_COLOR
        if i == 1 and not has_label:
            color = (0, 80, 255)  # red-ish for NO LABEL
        elif i == len(lines) - 1 and status:
            color = STATUS_COLORS.get(status, TEXT_COLOR)
        cv2.putText(img, line, (8, y), FONT, font_scale, color, thickness, cv2.LINE_AA)
        y += row_h


def _open_csv_log(log_path: Path):
    is_new = not log_path.exists()
    f = open(log_path, "a", newline="")
    writer = csv.DictWriter(
        f,
        fieldnames=["timestamp", "image_path", "label_path",
                    "action", "bbox_count", "note"],
    )
    if is_new:
        writer.writeheader()
    return f, writer


def _log(writer, image_path: Path, label_path: Path | None,
         action: str, bbox_count: int, note: str = ""):
    writer.writerow(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image_path": str(image_path),
            "label_path": str(label_path) if label_path else "",
            "action": action,
            "bbox_count": bbox_count,
            "note": note,
        }
    )


def _append_to_list(path: Path, filename: str):
    with open(path, "a") as f:
        f.write(filename + "\n")


def print_hints():
    print(
        "\nKey bindings:\n"
        "  n / Right Arrow  — next image\n"
        "  p / Left Arrow   — previous image\n"
        "  d                — delete label (moves to deleted_labels/)\n"
        "  m                — mark as needs manual review\n"
        "  k                — mark as ok\n"
        "  s                — skip (no action)\n"
        "  q / Esc          — quit\n"
        "  h                — print this help\n"
    )


def review(
    images_dir: Path,
    labels_dir: Path,
    output_review_dir: Path,
    classes_file: Path | None,
    start_name: str | None,
    min_conf: float | None,
    recursive: bool,
    dry_run: bool,
) -> None:
    class_names = _load_class_names(classes_file)

    images = _find_images(images_dir, recursive)
    if not images:
        print(f"No images found in {images_dir}")
        return

    deleted_dir = output_review_dir / "deleted_labels"
    output_review_dir.mkdir(parents=True, exist_ok=True)
    deleted_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_review_dir / "review_log.csv"
    needs_review_path = output_review_dir / "needs_manual_review.txt"
    ok_path = output_review_dir / "ok.txt"

    log_file, log_writer = _open_csv_log(log_path)

    # Determine start index
    start_idx = 0
    if start_name:
        for i, p in enumerate(images):
            if p.name == start_name or p.stem == start_name:
                start_idx = i
                break

    # Track per-image status for overlay (in-memory, this session only)
    statuses: dict[int, str] = {}

    # Counters
    counts = {"ok": 0, "needs_review": 0, "deleted": 0, "missing": 0, "viewed": 0}

    print(f"Found {len(images)} images. Starting from index {start_idx}.")
    print_hints()

    cv2.namedWindow("YOLO Label Review", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Label Review", MAX_WIDTH, MAX_HEIGHT)

    idx = start_idx
    while True:
        img_path = images[idx]
        label_path = labels_dir / f"{img_path.stem}.txt"
        has_label = label_path.exists()

        bboxes = []
        if has_label:
            try:
                bboxes = _read_labels(label_path, min_conf)
            except Exception as e:
                print(f"  [warn] Could not read {label_path}: {e}")
        else:
            counts["missing"] += 1  # counted each time viewed; reset-safe

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [warn] Could not open image {img_path}, skipping.")
            idx = (idx + 1) % len(images)
            continue

        counts["viewed"] += 1
        # Draw on full-resolution image so text stays sharp at any window size.
        font_scale, thickness = _render_scale(img.shape[1])

        _draw_bboxes(img, bboxes, class_names, font_scale, thickness)
        _draw_overlay(
            img, idx, len(images), img_path.name,
            len(bboxes), has_label, statuses.get(idx),
            font_scale, thickness,
        )

        cv2.imshow("YOLO Label Review", img)
        key = cv2.waitKey(0) & 0xFF
        key_ex = cv2.waitKeyEx(1) & 0xFFFF  # for arrow keys

        # Determine action from key
        action = None

        if key in (ord("q"), 27):  # q or Esc
            print("Quit.")
            break
        elif key in (ord("n"), 83) or key_ex in (65363, 0xFF53):  # n or Right
            idx = min(idx + 1, len(images) - 1)
            continue
        elif key in (ord("p"), 81) or key_ex in (65361, 0xFF51):  # p or Left
            idx = max(idx - 1, 0)
            continue
        elif key == ord("h"):
            print_hints()
            continue
        elif key == ord("s"):
            action = "skipped"
            idx = min(idx + 1, len(images) - 1)
        elif key == ord("d"):
            action = "deleted_label"
            if has_label:
                dest = deleted_dir / label_path.name
                if dry_run:
                    print(f"  [dry-run] Would move {label_path} → {dest}")
                else:
                    shutil.move(str(label_path), dest)
                    print(f"  Moved label → {dest}")
                counts["deleted"] += 1
                statuses[idx] = "DELETED"
            else:
                print(f"  No label to delete for {img_path.name}")
            idx = min(idx + 1, len(images) - 1)
        elif key == ord("m"):
            action = "needs_manual_review"
            if dry_run:
                print(f"  [dry-run] Would mark {img_path.name} as needs_manual_review")
            else:
                _append_to_list(needs_review_path, img_path.name)
                print(f"  Marked for manual review: {img_path.name}")
            counts["needs_review"] += 1
            statuses[idx] = "NEEDS REVIEW"
            idx = min(idx + 1, len(images) - 1)
        elif key == ord("k"):
            action = "ok"
            if dry_run:
                print(f"  [dry-run] Would mark {img_path.name} as ok")
            else:
                _append_to_list(ok_path, img_path.name)
                print(f"  Marked ok: {img_path.name}")
            counts["ok"] += 1
            statuses[idx] = "OK"
            idx = min(idx + 1, len(images) - 1)
        else:
            continue

        if action:
            if not dry_run:
                _log(log_writer, img_path, label_path if has_label else None,
                     action, len(bboxes))
                log_file.flush()

    cv2.destroyAllWindows()
    log_file.close()

    # Summary
    missing_count = sum(
        1 for i, p in enumerate(images)
        if not (labels_dir / f"{p.stem}.txt").exists()
    )
    print(
        f"\n--- Summary ---\n"
        f"  Total viewed:           {counts['viewed']}\n"
        f"  Marked ok:              {counts['ok']}\n"
        f"  Marked for manual rev:  {counts['needs_review']}\n"
        f"  Deleted labels:         {counts['deleted']}\n"
        f"  Missing labels:         {missing_count}\n"
        f"  Review log:             {log_path}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick local viewer and cleaner for YOLO bbox labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("images_dir", type=Path, help="Directory with images")
    parser.add_argument("labels_dir", type=Path, help="Directory with YOLO .txt label files")
    parser.add_argument(
        "--classes", type=Path, default=None,
        metavar="FILE",
        help="Path to classes.txt (one class name per line)",
    )
    parser.add_argument(
        "--output-review-dir", type=Path,
        default=Path("outputs/label_review"),
        metavar="DIR",
        help="Directory for review outputs (default: outputs/label_review)",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        metavar="FILENAME",
        help="Filename (or stem) to start from",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        metavar="THRESHOLD",
        help="Minimum confidence to display (requires 6th field in label file)",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search images recursively",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Do not delete/move/write anything, only print actions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.images_dir.exists():
        print(f"Error: images_dir does not exist: {args.images_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.labels_dir.exists():
        print(f"Error: labels_dir does not exist: {args.labels_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("[dry-run mode] No files will be modified.")

    review(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_review_dir=args.output_review_dir,
        classes_file=args.classes,
        start_name=args.start,
        min_conf=args.conf,
        recursive=args.recursive,
        dry_run=args.dry_run,
    )
