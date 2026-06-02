"""Export a YOLO .pt model to ONNX or TensorRT for faster / portable inference.

Why
---
- **ONNX** is an open, vendor-neutral format. The same .onnx file runs through
  ONNX Runtime on many backends (CPU, CUDA, DirectML/AMD, CoreML/Apple,
  OpenVINO/Intel), so it is the portable target for "run on other hardware".
- **TensorRT** (.engine) is NVIDIA-only and gives the best FPS on this GPU.
  Ultralytics builds the engine *through* ONNX internally, so ONNX is also the
  intermediate step on the way to TensorRT.

Examples
--------
    # ONNX (portable across hardware via ONNX Runtime)
    python -m scripts.export_model --weights models/yolov8n.pt --format onnx

    # ONNX with FP16 weights
    python -m scripts.export_model --weights models/yolov8n.pt --format onnx --half

    # TensorRT FP16 (NVIDIA GPU only)
    python -m scripts.export_model --weights models/yolov8n.pt --format engine --half

    # TensorRT INT8 (needs the training data.yaml as a calibration set)
    python -m scripts.export_model \
        --weights custom_models/final/custom_final_1/weights/best.pt \
        --format engine --int8 \
        --data data/yolo_final/<project>/data.yaml

Then point detector.model at the exported file in your config:

    detector:
      model: "models/yolov8n.onnx"     # or models/yolov8n.engine

Notes
-----
- The exported file is written next to the source weights (Ultralytics default).
  Use --output-dir to move it elsewhere (e.g. models/).
- A .engine is tied to the exact GPU + TensorRT version it was built on. Never
  commit it or reuse it across machines — re-export instead.
- INT8 requires a calibration dataset (--data); --half and --int8 are mutually
  exclusive (TensorRT picks one precision mode).
"""

import argparse
import shutil
import sys
from pathlib import Path

# Allow running as `python -m scripts.export_model` from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a YOLO .pt model to ONNX or TensorRT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/yolov8n.pt",
        help="Path to the source .pt weights.",
    )
    parser.add_argument(
        "--format",
        choices=["onnx", "engine"],
        required=True,
        help="Export format: 'onnx' (portable) or 'engine' (TensorRT, NVIDIA only).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size baked into the exported model.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="FP16 precision. Faster, half the memory, negligible accuracy loss.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="INT8 quantization (engine only). Requires --data for calibration.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="data.yaml used as the INT8 calibration dataset (required for --int8).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Dynamic input shapes (ONNX). Incompatible with --int8 / fixed engines.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Max batch size baked into the exported model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for export ('0' = first GPU, 'cpu'). Engine build needs a GPU.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If set, move the exported file into this directory.",
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")
    if weights.suffix != ".pt":
        sys.exit(f"Source weights must be a .pt file, got: {weights.suffix}")

    if args.int8:
        if args.format != "engine":
            sys.exit("--int8 is only supported with --format engine.")
        if not args.data:
            sys.exit("--int8 requires --data <data.yaml> for calibration.")
        if not Path(args.data).exists():
            sys.exit(f"Calibration data.yaml not found: {args.data}")
    if args.half and args.int8:
        sys.exit("--half and --int8 are mutually exclusive (pick one precision mode).")
    if args.dynamic and args.int8:
        sys.exit("--dynamic cannot be combined with --int8.")


def main() -> None:
    args = parse_args()
    validate(args)

    print(
        f"Exporting {args.weights} -> {args.format} "
        f"(imgsz={args.imgsz}, half={args.half}, int8={args.int8}, "
        f"dynamic={args.dynamic}, batch={args.batch}, device={args.device})"
    )

    model = YOLO(args.weights)

    export_kwargs = dict(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        batch=args.batch,
        device=args.device,
    )
    if args.int8:
        export_kwargs["data"] = args.data

    exported_path = Path(model.export(**export_kwargs))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / exported_path.name
        shutil.move(str(exported_path), str(dest))
        exported_path = dest

    print(f"\nExport complete: {exported_path}")
    print("Point detector.model at this file in your config to use it.")


if __name__ == "__main__":
    main()
