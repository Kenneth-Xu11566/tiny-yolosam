#!/usr/bin/env python3
"""
Run YOLOv12 inference on an image and save the annotated result.
Usage:
  python TinySAM/scripts/run_yolo_demo.py \
      --weights ../weights/yolov12s_turbo.pt \
      --image TinySAM/fig/picture2.jpg \
      --conf 0.25 \
      --iou 0.55 \
      --max-det 200 \
      --save-dir yolov12/runs/detect
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv12 demo inference.")
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to YOLOv12 weights (.pt)."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image to run inference on."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (lower = more boxes). Default 0.25",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold. Default 0.7",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image. Default 300",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional directory to store YOLO outputs (defaults to yolov12/runs/detect).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights).expanduser()
    image_path = Path(args.image).expanduser()

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[INFO] Loading YOLO model from {weights_path}")
    model = YOLO(str(weights_path))

    yolo_kwargs = {
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "save": True,
    }
    if args.save_dir:
        yolo_kwargs["project"] = str(Path(args.save_dir).parent)
        yolo_kwargs["name"] = Path(args.save_dir).name
        yolo_kwargs["exist_ok"] = True

    print(f"[INFO] Running inference on {image_path}")
    results = model(str(image_path), **yolo_kwargs)

    detections = len(results[0].boxes)
    save_dir = Path(results[0].save_dir)
    print(f"[INFO] Detections: {detections}")
    print(f"[INFO] Annotated image saved to: {save_dir}")


if __name__ == "__main__":
    main()

