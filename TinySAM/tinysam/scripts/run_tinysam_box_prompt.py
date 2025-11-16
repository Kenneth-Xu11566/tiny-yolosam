#!/usr/bin/env python3
"""
Step 2: TinySAM box-prompt segmentation driven by YOLO detections.

Example:
    python TinySAM/scripts/run_tinysam_box_prompt.py \
        --image TinySAM/fig/picture2.jpg \
        --yolo-weights weights/yolov12s_turbo.pt \
        --sam-weights TinySAM/weights/tinysam_42.3.pth \
        --conf 0.05 --iou 0.55 --max-det 350 \
        --output-dir outputs/yolo_box_masks
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from tinysam import SamPredictor, sam_model_registry


def parse_args():
    parser = argparse.ArgumentParser(description="TinySAM box prompting guided by YOLO boxes.")
    parser.add_argument("--image", type=str, required=True, help="Path to source image.")
    parser.add_argument("--yolo-weights", type=str, required=True, help="Path to YOLO weights (.pt).")
    parser.add_argument("--sam-weights", type=str, required=True, help="Path to TinySAM weights (.pth).")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="YOLO max detections.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for TinySAM (cpu/cuda).")
    parser.add_argument("--output-dir", type=str, default="outputs/yolo_box_masks", help="Directory to save masks/metadata.")
    parser.add_argument("--save-composite", action="store_true", help="Save image overlay showing masks.")
    return parser.parse_args()


def load_tinysam(weights_path: Path, device: str):
    print(f"[INFO] Loading TinySAM weights from {weights_path}")
    sam = sam_model_registry["vit_t"](checkpoint=str(weights_path))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def run_yolo(image_path: Path, weights_path: Path, conf: float, iou: float, max_det: int):
    print(f"[INFO] Running YOLOv12: {weights_path}")
    model = YOLO(str(weights_path))
    results = model(
        str(image_path),
        conf=conf,
        iou=iou,
        max_det=max_det,
        save=True,
        verbose=False,
    )
    return results[0]


def main():
    args = parse_args()
    image_path = Path(args.image).expanduser()
    yolo_weights = Path(args.yolo_weights).expanduser()
    sam_weights = Path(args.sam_weights).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not yolo_weights.exists():
        raise FileNotFoundError(yolo_weights)
    if not sam_weights.exists():
        raise FileNotFoundError(sam_weights)

    yolo_result = run_yolo(image_path, yolo_weights, args.conf, args.iou, args.max_det)
    num_boxes = len(yolo_result.boxes)
    print(f"[INFO] YOLO detections: {num_boxes}")

    predictor = load_tinysam(sam_weights, args.device)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    composite = image_bgr.copy()
    metadata = []
    for idx, box in enumerate(yolo_result.boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        score = float(box.conf[0].cpu().item()) if box.conf is not None else None
        cls_id = int(box.cls[0].cpu().item()) if box.cls is not None else None

        masks, scores, _ = predictor.predict(
            box=xyxy[None, :],
            point_coords=None,
            point_labels=None,
        )
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8)

        mask_path = out_dir / f"mask_{idx:04d}.png"
        cv2.imwrite(str(mask_path), mask * 255)

        if args.save_composite:
            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
            colored_mask = np.zeros_like(composite)
            colored_mask[mask > 0] = color
            composite = cv2.addWeighted(composite, 1.0, colored_mask, 0.4, 0)
            cv2.rectangle(
                composite,
                (int(xyxy[0]), int(xyxy[1])),
                (int(xyxy[2]), int(xyxy[3])),
                color.tolist(),
                2,
            )

        metadata.append(
            {
                "box_index": idx,
                "bbox_xyxy": xyxy.tolist(),
                "score": score,
                "class_id": cls_id,
                "mask_path": str(mask_path),
            }
        )

    if args.save_composite:
        composite_path = out_dir / "composite.png"
        cv2.imwrite(str(composite_path), composite)
        print(f"[INFO] Saved composite visualization to {composite_path}")

    meta_path = out_dir / "metadata.json"
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved mask metadata to {meta_path}")
    print(f"[INFO] Total TinySAM decoder calls: {len(metadata)}")


if __name__ == "__main__":
    main()

