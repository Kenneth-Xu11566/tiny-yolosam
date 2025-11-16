#!/usr/bin/env python3
"""
Visualize step-by-step pipeline for Hierarchical, YOLO-only and Hybrid approaches.

Hierarchical pipeline:
  Step 1: Input image
  Step 2: Dense point grid (32x32)
  Step 3: TinySAM hierarchical masks

YOLO-only pipeline:
  Step 1: Input image
  Step 2: YOLO bounding boxes
  Step 3: TinySAM segmentation masks

Hybrid pipeline:
  Step 1: Input image
  Step 2: YOLO bounding boxes
  Step 3: Sparse points (on uncovered regions)
  Step 4: TinySAM masks (YOLO + sparse)
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add TinySAM to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tinysam import sam_model_registry, SamPredictor
from tinysam.hierarchical_mask_generator import SamHierarchicalMaskGenerator
from ultralytics import YOLO


def visualize_hierarchical_pipeline(image, sam_weights, output_dir, points_per_side=32):
    """Generate step-by-step visualization for Hierarchical pipeline.
    
    Shows the actual two-stage adaptive process:
    1. Coarse grid (8x8) for initial coverage
    2. High-confidence regions identified
    3. Adaptive refinement points sampled in uncertain areas only
    4. Final hierarchical masks
    """
    print("\n" + "="*70)
    print("HIERARCHICAL PIPELINE VISUALIZATION (Adaptive Two-Stage)")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Input image
    step1_path = os.path.join(output_dir, 'step1_input.png')
    cv2.imwrite(step1_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[Step 1] Saved input image: {step1_path}")
    
    # Step 2: Coarse grid (first pass)
    coarse_size = points_per_side // 4  # 8x8 for 32
    H, W = image.shape[:2]
    cell_h, cell_w = H // coarse_size, W // coarse_size
    coarse_points = []
    for i in range(coarse_size):
        for j in range(coarse_size):
            y = int((i + 0.5) * cell_h)
            x = int((j + 0.5) * cell_w)
            coarse_points.append([x, y])
    coarse_points = np.array(coarse_points)
    
    step2_img = image.copy()
    for pt in coarse_points:
        cv2.circle(step2_img, tuple(pt), 4, (0, 255, 0), -1)  # Green for coarse
    
    step2_path = os.path.join(output_dir, 'step2_coarse_grid.png')
    cv2.imwrite(step2_path, cv2.cvtColor(step2_img, cv2.COLOR_RGB2BGR))
    print(f"[Step 2] Saved coarse grid (first pass: {len(coarse_points)} points, {coarse_size}x{coarse_size}): {step2_path}")
    
    # Step 3: Run hierarchical generation and capture adaptive points
    sam = sam_model_registry['vit_t'](checkpoint=sam_weights)
    mask_generator = SamHierarchicalMaskGenerator(sam, points_per_side=points_per_side)
    
    # Manually replicate the hierarchical logic to visualize intermediate steps
    mask_generator.set_image(image)
    mask_generator.set_points_per_side(points_per_side // 4)
    ori_masks, or_results = mask_generator.generate(image, True)
    
    # Visualize high-confidence coverage
    step3_img = image.copy()
    coverage_overlay = np.zeros_like(image)
    coverage_mask = or_results.cpu().numpy().T  # Transpose back to (H, W)
    coverage_overlay[coverage_mask] = [100, 150, 255]  # Blue for covered
    step3_img = cv2.addWeighted(step3_img, 0.7, coverage_overlay, 0.3, 0)
    
    # Draw coarse points
    for pt in coarse_points:
        cv2.circle(step3_img, tuple(pt), 4, (0, 255, 0), -1)
    
    step3_path = os.path.join(output_dir, 'step3_high_confidence_coverage.png')
    cv2.imwrite(step3_path, cv2.cvtColor(step3_img, cv2.COLOR_RGB2BGR))
    print(f"[Step 3] Saved high-confidence coverage (blue = well-segmented): {step3_path}")
    
    # Step 4: Adaptive refinement points (only in uncertain regions)
    ih, iw, _ = image.shape
    hstride = ih // points_per_side
    wstride = iw // points_per_side
    new_points = []
    full_point_grids = np.array(mask_generator.point_grids)
    
    for mask in range(full_point_grids.shape[1]):
        point_coords = [full_point_grids[0, mask, 0] * iw, full_point_grids[0, mask, 1] * ih]
        for sy in [-1, 0, 1]:
            for sx in [-1, 0, 1]:
                px, py = int(point_coords[0] + wstride * sy), int(point_coords[1] + hstride * sx)
                if (sy == 0 and sx == 0) or px < 0 or py < 0 or px >= iw or py >= ih:
                    continue
                if or_results[px, py]:  # Skip if already covered
                    continue
                new_points.append([px, py])
    
    step4_img = step3_img.copy()
    for pt in new_points:
        cv2.circle(step4_img, tuple(pt), 3, (255, 0, 0), -1)  # Red for refinement
    
    step4_path = os.path.join(output_dir, 'step4_adaptive_refinement.png')
    cv2.imwrite(step4_path, cv2.cvtColor(step4_img, cv2.COLOR_RGB2BGR))
    print(f"[Step 4] Saved adaptive refinement ({len(new_points)} points in uncertain regions only): {step4_path}")
    
    # Step 5: Final hierarchical masks
    masks = mask_generator.hierarchical_generate(image)
    
    step5_img = image.copy()
    overlay = np.zeros_like(image)
    for mask_dict in masks:
        mask = mask_dict['segmentation']
        color = np.random.randint(50, 255, size=3).tolist()
        overlay[mask] = color
    step5_img = cv2.addWeighted(step5_img, 0.6, overlay, 0.4, 0)
    
    step5_path = os.path.join(output_dir, 'step5_final_masks.png')
    cv2.imwrite(step5_path, cv2.cvtColor(step5_img, cv2.COLOR_RGB2BGR))
    print(f"[Step 5] Saved final hierarchical masks ({len(masks)} masks): {step5_path}")
    
    # Create combined visualization
    create_pipeline_comparison(
        [step1_path, step2_path, step3_path, step4_path, step5_path],
        [
            'Step 1: Input',
            f'Step 2: Coarse Grid ({coarse_size}x{coarse_size})',
            'Step 3: High-Conf Coverage',
            f'Step 4: Adaptive Refine ({len(new_points)})',
            f'Step 5: Final Masks ({len(masks)})'
        ],
        os.path.join(output_dir, 'pipeline_hierarchical.png')
    )
    
    return len(coarse_points) + len(new_points), len(masks)


def visualize_yolo_only_pipeline(image, yolo_weights, sam_weights, conf, iou, max_det, output_dir):
    """Generate step-by-step visualization for YOLO-only pipeline."""
    print("\n" + "="*70)
    print("YOLO-ONLY PIPELINE VISUALIZATION")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Input image
    step1_path = os.path.join(output_dir, 'step1_input.png')
    cv2.imwrite(step1_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[Step 1] Saved input image: {step1_path}")
    
    # Step 2: YOLO bounding boxes
    yolo_model = YOLO(yolo_weights)
    yolo_results = yolo_model(image, conf=conf, iou=iou, max_det=max_det, verbose=False)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    
    step2_img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(step2_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    step2_path = os.path.join(output_dir, 'step2_yolo_boxes.png')
    cv2.imwrite(step2_path, cv2.cvtColor(step2_img, cv2.COLOR_RGB2BGR))
    print(f"[Step 2] Saved YOLO boxes ({len(boxes)} detections): {step2_path}")
    
    # Step 3: TinySAM segmentation masks
    sam = sam_model_registry['vit_t'](checkpoint=sam_weights)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    all_masks = []
    for box in boxes:
        masks, scores, _ = predictor.predict(box=box[None, :])
        best_idx = np.argmax(scores)
        all_masks.append(masks[best_idx])
    
    step3_img = image.copy()
    overlay = np.zeros_like(image)
    for mask in all_masks:
        color = np.random.randint(50, 255, size=3).tolist()
        overlay[mask] = color
    step3_img = cv2.addWeighted(step3_img, 0.6, overlay, 0.4, 0)
    
    step3_path = os.path.join(output_dir, 'step3_tinysam_masks.png')
    cv2.imwrite(step3_path, cv2.cvtColor(step3_img, cv2.COLOR_RGB2BGR))
    print(f"[Step 3] Saved TinySAM masks ({len(all_masks)} masks): {step3_path}")
    
    # Create combined visualization
    create_pipeline_comparison(
        [step1_path, step2_path, step3_path],
        ['Step 1: Input', f'Step 2: YOLO Boxes ({len(boxes)})', f'Step 3: TinySAM Masks ({len(all_masks)})'],
        os.path.join(output_dir, 'pipeline_yolo_only.png')
    )
    
    return len(boxes), len(all_masks)


def visualize_hybrid_pipeline(image, yolo_weights, sam_weights, conf, iou, max_det, grid_size, output_dir):
    """Generate step-by-step visualization for Hybrid pipeline."""
    print("\n" + "="*70)
    print("HYBRID PIPELINE VISUALIZATION")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Input image
    step1_path = os.path.join(output_dir, 'step1_input.png')
    cv2.imwrite(step1_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"[Step 1] Saved input image: {step1_path}")
    
    # Step 2: YOLO bounding boxes
    yolo_model = YOLO(yolo_weights)
    yolo_results = yolo_model(image, conf=conf, iou=iou, max_det=max_det, verbose=False)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    
    step2_img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(step2_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    step2_path = os.path.join(output_dir, 'step2_yolo_boxes.png')
    cv2.imwrite(step2_path, cv2.cvtColor(step2_img, cv2.COLOR_RGB2BGR))
    print(f"[Step 2] Saved YOLO boxes ({len(boxes)} detections): {step2_path}")
    
    # Get YOLO masks and build coverage
    sam = sam_model_registry['vit_t'](checkpoint=sam_weights)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    yolo_masks = []
    coverage_mask = np.zeros(image.shape[:2], dtype=bool)
    for box in boxes:
        masks, scores, _ = predictor.predict(box=box[None, :])
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        yolo_masks.append(mask)
        coverage_mask |= mask
    
    # Step 3: Sparse points on uncovered regions
    H, W = image.shape[:2]
    cell_h, cell_w = H // grid_size, W // grid_size
    sparse_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            y = int((i + 0.5) * cell_h)
            x = int((j + 0.5) * cell_w)
            if not coverage_mask[y, x]:
                sparse_points.append([x, y])
    sparse_points = np.array(sparse_points)
    
    step3_img = step2_img.copy()
    # Draw YOLO coverage in semi-transparent blue
    coverage_overlay = np.zeros_like(image)
    coverage_overlay[coverage_mask] = [100, 100, 255]
    step3_img = cv2.addWeighted(step3_img, 0.8, coverage_overlay, 0.2, 0)
    
    # Draw sparse points in red
    for pt in sparse_points:
        cv2.circle(step3_img, tuple(pt), 3, (255, 0, 0), -1)
    
    step3_path = os.path.join(output_dir, 'step3_sparse_points.png')
    cv2.imwrite(step3_path, cv2.cvtColor(step3_img, cv2.COLOR_RGB2BGR))
    print(f"[Step 3] Saved sparse points ({len(sparse_points)} points on uncovered regions): {step3_path}")
    
    # Step 4: TinySAM masks (YOLO + sparse)
    point_masks = []
    for pt in sparse_points:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([pt]),
            point_labels=np.array([1])
        )
        best_idx = np.argmax(scores)
        point_masks.append(masks[best_idx])
    
    all_masks = yolo_masks + point_masks
    
    step4_img = image.copy()
    overlay = np.zeros_like(image)
    for mask in all_masks:
        color = np.random.randint(50, 255, size=3).tolist()
        overlay[mask] = color
    step4_img = cv2.addWeighted(step4_img, 0.6, overlay, 0.4, 0)
    
    step4_path = os.path.join(output_dir, 'step4_tinysam_masks.png')
    cv2.imwrite(step4_path, cv2.cvtColor(step4_img, cv2.COLOR_RGB2BGR))
    print(f"[Step 4] Saved TinySAM masks ({len(yolo_masks)} YOLO + {len(point_masks)} sparse = {len(all_masks)} total): {step4_path}")
    
    # Create combined visualization
    create_pipeline_comparison(
        [step1_path, step2_path, step3_path, step4_path],
        [
            'Step 1: Input',
            f'Step 2: YOLO Boxes ({len(boxes)})',
            f'Step 3: Sparse Points ({len(sparse_points)})',
            f'Step 4: TinySAM Masks ({len(all_masks)})'
        ],
        os.path.join(output_dir, 'pipeline_hybrid.png')
    )
    
    return len(boxes), len(sparse_points), len(all_masks)


def create_pipeline_comparison(image_paths, labels, output_path):
    """Create a horizontal comparison of pipeline steps."""
    images = [cv2.imread(path) for path in image_paths]
    
    # Resize to consistent height
    target_height = 400
    def resize_keep_aspect(img, target_h):
        h, w = img.shape[:2]
        ratio = target_h / h
        return cv2.resize(img, (int(w * ratio), target_h))
    
    images = [resize_keep_aspect(img, target_height) for img in images]
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    def add_label(img, text):
        img_copy = img.copy()
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = 35
        # Background rectangle
        cv2.rectangle(img_copy, (text_x - 10, text_y - text_size[1] - 10), 
                     (text_x + text_size[0] + 10, text_y + 10), bg_color, -1)
        cv2.putText(img_copy, text, (text_x, text_y), font, font_scale, color, thickness)
        return img_copy
    
    labeled_images = [add_label(img, label) for img, label in zip(images, labels)]
    
    # Stack horizontally
    comparison = np.hstack(labeled_images)
    
    # Save
    cv2.imwrite(output_path, comparison)
    print(f"\n[COMBINED] Saved pipeline visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize pipeline steps for Hierarchical, YOLO-only and Hybrid approaches')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--sam-weights', required=True, help='TinySAM weights')
    parser.add_argument('--yolo-weights', required=True, help='YOLOv12 weights')
    parser.add_argument('--conf', type=float, default=0.1, help='YOLO confidence threshold')
    parser.add_argument('--iou', type=float, default=0.55, help='YOLO NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=350, help='YOLO max detections')
    parser.add_argument('--grid-size', type=int, default=16, help='Sparse grid size for hybrid')
    parser.add_argument('--points-per-side', type=int, default=32, help='Points per side for hierarchical')
    parser.add_argument('--output-dir', default='outputs/pipeline_viz', help='Output directory')
    parser.add_argument('--mode', choices=['hierarchical', 'yolo', 'hybrid', 'all'], default='all', 
                       help='Which pipeline to visualize')
    args = parser.parse_args()
    
    # Load image
    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run visualizations
    if args.mode in ['hierarchical', 'all']:
        hier_points, hier_masks = visualize_hierarchical_pipeline(
            image, args.sam_weights,
            os.path.join(args.output_dir, 'hierarchical'),
            args.points_per_side
        )
    
    if args.mode in ['yolo', 'all']:
        yolo_boxes, yolo_masks = visualize_yolo_only_pipeline(
            image, args.yolo_weights, args.sam_weights,
            args.conf, args.iou, args.max_det,
            os.path.join(args.output_dir, 'yolo_only')
        )
    
    if args.mode in ['hybrid', 'all']:
        hybrid_boxes, hybrid_points, hybrid_masks = visualize_hybrid_pipeline(
            image, args.yolo_weights, args.sam_weights,
            args.conf, args.iou, args.max_det, args.grid_size,
            os.path.join(args.output_dir, 'hybrid')
        )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if args.mode in ['hierarchical', 'all']:
        print(f"Hierarchical: {hier_points} dense grid points → {hier_masks} TinySAM masks")
    if args.mode in ['yolo', 'all']:
        print(f"YOLO-only:    {yolo_boxes} YOLO boxes → {yolo_masks} TinySAM masks")
    if args.mode in ['hybrid', 'all']:
        print(f"Hybrid:       {hybrid_boxes} YOLO boxes + {hybrid_points} sparse points → {hybrid_masks} TinySAM masks")
    print("="*70)
    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()

