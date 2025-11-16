#!/usr/bin/env python3
"""
Steps 3-6: Coverage mask construction, sparse point sampling, 
point-prompt segmentation, and mask merging.

This script completes the hybrid YOLOv12 + TinySAM pipeline.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
import time

# Add TinySAM to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tinysam import sam_model_registry, SamPredictor


def load_coverage_mask(metadata_path, image_shape):
    """Step 3: Create binary coverage mask from YOLO-guided masks."""
    print("[Step 3] Creating coverage mask from YOLO-guided masks...")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Initialize coverage mask
    coverage_mask = np.zeros(image_shape[:2], dtype=bool)
    
    # Union all masks
    for entry in metadata:
        mask_path = entry['mask_path']
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            coverage_mask |= (mask > 127)
    
    print(f"[Step 3] Coverage: {coverage_mask.sum() / coverage_mask.size * 100:.1f}% of image")
    return coverage_mask


def sample_sparse_grid(image_shape, grid_size=16, coverage_mask=None):
    """Step 4: Sample sparse grid points, skipping covered areas."""
    print(f"[Step 4] Sampling {grid_size}x{grid_size} sparse grid...")
    
    H, W = image_shape[:2]
    cell_h, cell_w = H // grid_size, W // grid_size
    
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Center of cell
            y = int((i + 0.5) * cell_h)
            x = int((j + 0.5) * cell_w)
            
            # Skip if covered by YOLO masks
            if coverage_mask is not None and coverage_mask[y, x]:
                continue
            
            points.append([x, y])
    
    points = np.array(points)
    print(f"[Step 4] Sampled {len(points)} points (skipped covered areas)")
    return points


def run_tinysam_point_prompts(predictor, points, image):
    """Step 5: Run TinySAM on sparse point prompts."""
    print(f"[Step 5] Running TinySAM on {len(points)} point prompts...")
    
    predictor.set_image(image)
    
    masks_list = []
    scores_list = []
    
    for pt in points:
        point_coords = np.array([pt])
        point_labels = np.array([1])
        
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels
        )
        
        # Take highest-score mask
        best_idx = scores.argmax()
        masks_list.append(masks[best_idx])
        scores_list.append(scores[best_idx])
    
    print(f"[Step 5] Generated {len(masks_list)} masks from point prompts")
    return masks_list, scores_list


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def merge_masks_nms(yolo_masks, yolo_scores, point_masks, point_scores, iou_threshold=0.7, min_area=100):
    """Step 6: Merge YOLO and point masks with IoU-based NMS."""
    print(f"[Step 6] Merging masks with IoU threshold={iou_threshold}...")
    
    # Combine all masks and scores
    all_masks = yolo_masks + point_masks
    all_scores = yolo_scores + point_scores
    
    # Sort by score (descending)
    sorted_indices = np.argsort(all_scores)[::-1]
    
    kept_masks = []
    kept_scores = []
    
    for idx in sorted_indices:
        mask = all_masks[idx]
        score = all_scores[idx]
        
        # Filter by minimum area
        if mask.sum() < min_area:
            continue
        
        # Check IoU with already kept masks
        keep = True
        for kept_mask in kept_masks:
            if compute_iou(mask, kept_mask) > iou_threshold:
                keep = False
                break
        
        if keep:
            kept_masks.append(mask)
            kept_scores.append(score)
    
    print(f"[Step 6] Kept {len(kept_masks)} masks after NMS (from {len(all_masks)} total)")
    return kept_masks, kept_scores


def save_final_output(image, masks, scores, output_dir):
    """Save final merged masks and composite visualization."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual masks
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_path = os.path.join(output_dir, f'final_mask_{i:04d}.png')
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    
    # Create composite visualization
    composite = image.copy()
    overlay = np.zeros_like(image)
    
    for i, mask in enumerate(masks):
        color = np.random.randint(50, 255, size=3).tolist()
        overlay[mask] = color
    
    composite = cv2.addWeighted(composite, 0.6, overlay, 0.4, 0)
    
    composite_path = os.path.join(output_dir, 'final_composite.png')
    cv2.imwrite(composite_path, composite)
    
    print(f"[OUTPUT] Saved {len(masks)} final masks to {output_dir}")
    print(f"[OUTPUT] Saved composite to {composite_path}")


def main():
    parser = argparse.ArgumentParser(description='Hybrid YOLOv12 + TinySAM Pipeline (Steps 3-6)')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--sam-weights', required=True, help='TinySAM weights path')
    parser.add_argument('--yolo-metadata', required=True, help='Metadata JSON from Step 2')
    parser.add_argument('--grid-size', type=int, default=16, help='Sparse grid size (default: 16x16)')
    parser.add_argument('--iou-threshold', type=float, default=0.7, help='IoU threshold for NMS')
    parser.add_argument('--min-area', type=int, default=100, help='Minimum mask area (pixels)')
    parser.add_argument('--output-dir', default='outputs/hybrid_final', help='Output directory')
    args = parser.parse_args()
    
    # Load image
    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 3: Coverage mask
    coverage_mask = load_coverage_mask(args.yolo_metadata, image.shape)
    
    # Step 4: Sparse point sampling
    sparse_points = sample_sparse_grid(image.shape, args.grid_size, coverage_mask)
    
    # Step 5: TinySAM point prompts
    print(f"[INFO] Loading TinySAM from {args.sam_weights}")
    sam = sam_model_registry['vit_t'](checkpoint=args.sam_weights)
    predictor = SamPredictor(sam)
    
    start_time = time.time()
    point_masks, point_scores = run_tinysam_point_prompts(predictor, sparse_points, image)
    point_time = time.time() - start_time
    
    # Load YOLO-guided masks from Step 2
    with open(args.yolo_metadata, 'r') as f:
        yolo_metadata = json.load(f)
    
    yolo_masks = []
    yolo_scores = []
    for entry in yolo_metadata:
        mask_path = entry['mask_path']
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
            yolo_masks.append(mask)
            yolo_scores.append(entry['score'])
    
    # Step 6: Merge masks with NMS
    final_masks, final_scores = merge_masks_nms(
        yolo_masks, yolo_scores,
        point_masks, point_scores,
        iou_threshold=args.iou_threshold,
        min_area=args.min_area
    )
    
    # Save outputs
    save_final_output(image, final_masks, final_scores, args.output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"YOLO detections:        {len(yolo_masks)}")
    print(f"Sparse points sampled:  {len(sparse_points)}")
    print(f"Point masks generated:  {len(point_masks)}")
    print(f"Total masks (pre-NMS):  {len(yolo_masks) + len(point_masks)}")
    print(f"Final masks (post-NMS): {len(final_masks)}")
    print(f"TinySAM decoder calls:  {len(yolo_masks) + len(point_masks)}")
    print(f"Point prompt time:      {point_time:.2f}s")
    print("="*60)


if __name__ == '__main__':
    main()

