#!/usr/bin/env python3
"""
Benchmark all three systems:
1. TinySAM hierarchical everything (baseline)
2. YOLO-only → TinySAM
3. Hybrid (YOLO + sparse points)

Measures: decoder calls, latency, masks generated
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import time
from pathlib import Path

# Add TinySAM to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tinysam import sam_model_registry, SamPredictor
from tinysam.hierarchical_mask_generator import SamHierarchicalMaskGenerator
from tinysam.batch_utils import batch_point_prompts, batch_box_prompts
from ultralytics import YOLO


class DecoderCallCounter:
    """Wrapper to count TinySAM decoder calls."""
    def __init__(self, predictor):
        self.predictor = predictor
        self.call_count = 0
        self._original_predict = predictor.predict
        self._original_predict_torch = predictor.predict_torch
        predictor.predict = self._wrapped_predict
        predictor.predict_torch = self._wrapped_predict_torch
    
    def _wrapped_predict(self, *args, **kwargs):
        self.call_count += 1
        return self._original_predict(*args, **kwargs)
    
    def _wrapped_predict_torch(self, *args, **kwargs):
        # Count batched predictions (each batch is one decoder call)
        self.call_count += 1
        return self._original_predict_torch(*args, **kwargs)
    
    def reset(self):
        self.call_count = 0


def benchmark_system1_hierarchical(image, sam_weights, output_dir):
    """System 1: TinySAM hierarchical everything."""
    print("\n" + "="*60)
    print("SYSTEM 1: TinySAM Hierarchical Everything (Baseline)")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    sam = sam_model_registry['vit_t'](checkpoint=sam_weights)
    mask_generator = SamHierarchicalMaskGenerator(sam)
    
    # Count decoder calls by wrapping predictor
    counter = DecoderCallCounter(mask_generator.predictor)
    
    # Run hierarchical generation
    start_time = time.time()
    masks = mask_generator.hierarchical_generate(image)
    latency = time.time() - start_time
    
    # Save results
    result = {
        'system': 'hierarchical_everything',
        'num_masks': len(masks),
        'decoder_calls': counter.call_count,
        'latency_sec': latency,
        'masks_per_call': len(masks) / counter.call_count if counter.call_count > 0 else 0
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save composite visualization
    composite = image.copy()
    overlay = np.zeros_like(image)
    for i, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']
        color = np.random.randint(50, 255, size=3).tolist()
        overlay[mask] = color
    composite = cv2.addWeighted(composite, 0.6, overlay, 0.4, 0)
    cv2.imwrite(os.path.join(output_dir, 'composite.png'), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    
    print(f"Masks generated:     {result['num_masks']}")
    print(f"Decoder calls:       {result['decoder_calls']}")
    print(f"Latency:             {result['latency_sec']:.2f}s")
    print(f"Masks per call:      {result['masks_per_call']:.2f}")
    
    return result


def benchmark_system2_yolo_only(image, sam_weights, yolo_weights, conf, iou, max_det, output_dir):
    """System 2: YOLO-only → TinySAM."""
    print("\n" + "="*60)
    print("SYSTEM 2: YOLO-only → TinySAM")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run YOLO
    yolo_model = YOLO(yolo_weights)
    yolo_results = yolo_model(image, conf=conf, iou=iou, max_det=max_det, verbose=False)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    
    print(f"YOLO detections:     {len(boxes)}")
    
    # Load TinySAM
    sam = sam_model_registry['vit_t'](checkpoint=sam_weights)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    # Use batch processing for boxes
    start_time = time.time()
    masks_list, scores_list = batch_box_prompts(
        predictor=predictor,
        boxes=boxes,
        batch_size=32,
        verbose=True
    )
    latency = time.time() - start_time
    decoder_calls = len(boxes)  # Still counts logical decoder calls
    
    # Save results
    result = {
        'system': 'yolo_only',
        'yolo_detections': len(boxes),
        'num_masks': len(masks_list),
        'decoder_calls': decoder_calls,
        'latency_sec': latency
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save composite
    composite = image.copy()
    overlay = np.zeros_like(image)
    for mask in masks_list:
        color = np.random.randint(50, 255, size=3).tolist()
        overlay[mask] = color
    composite = cv2.addWeighted(composite, 0.6, overlay, 0.4, 0)
    cv2.imwrite(os.path.join(output_dir, 'composite.png'), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    
    print(f"Masks generated:     {result['num_masks']}")
    print(f"Decoder calls:       {result['decoder_calls']}")
    print(f"Latency:             {result['latency_sec']:.2f}s")
    
    return result


def benchmark_system3_hybrid(image, sam_weights, yolo_weights, conf, iou, max_det, grid_size, output_dir):
    """System 3: Hybrid (YOLO + sparse points)."""
    print("\n" + "="*60)
    print("SYSTEM 3: Hybrid (YOLO + Sparse Points)")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run YOLO
    yolo_model = YOLO(yolo_weights)
    yolo_results = yolo_model(image, conf=conf, iou=iou, max_det=max_det, verbose=False)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    
    print(f"YOLO detections:     {len(boxes)}")
    
    # Load TinySAM
    sam = sam_model_registry['vit_t'](checkpoint=sam_weights)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    # Get YOLO masks and build coverage using batch processing
    yolo_start = time.time()
    yolo_masks, yolo_scores = batch_box_prompts(
        predictor=predictor,
        boxes=boxes,
        batch_size=32,
        verbose=True
    )
    
    # Build coverage mask
    coverage_mask = np.zeros(image.shape[:2], dtype=bool)
    for mask in yolo_masks:
        coverage_mask |= mask
    yolo_time = time.time() - yolo_start
    
    # Sparse point sampling
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
    
    print(f"Sparse points:       {len(sparse_points)}")
    
    # Point prompts using batch processing
    point_start = time.time()
    point_masks, point_scores = batch_point_prompts(
        predictor=predictor,
        points=sparse_points,
        batch_size=64,
        multimask_output=True,
        return_best_only=True,
        min_confidence=0.7,
        min_area=100,
        verbose=True
    )
    point_time = time.time() - point_start
    
    total_time = yolo_time + point_time
    decoder_calls = len(boxes) + len(sparse_points)
    total_masks = len(yolo_masks) + len(point_masks)
    
    # Save results
    result = {
        'system': 'hybrid',
        'yolo_detections': len(boxes),
        'sparse_points': len(sparse_points),
        'num_masks': total_masks,
        'decoder_calls': decoder_calls,
        'latency_sec': total_time,
        'yolo_time': yolo_time,
        'point_time': point_time
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save composite
    composite = image.copy()
    overlay = np.zeros_like(image)
    for mask in yolo_masks + point_masks:
        color = np.random.randint(50, 255, size=3).tolist()
        overlay[mask] = color
    composite = cv2.addWeighted(composite, 0.6, overlay, 0.4, 0)
    cv2.imwrite(os.path.join(output_dir, 'composite.png'), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    
    print(f"Masks generated:     {result['num_masks']}")
    print(f"Decoder calls:       {result['decoder_calls']}")
    print(f"Latency:             {result['latency_sec']:.2f}s")
    print(f"  YOLO time:         {result['yolo_time']:.2f}s")
    print(f"  Point time:        {result['point_time']:.2f}s")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Benchmark all three systems')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--sam-weights', required=True, help='TinySAM weights')
    parser.add_argument('--yolo-weights', required=True, help='YOLOv12 weights')
    parser.add_argument('--conf', type=float, default=0.1, help='YOLO confidence threshold')
    parser.add_argument('--iou', type=float, default=0.55, help='YOLO NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=350, help='YOLO max detections')
    parser.add_argument('--grid-size', type=int, default=16, help='Sparse grid size for hybrid')
    parser.add_argument('--output-dir', default='outputs/benchmark', help='Output directory')
    args = parser.parse_args()
    
    # Load image
    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run all systems
    results = {}
    
    results['system1'] = benchmark_system1_hierarchical(
        image, args.sam_weights,
        os.path.join(args.output_dir, 'system1_hierarchical')
    )
    
    results['system2'] = benchmark_system2_yolo_only(
        image, args.sam_weights, args.yolo_weights,
        args.conf, args.iou, args.max_det,
        os.path.join(args.output_dir, 'system2_yolo_only')
    )
    
    results['system3'] = benchmark_system3_hybrid(
        image, args.sam_weights, args.yolo_weights,
        args.conf, args.iou, args.max_det, args.grid_size,
        os.path.join(args.output_dir, 'system3_hybrid')
    )
    
    # Summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'System':<30} {'Decoder Calls':<15} {'Latency (s)':<15} {'Masks':<10}")
    print("-"*60)
    print(f"{'1. Hierarchical (Baseline)':<30} {results['system1']['decoder_calls']:<15} {results['system1']['latency_sec']:<15.2f} {results['system1']['num_masks']:<10}")
    print(f"{'2. YOLO-only':<30} {results['system2']['decoder_calls']:<15} {results['system2']['latency_sec']:<15.2f} {results['system2']['num_masks']:<10}")
    print(f"{'3. Hybrid (Proposed)':<30} {results['system3']['decoder_calls']:<15} {results['system3']['latency_sec']:<15.2f} {results['system3']['num_masks']:<10}")
    print("-"*60)
    
    # Speedup
    speedup_calls = results['system1']['decoder_calls'] / results['system3']['decoder_calls']
    speedup_time = results['system1']['latency_sec'] / results['system3']['latency_sec']
    print(f"\nHybrid vs. Baseline:")
    print(f"  Decoder call reduction: {speedup_calls:.2f}x")
    print(f"  Speedup (latency):      {speedup_time:.2f}x")
    print("="*60)
    
    # Save summary
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visual comparison
    create_visual_comparison(args.output_dir)


def create_visual_comparison(output_dir):
    """Create side-by-side comparison of all three systems."""
    print("\n[INFO] Creating visual comparison...")
    
    # Load composites
    sys1_path = os.path.join(output_dir, 'system1_hierarchical', 'composite.png')
    sys2_path = os.path.join(output_dir, 'system2_yolo_only', 'composite.png')
    sys3_path = os.path.join(output_dir, 'system3_hybrid', 'composite.png')
    
    sys1 = cv2.imread(sys1_path)
    sys2 = cv2.imread(sys2_path)
    sys3 = cv2.imread(sys3_path)
    
    # Resize to consistent height
    target_height = 400
    def resize_keep_aspect(img, target_h):
        h, w = img.shape[:2]
        ratio = target_h / h
        return cv2.resize(img, (int(w * ratio), target_h))
    
    sys1 = resize_keep_aspect(sys1, target_height)
    sys2 = resize_keep_aspect(sys2, target_height)
    sys3 = resize_keep_aspect(sys3, target_height)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    def add_label(img, text):
        img_copy = img.copy()
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = 40
        # Background rectangle
        cv2.rectangle(img_copy, (text_x - 10, text_y - text_size[1] - 10), 
                     (text_x + text_size[0] + 10, text_y + 10), bg_color, -1)
        cv2.putText(img_copy, text, (text_x, text_y), font, font_scale, color, thickness)
        return img_copy
    
    sys1 = add_label(sys1, "System 1: Hierarchical")
    sys2 = add_label(sys2, "System 2: YOLO-only")
    sys3 = add_label(sys3, "System 3: Hybrid")
    
    # Stack horizontally
    comparison = np.hstack([sys1, sys2, sys3])
    
    # Save
    comparison_path = os.path.join(output_dir, 'visual_comparison.png')
    cv2.imwrite(comparison_path, comparison)
    print(f"[INFO] Saved visual comparison to {comparison_path}")


if __name__ == '__main__':
    main()

