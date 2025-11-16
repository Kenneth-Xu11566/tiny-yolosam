#!/usr/bin/env python3
"""
COCO Evaluation for all systems:
1. Hierarchical baseline (class-agnostic) - computes AR, mIoU, coverage
2. ViTDet → TinySAM (detector baseline with categories) - computes AP
3. YOLO → TinySAM (YOLO-only with categories) - computes AP  
4. Hybrid (YOLO + sparse points with categories) - computes AP + AR, mIoU

Detector-based systems (2-4): AP, APs/m/l
Class-agnostic comparison (1 vs 4): AR, mIoU, coverage, latency, decoder calls
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import time
from pathlib import Path
from tqdm import tqdm
import random

# Import pycocotools BEFORE torch to avoid NumPy conflicts
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

# Now import torch-related modules
import torch

# Add TinySAM to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tinysam import sam_model_registry, SamPredictor
from tinysam.hierarchical_mask_generator import SamHierarchicalMaskGenerator
from ultralytics import YOLO

# YOLO class index to COCO category ID mapping
YOLO_TO_COCO_CATEGORY = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]


def compute_class_agnostic_metrics(pred_masks, gt_anns, img_id, coco_gt, debug=False):
    """
    Compute class-agnostic metrics: AR (Average Recall) and mean IoU.
    Measures how well predicted masks overlap with ground truth, ignoring categories.
    """
    # Get ground truth masks for this image
    ann_ids = coco_gt.getAnnIds(imgIds=img_id)
    if len(ann_ids) == 0:
        return {'ar': 0.0, 'mean_iou': 0.0, 'coverage': 0.0}
    
    anns = coco_gt.loadAnns(ann_ids)
    gt_masks = [coco_gt.annToMask(ann) for ann in anns]
    
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return {'ar': 0.0, 'mean_iou': 0.0, 'coverage': 0.0}
    
    # For each GT mask, find best matching predicted mask
    gt_matched = 0
    total_iou = 0.0
    iou_details = []
    
    for i, gt_mask in enumerate(gt_masks):
        gt_area = gt_mask.sum()
        if gt_area == 0:
            continue
            
        best_iou = 0.0
        best_pred_idx = -1
        for j, pred_mask in enumerate(pred_masks):
            intersection = (gt_mask & pred_mask).sum()
            union = (gt_mask | pred_mask).sum()
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j
        
        total_iou += best_iou
        if best_iou > 0.5:  # IoU threshold for "matched"
            gt_matched += 1
        
        if debug and i < 5:  # Show first 5 GT objects
            iou_details.append({
                'gt_idx': i,
                'gt_area': int(gt_area),
                'best_iou': best_iou,
                'matched': best_iou > 0.5,
                'best_pred_idx': best_pred_idx
            })
    
    ar = gt_matched / len(gt_masks) if len(gt_masks) > 0 else 0.0
    mean_iou = total_iou / len(gt_masks) if len(gt_masks) > 0 else 0.0
    coverage = gt_matched / len(gt_masks) if len(gt_masks) > 0 else 0.0
    
    result = {'ar': ar, 'mean_iou': mean_iou, 'coverage': coverage}
    if debug:
        result['iou_details'] = iou_details
    
    return result


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
        self.call_count += 1
        return self._original_predict_torch(*args, **kwargs)
    
    def reset(self):
        self.call_count = 0


def load_coco_subset(coco_gt, val_img_path, num_images=100, random_seed=42):
    """Load a random subset of COCO images."""
    img_ids = list(coco_gt.imgs.keys())
    
    if num_images > 0 and num_images < len(img_ids):
        random.seed(random_seed)
        img_ids = random.sample(img_ids, num_images)
    
    images_info = []
    for img_id in img_ids:
        img_info = coco_gt.imgs[img_id]
        img_path = os.path.join(val_img_path, img_info['file_name'])
        if os.path.exists(img_path):
            images_info.append({
                'id': img_id,
                'path': img_path,
                'width': img_info['width'],
                'height': img_info['height']
            })
    
    return images_info


def eval_hierarchical_baseline(images_info, coco_gt, sam_weights, output_dir):
    """Hierarchical baseline: class-agnostic segment everything."""
    print("\n" + "="*70)
    print("HIERARCHICAL BASELINE: TinySAM Segment Everything")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, 'debug_viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load model
    sam = sam_model_registry['vit_t'](checkpoint=sam_weights)
    mask_generator = SamHierarchicalMaskGenerator(sam)
    counter = DecoderCallCounter(mask_generator.predictor)
    
    total_decoder_calls = 0
    total_time = 0.0
    all_ar = []
    all_mean_iou = []
    all_coverage = []
    all_num_masks = []
    
    for idx, img_info in enumerate(tqdm(images_info, desc="Hierarchical")):
        image = cv2.imread(img_info['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        counter.reset()
        start_time = time.time()
        mask_dicts = mask_generator.hierarchical_generate(image)
        elapsed = time.time() - start_time
        
        total_decoder_calls += counter.call_count
        total_time += elapsed
        
        # Extract binary masks
        pred_masks = [m['segmentation'] for m in mask_dicts]
        all_num_masks.append(len(pred_masks))
        
        # Get GT info for debugging
        ann_ids = coco_gt.getAnnIds(imgIds=img_info['id'])
        num_gt_objects = len(ann_ids)
        
        # Compute class-agnostic metrics
        debug_mode = idx < 2
        metrics = compute_class_agnostic_metrics(pred_masks, None, img_info['id'], coco_gt, debug=debug_mode)
        all_ar.append(metrics['ar'])
        all_mean_iou.append(metrics['mean_iou'])
        all_coverage.append(metrics['coverage'])
        
        # Debug print for first few images
        if idx < 2:
            print(f"\n[DEBUG Image {img_info['id']}]")
            print(f"  Generated masks: {len(pred_masks)}")
            print(f"  GT objects: {num_gt_objects}")
            print(f"  AR: {metrics['ar']:.3f} ({int(metrics['ar'] * num_gt_objects)}/{num_gt_objects} objects matched)")
            print(f"  Mean IoU: {metrics['mean_iou']:.3f}")
            
            if 'iou_details' in metrics:
                print(f"\n  First 5 GT objects:")
                for detail in metrics['iou_details']:
                    status = "✓ MATCHED" if detail['matched'] else "✗ MISSED"
                    print(f"    GT[{detail['gt_idx']}] area={detail['gt_area']:6d} → "
                          f"best_iou={detail['best_iou']:.3f} (pred[{detail['best_pred_idx']}]) {status}")
            
            # Save composite visualization
            composite = image.copy()
            for mask in pred_masks:
                color = np.random.randint(0, 255, 3)
                composite[mask] = composite[mask] * 0.5 + color * 0.5
            cv2.imwrite(os.path.join(viz_dir, f'img_{img_info["id"]}_composite.png'), 
                       cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    summary = {
        'system': 'hierarchical',
        'num_images': len(images_info),
        'avg_masks': float(np.mean(all_num_masks)),
        'total_decoder_calls': total_decoder_calls,
        'avg_decoder_calls': total_decoder_calls / len(images_info),
        'total_time': total_time,
        'avg_time_per_image': total_time / len(images_info),
        'AR': float(np.mean(all_ar)),
        'mean_IoU': float(np.mean(all_mean_iou)),
        'coverage': float(np.mean(all_coverage))
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[Hierarchical] AR: {summary['AR']:.3f}, mIoU: {summary['mean_IoU']:.3f}, "
          f"Time: {summary['avg_time_per_image']:.2f}s, Calls: {summary['avg_decoder_calls']:.1f}")
    
    return summary


def eval_system1_vitdet(images_info, coco_gt, sam_weights, vitdet_json_path, output_dir):
    """System 1: ViTDet boxes → TinySAM (paper baseline)."""
    print("\n" + "="*70)
    print("SYSTEM 1: ViTDet boxes → TinySAM (Paper Baseline)")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ViTDet detections
    with open(vitdet_json_path, 'r') as f:
        vitdet_results = json.load(f)
    
    # Filter for our image subset
    img_ids = set([img['id'] for img in images_info])
    vitdet_results = [det for det in vitdet_results if det['image_id'] in img_ids]
    print(f"Loaded {len(vitdet_results)} ViTDet detections for {len(img_ids)} images")
    
    # Load model
    sam = sam_model_registry['vit_t'](checkpoint=sam_weights)
    predictor = SamPredictor(sam)
    
    results = []
    total_decoder_calls = 0
    total_time = 0.0
    pre_img_id = 0
    
    for det in tqdm(vitdet_results, desc="System 1"):
        img_id = det['image_id']
        
        # Load new image if needed
        if pre_img_id != img_id:
            img_info = next((img for img in images_info if img['id'] == img_id), None)
            if img_info is None:
                continue
            image = cv2.imread(img_info['path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            pre_img_id = img_id
        
        # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        bbox = det['bbox']
        input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        
        start_time = time.time()
        masks, scores, _ = predictor.predict(box=input_box[None, :])
        elapsed = time.time() - start_time
        
        total_decoder_calls += 1
        total_time += elapsed
        
        # Take best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')
        
        results.append({
            'image_id': img_id,
            'category_id': det['category_id'],
            'segmentation': rle,
            'score': float(det['score'])
        })
    
    # Save and evaluate
    results_path = os.path.join(output_dir, 'coco_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    coco_dt = coco_gt.loadRes(results_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.params.imgIds = sorted(img_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    metrics = {
        'system': 'vitdet_baseline',
        'num_images': len(images_info),
        'total_masks': len(results),
        'total_decoder_calls': total_decoder_calls,
        'avg_decoder_calls': total_decoder_calls / len(images_info),
        'total_time': total_time,
        'avg_time_per_image': total_time / len(images_info),
        'AP': float(coco_eval.stats[0]),
        'AP50': float(coco_eval.stats[1]),
        'AP75': float(coco_eval.stats[2]),
        'APs': float(coco_eval.stats[3]),
        'APm': float(coco_eval.stats[4]),
        'APl': float(coco_eval.stats[5])
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def eval_system2_yolo_only(images_info, coco_gt, sam_weights, yolo_weights, conf, iou, max_det, output_dir):
    """System 2: YOLO-only → TinySAM."""
    print("\n" + "="*70)
    print("SYSTEM 2: YOLO-only → TinySAM")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    yolo_model = YOLO(yolo_weights)
    sam = sam_model_registry['vit_t'](checkpoint=sam_weights)
    predictor = SamPredictor(sam)
    
    results = []
    total_decoder_calls = 0
    total_time = 0.0
    
    for img_info in tqdm(images_info, desc="System 2"):
        image = cv2.imread(img_info['path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLO detection
        yolo_results = yolo_model(image, conf=conf, iou=iou, max_det=max_det, verbose=False)
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
        scores = yolo_results[0].boxes.conf.cpu().numpy()
        
        # TinySAM segmentation
        predictor.set_image(image_rgb)
        
        start_time = time.time()
        for box, cls, score in zip(boxes, classes, scores):
            masks, mask_scores, _ = predictor.predict(box=box)
            total_decoder_calls += 1
            
            best_idx = mask_scores.argmax()
            mask = masks[best_idx]
            
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            results.append({
                'image_id': img_info['id'],
                'category_id': YOLO_TO_COCO_CATEGORY[int(cls)],
                'segmentation': rle,
                'score': float(score)
            })
        elapsed = time.time() - start_time
        total_time += elapsed
    
    # Save and evaluate
    results_path = os.path.join(output_dir, 'coco_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    coco_dt = coco_gt.loadRes(results_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.params.imgIds = [img['id'] for img in images_info]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    metrics = {
        'system': 'yolo_only',
        'num_images': len(images_info),
        'total_masks': len(results),
        'total_decoder_calls': total_decoder_calls,
        'avg_decoder_calls': total_decoder_calls / len(images_info),
        'total_time': total_time,
        'avg_time_per_image': total_time / len(images_info),
        'AP': float(coco_eval.stats[0]),
        'AP50': float(coco_eval.stats[1]),
        'AP75': float(coco_eval.stats[2]),
        'APs': float(coco_eval.stats[3]),
        'APm': float(coco_eval.stats[4]),
        'APl': float(coco_eval.stats[5])
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def eval_system3_hybrid(images_info, coco_gt, sam_weights, yolo_weights, conf, iou, max_det, grid_size, output_dir):
    """System 3: Hybrid (YOLO + sparse points).
    
    AP evaluation: Only YOLO box masks (with category IDs)
    Class-agnostic metrics: All masks (YOLO + sparse points)
    """
    print("\n" + "="*70)
    print("SYSTEM 3: Hybrid (YOLO + Sparse Points)")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    yolo_model = YOLO(yolo_weights)
    sam = sam_model_registry['vit_t'](checkpoint=sam_weights)
    predictor = SamPredictor(sam)
    
    results_for_ap = []  # Only YOLO boxes for AP evaluation
    total_decoder_calls = 0
    total_time = 0.0
    all_ar = []
    all_mean_iou = []
    all_coverage = []
    all_num_masks = []
    
    for img_info in tqdm(images_info, desc="System 3"):
        image = cv2.imread(img_info['path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLO detection
        yolo_results = yolo_model(image, conf=conf, iou=iou, max_det=max_det, verbose=False)
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
        scores = yolo_results[0].boxes.conf.cpu().numpy()
        
        predictor.set_image(image_rgb)
        
        start_time = time.time()
        
        # YOLO-guided masks
        coverage_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        all_masks_for_img = []
        
        for box, cls, score in zip(boxes, classes, scores):
            masks, mask_scores, _ = predictor.predict(box=box)
            total_decoder_calls += 1
            
            best_idx = mask_scores.argmax()
            mask = masks[best_idx]
            coverage_mask |= mask
            all_masks_for_img.append(mask)
            
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')
            
            results_for_ap.append({
                'image_id': img_info['id'],
                'category_id': YOLO_TO_COCO_CATEGORY[int(cls)],
                'segmentation': rle,
                'score': float(score)
            })
        
        # Sparse point sampling
        H, W = image_rgb.shape[:2]
        cell_h, cell_w = H // grid_size, W // grid_size
        sparse_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                y = int((i + 0.5) * cell_h)
                x = int((j + 0.5) * cell_w)
                if not coverage_mask[y, x]:
                    sparse_points.append([x, y])
        
        # Point prompts
        
        for pt in sparse_points:
            masks, mask_scores, _ = predictor.predict(
                point_coords=np.array([pt]),
                point_labels=np.array([1])
            )
            total_decoder_calls += 1
            
            best_idx = mask_scores.argmax()
            mask = masks[best_idx]
            all_masks_for_img.append(mask)
            
            # Note: sparse point masks are NOT added to AP evaluation
            # They are class-agnostic and only used for coverage metrics
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Compute class-agnostic metrics for this image
        all_num_masks.append(len(all_masks_for_img))
        
        # Get GT info for debugging
        ann_ids = coco_gt.getAnnIds(imgIds=img_info['id'])
        num_gt_objects = len(ann_ids)
        
        debug_mode = len(all_ar) < 2  # Debug first 2 images
        metrics = compute_class_agnostic_metrics(all_masks_for_img, None, img_info['id'], coco_gt, debug=debug_mode)
        all_ar.append(metrics['ar'])
        all_mean_iou.append(metrics['mean_iou'])
        all_coverage.append(metrics['coverage'])
        
        # Debug print for first few images
        if debug_mode:
            print(f"\n[DEBUG HYBRID Image {img_info['id']}]")
            print(f"  YOLO boxes: {len(boxes)}")
            print(f"  Sparse points: {len(sparse_points)}")
            print(f"  Total masks: {len(all_masks_for_img)} ({len(boxes)} YOLO + {len(sparse_points)} sparse)")
            print(f"  GT objects: {num_gt_objects}")
            print(f"  AR: {metrics['ar']:.3f} ({int(metrics['ar'] * num_gt_objects)}/{num_gt_objects} objects matched)")
            print(f"  Mean IoU: {metrics['mean_iou']:.3f}")
            
            if 'iou_details' in metrics:
                print(f"\n  First 5 GT objects:")
                for detail in metrics['iou_details']:
                    status = "✓ MATCHED" if detail['matched'] else "✗ MISSED"
                    print(f"    GT[{detail['gt_idx']}] area={detail['gt_area']:6d} → "
                          f"best_iou={detail['best_iou']:.3f} (pred[{detail['best_pred_idx']}]) {status}")
    
    # Save and evaluate AP (YOLO boxes only)
    results_path = os.path.join(output_dir, 'coco_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_for_ap, f)
    
    coco_dt = coco_gt.loadRes(results_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.params.imgIds = [img['id'] for img in images_info]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    metrics = {
        'system': 'hybrid',
        'num_images': len(images_info),
        'total_masks_yolo': len(results_for_ap),
        'total_masks_all': int(np.sum(all_num_masks)),
        'avg_masks': float(np.mean(all_num_masks)),
        'total_decoder_calls': total_decoder_calls,
        'avg_decoder_calls': total_decoder_calls / len(images_info),
        'total_time': total_time,
        'avg_time_per_image': total_time / len(images_info),
        'AP': float(coco_eval.stats[0]),
        'AP50': float(coco_eval.stats[1]),
        'AP75': float(coco_eval.stats[2]),
        'APs': float(coco_eval.stats[3]),
        'APm': float(coco_eval.stats[4]),
        'APl': float(coco_eval.stats[5]),
        'AR': float(np.mean(all_ar)),
        'mean_IoU': float(np.mean(all_mean_iou)),
        'coverage': float(np.mean(all_coverage))
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[Hybrid] AP (YOLO boxes only): {metrics['AP']:.3f}, "
          f"Class-agnostic AR (all masks): {metrics['AR']:.3f}, mIoU: {metrics['mean_IoU']:.3f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='COCO evaluation for all three systems')
    parser.add_argument('--coco-gt', required=True, help='COCO ground truth JSON')
    parser.add_argument('--val-img-path', required=True, help='COCO val2017 images directory')
    parser.add_argument('--vitdet-json', required=True, help='ViTDet detection results JSON')
    parser.add_argument('--sam-weights', required=True, help='TinySAM weights')
    parser.add_argument('--yolo-weights', required=True, help='YOLOv12 weights')
    parser.add_argument('--num-images', type=int, default=100, help='Number of images to evaluate')
    parser.add_argument('--conf', type=float, default=0.1, help='YOLO confidence threshold')
    parser.add_argument('--iou', type=float, default=0.55, help='YOLO NMS IoU')
    parser.add_argument('--max-det', type=int, default=350, help='YOLO max detections')
    parser.add_argument('--grid-size', type=int, default=16, help='Sparse grid size for hybrid')
    parser.add_argument('--output-dir', default='outputs/coco_eval', help='Output directory')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for subset selection')
    args = parser.parse_args()
    
    # Load COCO ground truth
    print(f"Loading COCO ground truth from {args.coco_gt}...")
    coco_gt = COCO(args.coco_gt)
    
    # Load image subset
    print(f"Loading {args.num_images} random images...")
    images_info = load_coco_subset(coco_gt, args.val_img_path, args.num_images, args.random_seed)
    print(f"Loaded {len(images_info)} images")
    
    # Run evaluations
    results = {}
    
    # Hierarchical baseline (class-agnostic)
    results['hierarchical'] = eval_hierarchical_baseline(
        images_info, coco_gt, args.sam_weights,
        os.path.join(args.output_dir, 'hierarchical_baseline')
    )
    
    # Detector-based systems (with category IDs)
    results['system1'] = eval_system1_vitdet(
        images_info, coco_gt, args.sam_weights, args.vitdet_json,
        os.path.join(args.output_dir, 'system1_vitdet')
    )
    
    results['system2'] = eval_system2_yolo_only(
        images_info, coco_gt, args.sam_weights, args.yolo_weights,
        args.conf, args.iou, args.max_det,
        os.path.join(args.output_dir, 'system2_yolo_only')
    )
    
    results['system3'] = eval_system3_hybrid(
        images_info, coco_gt, args.sam_weights, args.yolo_weights,
        args.conf, args.iou, args.max_det, args.grid_size,
        os.path.join(args.output_dir, 'system3_hybrid')
    )
    
    # Print comparison
    print("\n" + "="*90)
    print("DETECTOR-BASED COMPARISON (with category labels)")
    print("="*90)
    print(f"{'System':<30} {'AP':<8} {'APs':<8} {'APm':<8} {'APl':<8} {'Time/img':<10} {'Calls/img':<10}")
    print("-"*90)
    
    for sys_name, sys_label in [('system1', '1. ViTDet→TinySAM (baseline)'), ('system2', '2. YOLO→TinySAM'), ('system3', '3. Hybrid (YOLO boxes only)')]:
        m = results[sys_name]
        print(f"{sys_label:<30} {m['AP']:<8.3f} {m['APs']:<8.3f} {m['APm']:<8.3f} {m['APl']:<8.3f} "
              f"{m['avg_time_per_image']:<10.2f} {m['avg_decoder_calls']:<10.1f}")
    
    print("="*90)
    print("Note: Hybrid AP computed from YOLO box masks only (sparse points are class-agnostic)")
    
    # Print class-agnostic comparison
    print("\n" + "="*90)
    print("CLASS-AGNOSTIC COMPARISON (Hierarchical vs Hybrid)")
    print("="*90)
    print(f"{'System':<30} {'AR':<8} {'mIoU':<8} {'Coverage':<10} {'Time/img':<10} {'Calls/img':<10}")
    print("-"*90)
    
    hier = results['hierarchical']
    hybrid = results['system3']
    print(f"{'Hierarchical (baseline)':<30} {hier['AR']:<8.3f} {hier['mean_IoU']:<8.3f} {hier['coverage']:<10.3f} "
          f"{hier['avg_time_per_image']:<10.2f} {hier['avg_decoder_calls']:<10.1f}")
    print(f"{'Hybrid (YOLO+sparse)':<30} {hybrid['AR']:<8.3f} {hybrid['mean_IoU']:<8.3f} {hybrid['coverage']:<10.3f} "
          f"{hybrid['avg_time_per_image']:<10.2f} {hybrid['avg_decoder_calls']:<10.1f}")
    
    print("="*90)
    
    # Print insights
    print("\nKEY INSIGHTS:")
    print(f"  • ViTDet baseline:       {results['system1']['AP']*100:.1f}% AP (paper: 42.3%)")
    print(f"  • YOLO-only:             {results['system2']['AP']*100:.1f}% AP")
    print(f"  • Hybrid:                {results['system3']['AP']*100:.1f}% AP")
    print(f"\n  • Hierarchical vs Hybrid:")
    print(f"    - AR: {hier['AR']:.3f} vs {hybrid['AR']:.3f} ({(hybrid['AR']/hier['AR']-1)*100:+.1f}%)")
    print(f"    - mIoU: {hier['mean_IoU']:.3f} vs {hybrid['mean_IoU']:.3f} ({(hybrid['mean_IoU']/hier['mean_IoU']-1)*100:+.1f}%)")
    print(f"    - Speedup: {hier['avg_time_per_image']/hybrid['avg_time_per_image']:.1f}x faster")
    print("="*90)
    
    # Save summary
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/summary.json")


if __name__ == '__main__':
    main()

