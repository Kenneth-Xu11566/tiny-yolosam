"""
Batch processing utilities for TinySAM point prompts.
Provides efficient batched inference for sparse point sampling.

Note: Batch processing is most effective on GPU. On CPU, sequential processing
may be faster due to lower overhead. This module auto-detects the device.
"""

import numpy as np
import torch
from typing import List, Tuple


def _sequential_point_prompts(
    predictor,
    points: np.ndarray,
    return_best_only: bool,
    min_confidence: float,
    min_area: int,
    verbose: bool
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Optimized sequential processing for CPU (faster than batching on CPU).
    """
    all_masks = []
    all_scores = []
    
    processed = 0
    for i, pt in enumerate(points):
        point_coords = np.array([pt])
        point_labels = np.array([1])
        
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels
        )
        
        if return_best_only:
            best_idx = scores.argmax()
            mask = masks[best_idx]
            score = scores[best_idx]
            
            if score >= min_confidence and mask.sum() >= min_area:
                all_masks.append(mask)
                all_scores.append(float(score))
                processed += 1
        else:
            for j in range(len(masks)):
                mask = masks[j]
                score = scores[j]
                
                if score >= min_confidence and mask.sum() >= min_area:
                    all_masks.append(mask)
                    all_scores.append(float(score))
                    processed += 1
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(points)} points... ({processed} masks kept)")
    
    if verbose:
        filtered_out = len(points) - len(all_masks)
        print(f"[Sequential Processing] Generated {len(all_masks)} masks from {len(points)} points")
        if filtered_out > 0:
            print(f"  Filtered out {filtered_out} low-quality masks")
    
    return all_masks, all_scores


def batch_point_prompts(
    predictor,
    points: np.ndarray,
    batch_size: int = 64,
    multimask_output: bool = False,
    return_best_only: bool = True,
    min_confidence: float = 0.0,
    min_area: int = 0,
    verbose: bool = False,
    force_batch: bool = False
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Process multiple point prompts in batches for efficient inference.
    
    Args:
        predictor: TinySAM predictor with image already set
        points: Array of shape [N, 2] with point coordinates (x, y)
        batch_size: Number of points to process in each batch
        multimask_output: If True, output 3 masks per point; if False, output 1 mask per point
        return_best_only: If True, only return the highest-scoring mask per point
        min_confidence: Minimum IoU score to keep a mask (0.0 = keep all)
        min_area: Minimum mask area in pixels (0 = no filtering)
        verbose: Print progress information
        
    Returns:
        masks_list: List of masks (each mask is HxW boolean array)
        scores_list: List of confidence scores for each mask
    """
    if len(points) == 0:
        return [], []
    
    # Auto-detect device: CPU doesn't benefit much from batching
    device = predictor.device
    is_cuda = device.type == 'cuda'
    
    # On CPU, use optimized sequential processing unless forced
    if not is_cuda and not force_batch:
        if verbose:
            print(f"[CPU Mode] Using optimized sequential processing for {len(points)} points...")
        return _sequential_point_prompts(
            predictor, points, return_best_only, min_confidence, min_area, verbose
        )
    
    num_points = len(points)
    num_batches = (num_points + batch_size - 1) // batch_size
    
    all_masks = []
    all_scores = []
    
    if verbose:
        print(f"[Batch Processing] {num_points} points in {num_batches} batches (batch_size={batch_size}, device={device})")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_points)
        batch_points = points[start_idx:end_idx]
        batch_labels = np.ones(len(batch_points), dtype=np.int32)
        
        # Transform coordinates
        batch_points_transformed = predictor.transform.apply_coords(
            batch_points, 
            predictor.original_size
        )
        
        # Convert to torch tensors with batch dimension
        coords_torch = torch.as_tensor(
            batch_points_transformed, 
            dtype=torch.float, 
            device=predictor.device
        )
        labels_torch = torch.as_tensor(
            batch_labels, 
            dtype=torch.int, 
            device=predictor.device
        )
        
        # Add batch dimension: [N, 2] -> [B, N, 2] where B = num points in batch
        # For point prompts, we want each point to be processed independently
        # So we reshape to [B, 1, 2] where B is the number of points
        coords_torch = coords_torch.unsqueeze(1)  # [B, 1, 2]
        labels_torch = labels_torch.unsqueeze(1)  # [B, 1]
        
        # Predict masks for this batch
        # Note: predict_torch doesn't have multimask_output parameter
        # It always returns 3 masks per prompt by default
        with torch.no_grad():
            masks_batch, scores_batch, _ = predictor.predict_torch(
                point_coords=coords_torch,
                point_labels=labels_torch,
            )
        
        # Process results: masks_batch shape is [B, C, H, W]
        # where B is batch size, C is number of masks per prompt (1 or 3)
        masks_batch = masks_batch.cpu().numpy()
        scores_batch = scores_batch.cpu().numpy()
        
        for i in range(len(batch_points)):
            point_masks = masks_batch[i]  # [C, H, W]
            point_scores = scores_batch[i]  # [C]
            
            if return_best_only:
                # Take only the best mask
                best_idx = point_scores.argmax()
                mask = point_masks[best_idx]
                score = point_scores[best_idx]
                
                # Apply filters
                if score >= min_confidence and mask.sum() >= min_area:
                    all_masks.append(mask)
                    all_scores.append(float(score))
            else:
                # Return all masks for this point
                for j in range(len(point_masks)):
                    mask = point_masks[j]
                    score = point_scores[j]
                    
                    if score >= min_confidence and mask.sum() >= min_area:
                        all_masks.append(mask)
                        all_scores.append(float(score))
        
        if verbose and (batch_idx + 1) % 5 == 0:
            print(f"  Processed {end_idx}/{num_points} points...")
    
    if verbose:
        filtered_out = num_points - len(all_masks)
        print(f"[Batch Processing] Generated {len(all_masks)} masks from {num_points} points")
        if filtered_out > 0:
            print(f"  Filtered out {filtered_out} low-quality masks")
    
    return all_masks, all_scores


def batch_box_prompts(
    predictor,
    boxes: np.ndarray,
    batch_size: int = 32,
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Process multiple box prompts in batches for efficient inference.
    
    Args:
        predictor: TinySAM predictor with image already set
        boxes: Array of shape [N, 4] with box coordinates (x1, y1, x2, y2)
        batch_size: Number of boxes to process in each batch
        verbose: Print progress information
        
    Returns:
        masks_list: List of masks (each mask is HxW boolean array)
        scores_list: List of confidence scores for each mask
    """
    if len(boxes) == 0:
        return [], []
    
    num_boxes = len(boxes)
    num_batches = (num_boxes + batch_size - 1) // batch_size
    
    all_masks = []
    all_scores = []
    
    if verbose:
        print(f"[Batch Processing] {num_boxes} boxes in {num_batches} batches (batch_size={batch_size})")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_boxes)
        batch_boxes = boxes[start_idx:end_idx]
        
        # Transform boxes
        batch_boxes_transformed = predictor.transform.apply_boxes(
            batch_boxes,
            predictor.original_size
        )
        
        # Convert to torch tensor
        boxes_torch = torch.as_tensor(
            batch_boxes_transformed,
            dtype=torch.float,
            device=predictor.device
        )
        
        # Predict masks for this batch
        # Note: predict_torch doesn't have multimask_output parameter
        # It always returns 3 masks per prompt by default
        with torch.no_grad():
            masks_batch, scores_batch, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=boxes_torch,
            )
        
        # Process results
        masks_batch = masks_batch.cpu().numpy()
        scores_batch = scores_batch.cpu().numpy()
        
        for i in range(len(batch_boxes)):
            point_masks = masks_batch[i]  # [C, H, W]
            point_scores = scores_batch[i]  # [C]
            
            # Take best mask
            best_idx = point_scores.argmax()
            all_masks.append(point_masks[best_idx])
            all_scores.append(float(point_scores[best_idx]))
    
    if verbose:
        print(f"[Batch Processing] Generated {len(all_masks)} masks from {num_boxes} boxes")
    
    return all_masks, all_scores

