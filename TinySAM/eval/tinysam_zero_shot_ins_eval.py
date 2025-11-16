import os,sys
import numpy as np
import cv2
# Import pycocotools BEFORE torch/tinysam to avoid initialization conflicts
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
#from lvis.eval import LVISEval
import json
import random
# Now import torch-related modules
import torch
import matplotlib.pyplot as plt
sys.path.append("..")
from tinysam import sam_model_registry, SamPredictor
import argparse

def eval_zero_shot(eval_type,val_img_path,val_json_path,vit_det_file_path,sam_checkpoint_path,num_images=None,random_seed=42):
    if eval_type=='coco':
        print("============== Evaluating on COCO dataset:",sam_checkpoint_path)
    elif eval_type=='lvis':
        print("============== Evaluating on LVIS dataset:",sam_checkpoint_path)
    else: 
        print("Error! Unsupported evaluation dataset!")
        return
        
    with open(vit_det_file_path) as f:
         res = json.load(f)
    
    # Random sampling of images if num_images is specified
    if num_images is not None and num_images > 0:
        print(f"\n=== Randomly sampling {num_images} images (seed={random_seed}) ===")
        # Get unique image IDs
        all_image_ids = list(set([det['image_id'] for det in res]))
        print(f"Total images available: {len(all_image_ids)}")
        
        # Random sample
        random.seed(random_seed)
        if num_images >= len(all_image_ids):
            sampled_image_ids = set(all_image_ids)
            print(f"Using all {len(all_image_ids)} images")
        else:
            sampled_image_ids = set(random.sample(all_image_ids, num_images))
            print(f"Sampled {num_images} random images")
        
        # Filter detections for sampled images only
        res = [det for det in res if det['image_id'] in sampled_image_ids]
        print(f"Filtered to {len(res)} detections")
    
    model_type = "vit_t"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Import quantization layers for quantized model
    if 'w8a8' in sam_checkpoint_path or 'quant' in sam_checkpoint_path.lower():
        print("Loading quantized model...")
        import tinysam.quantization_layer as quantization_layer
        sys.modules['quantization_layer'] = quantization_layer
        sam = torch.load(sam_checkpoint_path, map_location='cpu')
    else:
        print("Loading full precision model...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    pre_img_id=0
    total_time=0
    print(f"\nTotal detections to process: {len(res)}")
    
    # Track unique images processed
    processed_images = set()

    for i,res_ins in enumerate(res):
        res_ins=res[i]
        # Better progress tracking
        if i % 100 == 0:
            print(f"Progress: {i}/{len(res)} detections processed ({len(processed_images)} images)")
        
        img_id=res_ins['image_id']
        img_file_name=f'{img_id:012d}'+'.jpg'

        if pre_img_id!=img_id:
            image = cv2.imread(val_img_path+img_file_name)
            if image is None:
                print(f"Warning: Could not load image {img_file_name}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            pre_img_id=img_id
            processed_images.add(img_id)

        input_box=res_ins['bbox']
        input_box=[input_box[0],input_box[1],input_box[0]+input_box[2],input_box[1]+input_box[3]]
        input_box = np.array(input_box)
        masks, ious, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
        )
        
        new_seg=mask_util.encode(np.array(masks[np.argmax(ious)],order="F", dtype="uint8"))
        new_seg["counts"] = new_seg["counts"].decode("utf-8")
        res[i]["segmentation"]=new_seg
    
    print(f"\nProcessed {len(res)} detections across {len(processed_images)} images")

    for c in res:
         c.pop("bbox", None)
    save_res_json_file=eval_type+'_res_tinysam.json'
    
    with open(save_res_json_file, 'w') as fnew:
        json.dump(res, fnew)
        
    if eval_type=='coco':
        cocoGT= COCO(val_json_path)
        
        # If we sampled images, filter ground truth to match
        if num_images is not None and num_images > 0:
            # Get image IDs from our predictions
            pred_img_ids = list(set([r['image_id'] for r in res]))
            print(f"\nFiltering ground truth to {len(pred_img_ids)} predicted images...")
            
            # Filter to only evaluate on images we have predictions for
            cocoGT.imgs = {k: v for k, v in cocoGT.imgs.items() if k in pred_img_ids}
            cocoGT.imgToAnns = {k: v for k, v in cocoGT.imgToAnns.items() if k in pred_img_ids}
        
        coco_dt = cocoGT.loadRes(res)
        coco_eval = COCOeval(cocoGT, coco_dt, "segm")
        
        # Only evaluate on images we have predictions for
        if num_images is not None and num_images > 0:
            coco_eval.params.imgIds = sorted(cocoGT.imgs.keys())
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Print cleaner summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Model: {sam_checkpoint_path}")
        print(f"Images evaluated: {len(pred_img_ids) if num_images else 'all'}")
        print(f"Total detections: {len(res)}")
        print("-"*70)
        print(f"{'Metric':<40} {'Score':<10}")
        print("-"*70)
        stats = coco_eval.stats
        print(f"{'AP @ IoU=0.50:0.95 (main metric)':<40} {stats[0]*100:>6.2f}%")
        print(f"{'AP @ IoU=0.50':<40} {stats[1]*100:>6.2f}%")
        print(f"{'AP @ IoU=0.75':<40} {stats[2]*100:>6.2f}%")
        print(f"{'AP (small objects)':<40} {stats[3]*100:>6.2f}%")
        print(f"{'AP (medium objects)':<40} {stats[4]*100:>6.2f}%")
        print(f"{'AP (large objects)':<40} {stats[5]*100:>6.2f}%")
        print("-"*70)
        print(f"{'AR @ maxDets=1':<40} {stats[6]*100:>6.2f}%")
        print(f"{'AR @ maxDets=10':<40} {stats[7]*100:>6.2f}%")
        print(f"{'AR @ maxDets=100':<40} {stats[8]*100:>6.2f}%")
        print("="*70)
        
        # Compare to paper if using tinysam_42.3.pth
        if 'tinysam_42.3' in sam_checkpoint_path or 'tinysam.pth' in sam_checkpoint_path:
            print(f"\nPaper (TinySAM, 5000 imgs):     42.3% AP")
            print(f"Your result ({len(pred_img_ids) if num_images else 5000} imgs): {stats[0]*100:>6.2f}% AP")
        elif 'w8a8' in sam_checkpoint_path:
            print(f"\nPaper (Q-TinySAM, 5000 imgs):   41.4% AP")
            print(f"Your result ({len(pred_img_ids) if num_images else 5000} imgs): {stats[0]*100:>6.2f}% AP")
        print("="*70 + "\n")
        
        return
    elif eval_type=='lvis':
        lvis_eval = LVISEval(val_json_path, save_res_json_file, "segm")
        lvis_eval.run()
        lvis_eval.print_results()
        return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='zero shot instance segmentation on COCO or lvis')
    parser.add_argument('--eval_type', type=str, default='coco', help='coco or lvis')
    parser.add_argument('--val_img_path', type=str, default="val2017/", help='path to validation imgs')
    parser.add_argument('--val_json_path', type=str, default="json_files/instances_val2017.json", help='path to val2017 annotation json file')
    parser.add_argument('--vit_det_file_path', type=str, default="json_files/coco_instances_results_vitdet.json", help='path to vitdet detection results json file')
    parser.add_argument('--sam_checkpoint_path', type=str, default="../weights/tinysam_42.3.pth", help='path to ckpt file')
    parser.add_argument('--num_images', type=int, default=None, help='number of images to randomly sample (None = use all images)')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed for reproducibility')
    
    args = parser.parse_args()
    eval_zero_shot(
        eval_type=args.eval_type,
        val_img_path=args.val_img_path,
        val_json_path=args.val_json_path,
        vit_det_file_path=args.vit_det_file_path,
        sam_checkpoint_path=args.sam_checkpoint_path,
        num_images=args.num_images,
        random_seed=args.random_seed
    )
    