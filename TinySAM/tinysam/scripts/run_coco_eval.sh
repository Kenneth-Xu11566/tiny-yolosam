#!/usr/bin/env bash
set -euo pipefail

# Determine project root (/Users/.../MLRE)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Default: evaluate on 100 images
NUM_IMAGES=${1:-100}

echo "[INFO] Running COCO evaluation on $NUM_IMAGES images..."
echo "[INFO] Comparing 3 systems:"
echo "  1. ViTDet → TinySAM (paper baseline)"
echo "  2. YOLO → TinySAM (YOLO-only)"
echo "  3. Hybrid (YOLO + sparse points)"
echo ""
echo "[INFO] Estimated time on M4:"
echo "  - 100 images: ~1-2 hours"
echo "  - 500 images: ~5-10 hours"
echo ""

python TinySAM/TinySAM/scripts/eval_coco_all_systems.py \
  --coco-gt TinySAM/eval/json_files/instances_val2017.json \
  --val-img-path TinySAM/eval/val2017/ \
  --vitdet-json TinySAM/eval/json_files/coco_instances_results_vitdet.json \
  --sam-weights TinySAM/weights/tinysam_42.3.pth \
  --yolo-weights weights/yolov12s_turbo.pt \
  --num-images $NUM_IMAGES \
  --conf 0.1 \
  --iou 0.55 \
  --max-det 350 \
  --grid-size 16 \
  --output-dir outputs/coco_eval_${NUM_IMAGES}img

echo ""
echo "[INFO] Evaluation complete!"
echo "[INFO] Results saved to outputs/coco_eval_${NUM_IMAGES}img/summary.json"

