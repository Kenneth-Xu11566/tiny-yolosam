#!/usr/bin/env bash
set -euo pipefail

# Determine project root (/Users/.../MLRE)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Default to picture2.jpg if no image provided
IMAGE=${1:-"TinySAM/fig/picture2.jpg"}

echo "[INFO] Visualizing pipeline steps for: $IMAGE"
echo ""

python TinySAM/scripts/visualize_pipeline_steps.py \
  --image "$IMAGE" \
  --sam-weights TinySAM/weights/tinysam_42.3.pth \
  --yolo-weights weights/yolov12s_turbo.pt \
  --conf 0.1 \
  --iou 0.55 \
  --max-det 350 \
  --grid-size 16 \
  --mode both \
  --output-dir outputs/pipeline_viz

echo ""
echo "[INFO] Pipeline visualizations saved!"
echo ""
echo "YOLO-only pipeline:"
echo "  - outputs/pipeline_viz/yolo_only/pipeline_yolo_only.png"
echo "  - Individual steps: step1_input.png, step2_yolo_boxes.png, step3_tinysam_masks.png"
echo ""
echo "Hybrid pipeline:"
echo "  - outputs/pipeline_viz/hybrid/pipeline_hybrid.png"
echo "  - Individual steps: step1_input.png, step2_yolo_boxes.png, step3_sparse_points.png, step4_tinysam_masks.png"

