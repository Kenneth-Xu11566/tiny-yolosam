#!/usr/bin/env bash
set -euo pipefail

# Determine project root (/Users/.../MLRE)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "[INFO] Running Step 2: TinySAM box prompting"
python TinySAM/TinySAM/scripts/run_tinysam_box_prompt.py \
  --image TinySAM/fig/picture2.jpg \
  --yolo-weights weights/yolov12s_turbo.pt \
  --sam-weights TinySAM/weights/tinysam_42.3.pth \
  --conf 0.1 \
  --iou 0.55 \
  --max-det 350 \
  --output-dir outputs/yolo_box_masks \
  --save-composite

