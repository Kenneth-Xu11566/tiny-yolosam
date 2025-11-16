#!/usr/bin/env bash
set -euo pipefail

# Determine project root (/Users/.../MLRE)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "[INFO] Running Steps 3-6: Hybrid Pipeline (Coverage + Sparse Points + Merging)"
python TinySAM/TinySAM/scripts/run_hybrid_pipeline.py \
  --image TinySAM/fig/picture2.jpg \
  --sam-weights TinySAM/weights/tinysam_42.3.pth \
  --yolo-metadata outputs/yolo_box_masks/metadata.json \
  --grid-size 16 \
  --iou-threshold 0.7 \
  --min-area 100 \
  --output-dir outputs/hybrid_final

