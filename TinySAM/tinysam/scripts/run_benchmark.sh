#!/usr/bin/env bash
set -euo pipefail

# Determine project root (/Users/.../MLRE)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "[INFO] Benchmarking all three systems on picture2.jpg"
python TinySAM/TinySAM/scripts/benchmark_all_systems.py \
  --image TinySAM/fig/picture2.jpg \
  --sam-weights TinySAM/weights/tinysam_42.3.pth \
  --yolo-weights weights/yolov12s_turbo.pt \
  --conf 0.1 \
  --iou 0.55 \
  --max-det 350 \
  --grid-size 16 \
  --output-dir outputs/benchmark

echo ""
echo "[INFO] Benchmark complete! Results saved to outputs/benchmark/"
echo "[INFO] Visual comparison: outputs/benchmark/visual_comparison.png"
echo "[INFO] Individual composites:"
echo "  - outputs/benchmark/system1_hierarchical/composite.png"
echo "  - outputs/benchmark/system2_yolo_only/composite.png"
echo "  - outputs/benchmark/system3_hybrid/composite.png"

