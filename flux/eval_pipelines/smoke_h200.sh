#!/bin/bash
# Minimal end-to-end H200 test: CUDA/Triton preflight plus one generated image.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

if [ -n "${H200_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$H200_GPU"
fi

python flux/h200_preflight.py

cd flux
python sample.py \
    --reward palette \
    --palette cool_ocean \
    --prompt "A small sailboat on a calm ocean at sunrise" \
    --method second_order \
    --lam 1.0 \
    --gradient-norm-scale 10.0 \
    --num-guidance-steps 1 \
    --num-images 1 \
    --num-steps "${NUM_STEPS:-8}" \
    --height "${IMAGE_SIZE:-256}" \
    --width "${IMAGE_SIZE:-256}" \
    --verbose \
    --output-dir ./results/h200_smoke

echo "H200 image smoke test: PASS"
