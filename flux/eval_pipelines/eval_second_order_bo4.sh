#!/bin/bash
# eval_second_order_bo4.sh: Evaluate Second-Order Guidance + Bo4 on FLUX
# Generates 64 images (16 * 4) and we will later pick the top 16 based on the saved rewards.npy
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

REWARD=${1:-"imagereward"}
PROMPT=${2:-"A cute fluffy cat"}
NUM_IMAGES=${NUM_IMAGES:-64}
NUM_STEPS=${NUM_STEPS:-28}
CFG=${CFG:-3.5}

echo "========================================"
echo "Evaluating Second-Order + Bo4 for $REWARD"
echo "========================================"

# Run the unnormalized mode and select the best outputs using rewards.npy.
python3 sample.py \
    --reward "$REWARD" \
    --prompt "$PROMPT" \
    --method second_order \
    --lam 1.0 \
    --gradient-norm-scale 0.0 \
    --num-images "$NUM_IMAGES" \
    --num-steps "$NUM_STEPS" \
    --cfg-scale "$CFG" \
    --verbose \
    --output-dir "./results/second_order_bo4_unnorm"

echo "Done Second-Order Bo4 Evaluation (generated $NUM_IMAGES images)."
