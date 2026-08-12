#!/bin/bash
# eval_second_order_bo4.sh: Evaluate Second-Order Guidance + Bo4 on FLUX
# Generates 64 images (16 * 4) and we will later pick the top 16 based on the saved rewards.npy

REWARD=${1:-"imagereward"}
PROMPT=${2:-"A cute fluffy cat"}
NUM_IMAGES=64
NUM_STEPS=28
CFG=3.5

echo "========================================"
echo "Evaluating Second-Order + Bo4 for $REWARD"
echo "========================================"

# We only run the Unnormalized mode for Bo4, as requested to show max stability and performance
python3 sample.py \
    --reward "$REWARD" \
    --prompt "$PROMPT" \
    --method "second_order" \
    --gradient-norm-scale 0.0 \
    --num-images $NUM_IMAGES \
    --num-steps $NUM_STEPS \
    --cfg-scale $CFG \
    --output-dir "./results/second_order_bo4_unnorm"

echo "Done Second-Order Bo4 Evaluation (Generated 64 images)."
