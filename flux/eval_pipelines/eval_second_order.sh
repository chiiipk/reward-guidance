#!/bin/bash
# eval_second_order.sh: Evaluate Second-Order Guidance on FLUX

REWARD=${1:-"imagereward"}
PROMPT=${2:-"A cute fluffy cat"}
NUM_IMAGES=16
NUM_STEPS=28
CFG=3.5

echo "========================================"
echo "Evaluating Second-Order for $REWARD"
echo "========================================"

# 1. Second-Order (Normalized GNS = 10, for apple-to-apple direction test)
python3 sample.py \
    --reward "$REWARD" \
    --prompt "$PROMPT" \
    --method "second_order" \
    --gradient-norm-scale 10.0 \
    --num-images $NUM_IMAGES \
    --num-steps $NUM_STEPS \
    --cfg-scale $CFG \
    --output-dir "./results/second_order_norm10"

# 2. Second-Order (Unnormalized, GNS = 0 -> disabled, shows automatic damping)
python3 sample.py \
    --reward "$REWARD" \
    --prompt "$PROMPT" \
    --method "second_order" \
    --gradient-norm-scale 0.0 \
    --num-images $NUM_IMAGES \
    --num-steps $NUM_STEPS \
    --cfg-scale $CFG \
    --output-dir "./results/second_order_unnorm"

echo "Done Second-Order Evaluation."
