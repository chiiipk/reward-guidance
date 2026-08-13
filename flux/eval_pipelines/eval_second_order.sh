#!/bin/bash
# eval_second_order.sh: Evaluate Second-Order Guidance on FLUX
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

REWARD=${1:-"imagereward"}
PROMPT=${2:-"A cute fluffy cat"}
NUM_IMAGES=${NUM_IMAGES:-16}
NUM_STEPS=${NUM_STEPS:-28}
CFG=${CFG:-3.5}

echo "========================================"
echo "Evaluating Second-Order for $REWARD"
echo "========================================"

# 1. Second-Order (Normalized GNS = 10, for apple-to-apple direction test)
python3 sample.py \
    --reward "$REWARD" \
    --prompt "$PROMPT" \
    --method second_order \
    --lam 1.0 \
    --gradient-norm-scale 10.0 \
    --num-images "$NUM_IMAGES" \
    --num-steps "$NUM_STEPS" \
    --cfg-scale "$CFG" \
    --verbose \
    --output-dir "./results/second_order_norm10"

# 2. Second-Order (Unnormalized, GNS = 0 -> disabled, shows automatic damping)
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
    --output-dir "./results/second_order_unnorm"

echo "Done Second-Order Evaluation."
