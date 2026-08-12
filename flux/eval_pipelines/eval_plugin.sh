#!/bin/bash
# eval_plugin.sh: Evaluate Plug-in (k=1) and Plug-in + Damping baselines.

REWARD=${1:-"imagereward"}
PROMPT=${2:-"A cute fluffy cat"}
NUM_IMAGES=16
NUM_STEPS=28
CFG=3.5

echo "========================================"
echo "Evaluating Plug-in (k=1) for $REWARD"
echo "========================================"

# 1. Plug-in k=1 (Normalized GNS = 10)
python3 sample.py \
    --reward "$REWARD" \
    --prompt "$PROMPT" \
    --gradient-norm-scale 10.0 \
    --num-images $NUM_IMAGES \
    --num-steps $NUM_STEPS \
    --cfg-scale $CFG \
    --output-dir "./results/plugin_k1_norm10"

# 2. Plug-in k=1 + Damping (sigma-damp = 0.5)
python3 sample.py \
    --reward "$REWARD" \
    --prompt "$PROMPT" \
    --gradient-norm-scale 10.0 \
    --sigma-damp 0.5 \
    --num-images $NUM_IMAGES \
    --num-steps $NUM_STEPS \
    --cfg-scale $CFG \
    --output-dir "./results/plugin_k1_damp_norm10"

echo "Done Plug-in Evaluation."
