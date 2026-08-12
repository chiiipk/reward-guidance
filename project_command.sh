#!/bin/bash
set -e

# ==============================================================================
# project_command.sh — Reproduce toàn bộ paper + so sánh second-order
#
# Bao gồm:
#   1. Cài đặt môi trường
#   2. Gaussian mixture (CPU)
#   3. Mode selection 1D (CPU)
#   4. Checkerboard: train + sample + figure (GPU, ~60 phút train)
#   5. FLUX: tất cả 9 figure trong paper (GPU ≥ 48GB)
#   6. FLUX second-order (ours) trên ImageReward — apple-to-apple
#   7. Tổng hợp kết quả → export_results/ (< 25MB)
#
# Cách chạy:   bash project_command.sh
# Yêu cầu:     GPU ≥ 48GB, Python 3.10+, huggingface-cli login
# ==============================================================================

EXPORT_DIR="export_results"
FLUX_COMMON="--num-steps 28 --height 512 --width 512 --cfg-scale 3.5 --snr-factor 5 --num-guidance-steps 5 --guidance-start-step 1 --reward-scale 1 --num-images 20"

# Chỉ định chạy trên GPU 6 và 7
export CUDA_VISIBLE_DEVICES="6,7"

# Mảng chứa các lệnh chạy FLUX để sau đó phân bổ song song cho các GPU
rm -f flux_commands.txt

# Helper: lưu 1 condition cho 1 figure vào file thay vì chạy ngay
run_flux() {
    local FIGURE=$1; shift
    local CONDITION=$1; shift
    local OUTDIR="../data/${FIGURE}/${CONDITION}"
    # Lưu command thành chuỗi để Python đọc và phân bổ GPU sau
    echo "python sample.py $@ --output-dir $OUTDIR" >> flux_commands.txt
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. CÀI ĐẶT MÔI TRƯỜNG
# ──────────────────────────────────────────────────────────────────────────────
echo "=== 1. Cài đặt môi trường ==="

cat << 'EOF' > pyproject.toml
[project]
name = "reward-guidance"
version = "0.1.0"
description = "Reward Guidance — full paper reproduction"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "torchvision",
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "pillow>=10.0",
    "tqdm>=4.65",
    "diffusers>=0.30",
    "transformers>=4.44",
    "accelerate>=0.30",
    "sentencepiece",
    "protobuf",
    "image-reward",
    "openai-clip"
]
EOF

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml

# ──────────────────────────────────────────────────────────────────────────────
# 2. GAUSSIAN MIXTURE (CPU, ~10s)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 2. Gaussian mixture ==="
cd gaussian_mixture/
python make_grid_figures.py
python make_fmrg_figure.py
cd ..

# ──────────────────────────────────────────────────────────────────────────────
# 3. MODE SELECTION 1D (CPU, ~10s)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 3. Mode selection 1D ==="
cd mode_selection/
python sample.py --reward step     --num-samples 2000 --max-n 16 --lam 5.0 \
    --record-trajectories --output-dir results/step_lam5.0
python sample.py --reward gaussian --num-samples 2000 --max-n 16 --lam 5.0 \
    --record-trajectories --output-dir results/gaussian_lam5.0
python make_overview_figure.py
python make_trajectory_figures.py
cd ..

# ──────────────────────────────────────────────────────────────────────────────
# 4. CHECKERBOARD (GPU, ~60 min train + vài phút sample)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 4. Checkerboard ==="
cd checkerboard/

if [ ! -f "results/velocity_net.pt" ]; then
    echo "  Training (500k steps)..."
    python train.py --num-steps 500000
else
    echo "  Checkpoint đã tồn tại, skip training."
fi

python sample.py --analytic-tilt --lam 10.0
python sample.py --k 1 --lam 10.0 --num-samples 20000
python sample.py --k 8 --lam 10.0 --num-samples 5000
python sample.py --k 1 --lam 10.0 --sigma-damp 0.2 --num-samples 20000
python make_main_figure.py
python plot.py --bon-vs-softmax --lam 10.0
cd ..

# ──────────────────────────────────────────────────────────────────────────────
# 5. FLUX — TẤT CẢ 9 FIGURE TRONG PAPER
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 5. FLUX — Reproduce toàn bộ paper figures ==="
cd flux/

# ═══════════════════════════════════════════════════════════════════════
# Figure: blueness_fox (appendix)
# Reward: blue_minus_rg
# ═══════════════════════════════════════════════════════════════════════
echo "  --- blueness_fox ---"
FOX_PROMPT="a baby fox wearing a cozy knitted sweater"

run_flux blueness_fox unguided \
    --reward blue_minus_rg --prompt "$FOX_PROMPT" \
    --reward-scale 0 $FLUX_COMMON

run_flux blueness_fox gns100 \
    --reward blue_minus_rg --prompt "$FOX_PROMPT" \
    --gradient-norm-scale 100 $FLUX_COMMON

run_flux blueness_fox gns50 \
    --reward blue_minus_rg --prompt "$FOX_PROMPT" \
    --gradient-norm-scale 50 $FLUX_COMMON

run_flux blueness_fox gns50_k8 \
    --reward blue_minus_rg --prompt "$FOX_PROMPT" \
    --gradient-norm-scale 50 --num-particles 8 --lam 1.0 $FLUX_COMMON

run_flux blueness_fox gns100_damp0.1 \
    --reward blue_minus_rg --prompt "$FOX_PROMPT" \
    --gradient-norm-scale 100 --sigma-damp 0.1 $FLUX_COMMON

# ═══════════════════════════════════════════════════════════════════════
# Figure: blueness_rococo (main text)
# Reward: blue_minus_rg
# ═══════════════════════════════════════════════════════════════════════
echo "  --- blueness_rococo ---"
ROCOCO_PROMPT="Artist painting in the center of a cluttered room lit by candlelight, rococo"

run_flux blueness_rococo unguided \
    --reward blue_minus_rg --prompt "$ROCOCO_PROMPT" \
    --reward-scale 0 $FLUX_COMMON

run_flux blueness_rococo gns50 \
    --reward blue_minus_rg --prompt "$ROCOCO_PROMPT" \
    --gradient-norm-scale 50 $FLUX_COMMON

run_flux blueness_rococo gns30 \
    --reward blue_minus_rg --prompt "$ROCOCO_PROMPT" \
    --gradient-norm-scale 30 $FLUX_COMMON

run_flux blueness_rococo gns50_k8 \
    --reward blue_minus_rg --prompt "$ROCOCO_PROMPT" \
    --gradient-norm-scale 50 --num-particles 8 --lam 1.0 $FLUX_COMMON

run_flux blueness_rococo gns100_damp0.1 \
    --reward blue_minus_rg --prompt "$ROCOCO_PROMPT" \
    --gradient-norm-scale 100 --sigma-damp 0.1 $FLUX_COMMON

# ═══════════════════════════════════════════════════════════════════════
# Figure: masked_brightness_welder (main text)
# Reward: masked_brightness --mask-region topright_circle
# ═══════════════════════════════════════════════════════════════════════
echo "  --- masked_brightness_welder ---"
WELDER_PROMPT="Photorealistic worm's-eye view of a welder mid-spark inside a rusted ship hull, sweat, smoke, orange backlight"

run_flux masked_brightness_welder unguided \
    --reward masked_brightness --mask-region topright_circle --prompt "$WELDER_PROMPT" \
    --reward-scale 0 $FLUX_COMMON

run_flux masked_brightness_welder gns100 \
    --reward masked_brightness --mask-region topright_circle --prompt "$WELDER_PROMPT" \
    --gradient-norm-scale 100 $FLUX_COMMON

run_flux masked_brightness_welder gns50 \
    --reward masked_brightness --mask-region topright_circle --prompt "$WELDER_PROMPT" \
    --gradient-norm-scale 50 $FLUX_COMMON

run_flux masked_brightness_welder gns50_k8 \
    --reward masked_brightness --mask-region topright_circle --prompt "$WELDER_PROMPT" \
    --gradient-norm-scale 50 --num-particles 8 --lam 1.0 $FLUX_COMMON

run_flux masked_brightness_welder gns100_damp0.1 \
    --reward masked_brightness --mask-region topright_circle --prompt "$WELDER_PROMPT" \
    --gradient-norm-scale 100 --sigma-damp 0.1 $FLUX_COMMON

# ═══════════════════════════════════════════════════════════════════════
# Figure: imagereward_archaeologist
# Reward: imagereward, damp σ=0.15
# ═══════════════════════════════════════════════════════════════════════
echo "  --- imagereward_archaeologist ---"
ARCH_PROMPT="a young archaeologist gently brushing dust from an ancient ceramic vase, soft museum lighting, intricate details, cinematic composition"

run_flux imagereward_archaeologist unguided \
    --reward imagereward --prompt "$ARCH_PROMPT" --ir-prompt "$ARCH_PROMPT" \
    --reward-scale 0 $FLUX_COMMON

run_flux imagereward_archaeologist gns100 \
    --reward imagereward --prompt "$ARCH_PROMPT" --ir-prompt "$ARCH_PROMPT" \
    --gradient-norm-scale 100 $FLUX_COMMON

run_flux imagereward_archaeologist gns50 \
    --reward imagereward --prompt "$ARCH_PROMPT" --ir-prompt "$ARCH_PROMPT" \
    --gradient-norm-scale 50 $FLUX_COMMON

run_flux imagereward_archaeologist gns50_k8 \
    --reward imagereward --prompt "$ARCH_PROMPT" --ir-prompt "$ARCH_PROMPT" \
    --gradient-norm-scale 50 --num-particles 8 --lam 1.0 $FLUX_COMMON

run_flux imagereward_archaeologist gns100_damp0.15 \
    --reward imagereward --prompt "$ARCH_PROMPT" --ir-prompt "$ARCH_PROMPT" \
    --gradient-norm-scale 100 --sigma-damp 0.15 $FLUX_COMMON

# ═══════════════════════════════════════════════════════════════════════
# Figure: imagereward_miner
# Reward: imagereward, damp σ=0.10
# ═══════════════════════════════════════════════════════════════════════
echo "  --- imagereward_miner ---"
MINER_PROMPT="a coal miner pausing for a moment underground, hard hat lamp glowing, dust in the air, painterly chiaroscuro"

run_flux imagereward_miner unguided \
    --reward imagereward --prompt "$MINER_PROMPT" --ir-prompt "$MINER_PROMPT" \
    --reward-scale 0 $FLUX_COMMON

run_flux imagereward_miner gns100 \
    --reward imagereward --prompt "$MINER_PROMPT" --ir-prompt "$MINER_PROMPT" \
    --gradient-norm-scale 100 $FLUX_COMMON

run_flux imagereward_miner gns50 \
    --reward imagereward --prompt "$MINER_PROMPT" --ir-prompt "$MINER_PROMPT" \
    --gradient-norm-scale 50 $FLUX_COMMON

run_flux imagereward_miner gns50_k8 \
    --reward imagereward --prompt "$MINER_PROMPT" --ir-prompt "$MINER_PROMPT" \
    --gradient-norm-scale 50 --num-particles 8 --lam 1.0 $FLUX_COMMON

run_flux imagereward_miner gns100_damp0.10 \
    --reward imagereward --prompt "$MINER_PROMPT" --ir-prompt "$MINER_PROMPT" \
    --gradient-norm-scale 100 --sigma-damp 0.10 $FLUX_COMMON

# ═══════════════════════════════════════════════════════════════════════
# Figure: imagereward_market
# Reward: imagereward, damp σ=0.05
# ═══════════════════════════════════════════════════════════════════════
echo "  --- imagereward_market ---"
MARKET_PROMPT="a vibrant Indian outdoor market with colorful stalls and produce"

run_flux imagereward_market unguided \
    --reward imagereward --prompt "$MARKET_PROMPT" --ir-prompt "$MARKET_PROMPT" \
    --reward-scale 0 $FLUX_COMMON

run_flux imagereward_market gns50 \
    --reward imagereward --prompt "$MARKET_PROMPT" --ir-prompt "$MARKET_PROMPT" \
    --gradient-norm-scale 50 $FLUX_COMMON

run_flux imagereward_market gns30 \
    --reward imagereward --prompt "$MARKET_PROMPT" --ir-prompt "$MARKET_PROMPT" \
    --gradient-norm-scale 30 $FLUX_COMMON

run_flux imagereward_market gns50_k8 \
    --reward imagereward --prompt "$MARKET_PROMPT" --ir-prompt "$MARKET_PROMPT" \
    --gradient-norm-scale 50 --num-particles 8 --lam 1.0 $FLUX_COMMON

run_flux imagereward_market gns100_damp0.05 \
    --reward imagereward --prompt "$MARKET_PROMPT" --ir-prompt "$MARKET_PROMPT" \
    --gradient-norm-scale 100 --sigma-damp 0.05 $FLUX_COMMON

# ═══════════════════════════════════════════════════════════════════════
# Figure: vlm_diner_eclipse
# Reward: skywork (Qwen2.5-VL-3B)
# ═══════════════════════════════════════════════════════════════════════
echo "  --- vlm_diner_eclipse ---"
DINER_PROMPT="A roadside American diner in the Nevada desert, shot at twilight, a neon sign on the roof glowing ECLIPSE DINER in cherry-red and cream tubes, a long empty highway behind it, painterly warm light on chrome surfaces"
DINER_Q="Does this image clearly show a neon sign with the word 'ECLIPSE' as the main readable text? Answer Yes or No."

run_flux vlm_diner_eclipse unguided \
    --reward skywork --prompt "$DINER_PROMPT" \
    --skywork-question "$DINER_Q" --skywork-model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --reward-scale 0 $FLUX_COMMON

run_flux vlm_diner_eclipse gns100 \
    --reward skywork --prompt "$DINER_PROMPT" \
    --skywork-question "$DINER_Q" --skywork-model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --gradient-norm-scale 100 $FLUX_COMMON

run_flux vlm_diner_eclipse gns50_k8 \
    --reward skywork --prompt "$DINER_PROMPT" \
    --skywork-question "$DINER_Q" --skywork-model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --gradient-norm-scale 50 --num-particles 8 --lam 1.0 $FLUX_COMMON

run_flux vlm_diner_eclipse gns100_damp0.1 \
    --reward skywork --prompt "$DINER_PROMPT" \
    --skywork-question "$DINER_Q" --skywork-model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --gradient-norm-scale 100 --sigma-damp 0.1 $FLUX_COMMON

# ═══════════════════════════════════════════════════════════════════════
# Figure: vlm_subway_mars
# Reward: skywork (Qwen2.5-VL-3B)
# ═══════════════════════════════════════════════════════════════════════
echo "  --- vlm_subway_mars ---"
SUBWAY_PROMPT="cyberpunk subway platform with a holographic display that says NEXT TRAIN MARS, teal neon, commuters in silhouette"
SUBWAY_Q="Does this image clearly show a display with the text 'NEXT TRAIN MARS' as the main readable text? Answer Yes or No."

run_flux vlm_subway_mars unguided \
    --reward skywork --prompt "$SUBWAY_PROMPT" \
    --skywork-question "$SUBWAY_Q" --skywork-model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --reward-scale 0 $FLUX_COMMON

run_flux vlm_subway_mars gns100 \
    --reward skywork --prompt "$SUBWAY_PROMPT" \
    --skywork-question "$SUBWAY_Q" --skywork-model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --gradient-norm-scale 100 $FLUX_COMMON

run_flux vlm_subway_mars gns50_k8 \
    --reward skywork --prompt "$SUBWAY_PROMPT" \
    --skywork-question "$SUBWAY_Q" --skywork-model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --gradient-norm-scale 50 --num-particles 8 --lam 1.0 $FLUX_COMMON

run_flux vlm_subway_mars gns100_damp0.1 \
    --reward skywork --prompt "$SUBWAY_PROMPT" \
    --skywork-question "$SUBWAY_Q" --skywork-model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --gradient-norm-scale 100 --sigma-damp 0.1 $FLUX_COMMON

# ═══════════════════════════════════════════════════════════════════════
# Figure: fmrg_blueness_dragon (appendix)
# Reward: blue_minus_rg — so sánh plugin vs FMRG
# ═══════════════════════════════════════════════════════════════════════
echo "  --- fmrg_blueness_dragon ---"
DRAGON_PROMPT="a massive dragon perched on basalt cliffs above lava waterfalls, volcanic ash, crimson sunset, ultra-detailed fantasy"

run_flux fmrg_blueness_dragon dragon_unguided \
    --reward blue_minus_rg --prompt "$DRAGON_PROMPT" \
    --reward-scale 0 $FLUX_COMMON

run_flux fmrg_blueness_dragon dragon_plugin \
    --reward blue_minus_rg --prompt "$DRAGON_PROMPT" \
    --gradient-norm-scale 50 --snr-factor 5 \
    --num-steps 28 --height 512 --width 512 --cfg-scale 3.5 \
    --num-guidance-steps 5 --guidance-start-step 1 --reward-scale 1 --num-images 20

run_flux fmrg_blueness_dragon dragon_fmrg \
    --reward blue_minus_rg --prompt "$DRAGON_PROMPT" \
    --gradient-norm-scale 50 --snr-factor 1 \
    --num-steps 28 --height 512 --width 512 --cfg-scale 3.5 \
    --num-guidance-steps 5 --guidance-start-step 1 --reward-scale 1 --num-images 20

# Render tất cả figures
echo "  --- Rendering all figures ---"
for fig in ../figures/*/; do
    if [ -f "$fig/regenerate.py" ]; then
        echo "    Rendering $(basename $fig)..."
        (cd "$fig" && python regenerate.py) || echo "    WARNING: $(basename $fig) failed"
    fi
done

# ──────────────────────────────────────────────────────────────────────────────
# 6. FLUX SECOND-ORDER (OURS) — apple-to-apple trên 3 prompt ImageReward
#    Chỉ ImageReward hỗ trợ second-order (cần supports_features)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 6. FLUX Second-Order (ours) ==="

for LABEL_PROMPT_DAMP in \
    "archaeologist|$ARCH_PROMPT|0.15" \
    "miner|$MINER_PROMPT|0.10" \
    "market|$MARKET_PROMPT|0.05"; do

    LABEL=$(echo "$LABEL_PROMPT_DAMP" | cut -d'|' -f1)
    PROMPT=$(echo "$LABEL_PROMPT_DAMP" | cut -d'|' -f2)
    DAMP=$(echo "$LABEL_PROMPT_DAMP" | cut -d'|' -f3)

    # 6a. Second-Order, GNS=50 (apple-to-apple with plugin GNS=50)
    run_flux "imagereward_${LABEL}" "2nd_order_gns50" \
        --reward imagereward --prompt "$PROMPT" --ir-prompt "$PROMPT" \
        --method second_order --gradient-norm-scale 50 $FLUX_COMMON

    # 6b. Second-Order, GNS=100
    run_flux "imagereward_${LABEL}" "2nd_order_gns100" \
        --reward imagereward --prompt "$PROMPT" --ir-prompt "$PROMPT" \
        --method second_order --gradient-norm-scale 100 $FLUX_COMMON

    # 6c. Second-Order, Unnormalized (automatic Woodbury damping)
    run_flux "imagereward_${LABEL}" "2nd_order_unnorm" \
        --reward imagereward --prompt "$PROMPT" --ir-prompt "$PROMPT" \
        --method second_order --gradient-norm-scale 0.0 $FLUX_COMMON

    # 6d. Second-Order + Bo4: sinh 80 ảnh, pick top 20
    run_flux "imagereward_${LABEL}" "2nd_order_bo4_raw" \
        --reward imagereward --prompt "$PROMPT" --ir-prompt "$PROMPT" \
        --method second_order --gradient-norm-scale 50 \
        --num-images 80 --num-steps 28 --height 512 --width 512 --cfg-scale 3.5 \
        --snr-factor 5 --num-guidance-steps 5 --guidance-start-step 1 --reward-scale 1

done

echo "  --- Đang chạy TẤT CẢ $(wc -l < flux_commands.txt | tr -d ' ') thí nghiệm FLUX song song trên các GPU ---"
python3 << 'PYEOF'
import subprocess, os, sys, threading
from concurrent.futures import ThreadPoolExecutor

def get_available_gpus():
    env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_gpus:
        # User specified GPUs like "6,7"
        return [g.strip() for g in env_gpus.split(",") if g.strip()]
    try:
        # Fallback: get all GPUs
        num = len(subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8').strip().split('\n'))
        return [str(i) for i in range(num)]
    except:
        return ["0"]

available_gpus = get_available_gpus()
num_gpus = len(available_gpus)
print(f"  [Dispatcher] Tìm thấy {num_gpus} GPU ({','.join(available_gpus)}). Bắt đầu phân bổ lệnh...")

with open('flux_commands.txt', 'r') as f:
    commands = [line.strip() for line in f if line.strip()]

gpu_lock = threading.Lock()

def run_cmd(cmd):
    with gpu_lock:
        gpu_id = available_gpus.pop(0)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"  [GPU {gpu_id}] Đang chạy: {cmd.split('--output-dir ')[-1]}")
    # Chạy ẩn output để console đỡ rối
    subprocess.run(cmd, shell=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    with gpu_lock:
        available_gpus.append(gpu_id)

with ThreadPoolExecutor(max_workers=num_gpus) as executor:
    executor.map(run_cmd, commands)
PYEOF

echo "  --- Hoàn thành chạy song song. Đang xử lý Bo4 ---"
# Pick top 20 cho Bo4 sau khi tất cả ảnh đã gen xong
for LABEL in "archaeologist" "miner" "market"; do
    python3 << PYEOF
import os, shutil, numpy as np
src = '../data/imagereward_${LABEL}/2nd_order_bo4_raw'
dst = '../data/imagereward_${LABEL}/2nd_order_bo4'
if not os.path.exists(src):
    exit(0)
os.makedirs(dst, exist_ok=True)
rewards = np.load(os.path.join(src, 'rewards.npy'))
top_idx = np.argsort(rewards)[-20:][::-1]
top_r = []
for rank, idx in enumerate(top_idx):
    s = os.path.join(src, f'{idx:04d}.png')
    d = os.path.join(dst, f'{rank:04d}.png')
    if os.path.exists(s): shutil.copy2(s, d)
    top_r.append(float(rewards[idx]))
np.save(os.path.join(dst, 'rewards.npy'), np.array(top_r))
meta = os.path.join(src, 'metadata.txt')
if os.path.exists(meta):
    shutil.copy2(meta, os.path.join(dst, 'metadata.txt'))
    with open(os.path.join(dst, 'metadata.txt'), 'a') as f:
        f.write(f'\n--- Bo4 ---\norig={len(rewards)} sel=20 mean_sel={np.mean(top_r):+.4f} mean_all={np.mean(rewards):+.4f}\n')
print(f'  Bo4 {src}: all={np.mean(rewards):+.4f} top20={np.mean(top_r):+.4f}')
PYEOF
done

cd ..

# ──────────────────────────────────────────────────────────────────────────────
# 7. TỔNG HỢP KẾT QUẢ
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 7. Tổng hợp kết quả ==="

# 7a. Bảng reward summary
python3 << 'PYEOF'
import os, json, numpy as np

summary = {}
for base in ['data', 'flux/results', 'checkerboard/results', 'mode_selection/results']:
    if not os.path.isdir(base):
        continue
    for root, dirs, files in os.walk(base):
        if 'rewards.npy' in files and 'bo4_raw' not in root:
            r = np.load(os.path.join(root, 'rewards.npy'))
            key = root.replace('\\', '/')
            summary[key] = {
                'mean': float(np.mean(r)), 'std': float(np.std(r)),
                'max': float(np.max(r)), 'min': float(np.min(r)),
                'n': int(len(r)),
            }

print(f'{"Condition":<70} {"N":>4} {"Mean":>8} {"Std":>8} {"Max":>8} {"Min":>8}')
print('=' * 110)
for k in sorted(summary):
    s = summary[k]
    print(f'{k:<70} {s["n"]:>4} {s["mean"]:>+8.4f} {s["std"]:>8.4f} {s["max"]:>+8.4f} {s["min"]:>+8.4f}')

with open('reward_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('\nSaved reward_summary.json')
PYEOF

# 7b. Export
rm -rf "$EXPORT_DIR"
mkdir -p "$EXPORT_DIR"

# Copy rewards.npy + metadata.txt cho mọi thí nghiệm (skip bo4_raw)
for base in data flux/results checkerboard/results mode_selection/results; do
    [ -d "$base" ] || continue
    find "$base" -name 'rewards.npy' -not -path '*bo4_raw*' | while read f; do
        d=$(dirname "$f")
        dest="$EXPORT_DIR/$d"
        mkdir -p "$dest"
        cp "$d"/rewards.npy "$dest/" 2>/dev/null || true
        cp "$d"/metadata.txt "$dest/" 2>/dev/null || true
    done
done

# Copy 4 ảnh mẫu mỗi condition (PNG → JPG 85%)
python3 << 'PYEOF'
import os
from PIL import Image

export_dir = 'export_results'
for base in ['data', 'flux/results']:
    if not os.path.isdir(base):
        continue
    for root, dirs, files in os.walk(base):
        if 'bo4_raw' in root:
            continue
        pngs = sorted([f for f in files if f.endswith('.png')])[:3]
        if not pngs:
            continue
        dest = os.path.join(export_dir, root)
        os.makedirs(dest, exist_ok=True)
        for f in pngs:
            img = Image.open(os.path.join(root, f)).convert('RGB')
            img.save(os.path.join(dest, f.replace('.png', '.jpg')), 'JPEG', quality=85)
PYEOF

# Copy figures (PDF/PNG)
for fig_dir in figures gaussian_mixture mode_selection checkerboard; do
    if [ -d "$fig_dir/figures" ]; then
        mkdir -p "$EXPORT_DIR/$fig_dir/figures"
        find "$fig_dir/figures" \( -name "*.pdf" -o -name "*.png" \) \
            -exec cp {} "$EXPORT_DIR/$fig_dir/figures/" \;
    fi
done

cp reward_summary.json "$EXPORT_DIR/"

# Nén
tar -czvf export_results.tar.gz "$EXPORT_DIR"/

SIZE=$(du -sm export_results.tar.gz | cut -f1)
echo ""
echo "================================================================="
echo "HOÀN TẤT!"
echo "   File: export_results.tar.gz ($SIZE MB)"
if [ "$SIZE" -gt 25 ]; then
    echo "    > 25MB — có thể cần giảm quality hoặc số ảnh"
else
    echo "   < 25MB"
fi
echo ""
echo "   Nội dung:"
echo "   - reward_summary.json"
echo "   - data/*/: 9 paper figures × tất cả conditions"
echo "   - data/imagereward_*/2nd_order_*: second-order (ours)"
echo "   - checkerboard/results/"
echo "   - figures/: rendered PDF"
echo "================================================================="
