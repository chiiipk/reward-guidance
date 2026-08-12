#!/bin/bash
set -e

# ==============================================================================
# project_command.sh — Chạy TOÀN BỘ thí nghiệm trong paper
#
# Bao gồm:
#   1. Cài đặt môi trường bằng uv + pyproject.toml
#   2. Gaussian mixture (CPU, vài giây)
#   3. Mode selection 1D (CPU, vài giây)
#   4. Checkerboard: train + sample + figure (1 GPU, ~60 phút train)
#   5. FLUX baselines: unguided + plugin + plugin+damp + plugin k=8
#   6. FLUX second-order (ours) — apple-to-apple với baselines
#   7. FLUX second-order + Bo4 (ours) — sinh 80 ảnh, pick top 20
#   8. Tổng hợp kết quả → export_results/ (< 25MB)
#
# NOTE: Đây là ablation trên 1 prompt (archaeologist) + 1 reward (ImageReward)
#       để so sánh apple-to-apple giữa second-order và baselines.
#       Không phải full reproduction (9 figures × nhiều prompt/reward).
#
# Cách chạy:
#   bash project_command.sh
#
# Yêu cầu: GPU ≥ 48GB (H200/A6000/L40S), Python 3.10+
# Cần đăng nhập HuggingFace trước: huggingface-cli login
# ==============================================================================

EXPORT_DIR="export_results"
COMMON_FLUX_ARGS="--num-images 20 --num-steps 28 --height 512 --width 512 --cfg-scale 3.5 --snr-factor 5 --num-guidance-steps 5 --guidance-start-step 1 --reward-scale 1"
FLUX_PROMPT="a young archaeologist gently brushing dust from an ancient ceramic vase, soft museum lighting, intricate details, cinematic composition"

# ──────────────────────────────────────────────────────────────────────────────
# 1. CÀI ĐẶT MÔI TRƯỜNG
# ──────────────────────────────────────────────────────────────────────────────
echo "=== 1. Tạo pyproject.toml và cài đặt môi trường ==="

cat << 'EOF' > pyproject.toml
[project]
name = "reward-guidance"
version = "0.1.0"
description = "Second-Order Reward Guidance — full paper reproduction"
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
    echo "Đang cài đặt uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml

# ──────────────────────────────────────────────────────────────────────────────
# 2. GAUSSIAN MIXTURE (CPU, ~10 giây)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 2. Gaussian mixture ==="
cd gaussian_mixture/
python make_grid_figures.py
python make_fmrg_figure.py
cd ..

# ──────────────────────────────────────────────────────────────────────────────
# 3. MODE SELECTION 1D (CPU, ~10 giây)
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
# 4. CHECKERBOARD: train + sample + figure (~60 phút train, vài phút sample)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 4. Checkerboard ==="
cd checkerboard/

# 4a. Train velocity field (skip nếu checkpoint đã tồn tại)
#     train.py default: --output-dir ./results → checkpoint tại results/velocity_net.pt
if [ ! -f "results/velocity_net.pt" ]; then
    echo "  Training checkerboard velocity field (500k steps)..."
    python train.py --num-steps 500000
else
    echo "  Checkpoint results/velocity_net.pt đã tồn tại, bỏ qua training."
fi

# 4b. Sample các conditions dùng trong paper (lambda=10)
echo "  Sampling: analytic tilt..."
python sample.py --analytic-tilt --lam 10.0
echo "  Sampling: plugin k=1..."
python sample.py --k 1 --lam 10.0 --num-samples 20000
echo "  Sampling: plugin k=8..."
python sample.py --k 8 --lam 10.0 --num-samples 5000
echo "  Sampling: plugin k=1 + damping..."
python sample.py --k 1 --lam 10.0 --sigma-damp 0.2 --num-samples 20000

# 4c. Render figures
echo "  Rendering figures..."
python make_main_figure.py
python plot.py --bon-vs-softmax --lam 10.0

cd ..

# ──────────────────────────────────────────────────────────────────────────────
# 5. FLUX BASELINES: unguided + plugin + plugin+damp + k=8 (ImageReward)
#    Apple-to-apple so sánh với second-order
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 5. FLUX baselines (ImageReward) ==="
cd flux/

# 5a. Unguided (reward-scale=0 overrides, no guidance at all)
echo "  [5a] Unguided..."
python sample.py \
    --reward imagereward \
    --prompt "$FLUX_PROMPT" \
    --ir-prompt "$FLUX_PROMPT" \
    --reward-scale 0 \
    --num-images 20 --num-steps 28 --height 512 --width 512 --cfg-scale 3.5 \
    --output-dir "./results/imagereward_unguided"

# 5b. Plugin k=1, GNS=50
echo "  [5b] Plugin k=1, GNS=50..."
python sample.py \
    --reward imagereward \
    --prompt "$FLUX_PROMPT" \
    --ir-prompt "$FLUX_PROMPT" \
    --gradient-norm-scale 50 \
    $COMMON_FLUX_ARGS \
    --output-dir "./results/imagereward_plugin_gns50"

# 5c. Plugin k=1, GNS=100
echo "  [5c] Plugin k=1, GNS=100..."
python sample.py \
    --reward imagereward \
    --prompt "$FLUX_PROMPT" \
    --ir-prompt "$FLUX_PROMPT" \
    --gradient-norm-scale 100 \
    $COMMON_FLUX_ARGS \
    --output-dir "./results/imagereward_plugin_gns100"

# 5d. Plugin k=1, GNS=100 + Damping (σ=0.15, same as paper Table for archaeologist)
echo "  [5d] Plugin k=1, GNS=100 + Damping 0.15..."
python sample.py \
    --reward imagereward \
    --prompt "$FLUX_PROMPT" \
    --ir-prompt "$FLUX_PROMPT" \
    --gradient-norm-scale 100 \
    --sigma-damp 0.15 \
    $COMMON_FLUX_ARGS \
    --output-dir "./results/imagereward_plugin_gns100_damp"

# 5e. Plugin k=8, GNS=50
echo "  [5e] Plugin k=8, GNS=50..."
python sample.py \
    --reward imagereward \
    --prompt "$FLUX_PROMPT" \
    --ir-prompt "$FLUX_PROMPT" \
    --gradient-norm-scale 50 \
    --num-particles 8 --lam 1.0 \
    $COMMON_FLUX_ARGS \
    --output-dir "./results/imagereward_plugin_k8_gns50"

# ──────────────────────────────────────────────────────────────────────────────
# 6. FLUX SECOND-ORDER (OURS) — same reward, same prompt, same hyperparams
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 6. FLUX Second-Order (ours) ==="

# 6a. Second-Order, GNS=50 (apple-to-apple with plugin GNS=50)
echo "  [6a] Second-Order, GNS=50..."
python sample.py \
    --reward imagereward \
    --prompt "$FLUX_PROMPT" \
    --ir-prompt "$FLUX_PROMPT" \
    --method second_order \
    --gradient-norm-scale 50 \
    $COMMON_FLUX_ARGS \
    --output-dir "./results/imagereward_2nd_order_gns50"

# 6b. Second-Order, GNS=100
echo "  [6b] Second-Order, GNS=100..."
python sample.py \
    --reward imagereward \
    --prompt "$FLUX_PROMPT" \
    --ir-prompt "$FLUX_PROMPT" \
    --method second_order \
    --gradient-norm-scale 100 \
    $COMMON_FLUX_ARGS \
    --output-dir "./results/imagereward_2nd_order_gns100"

# 6c. Second-Order, Unnormalized (automatic damping from Woodbury)
echo "  [6c] Second-Order, Unnormalized..."
python sample.py \
    --reward imagereward \
    --prompt "$FLUX_PROMPT" \
    --ir-prompt "$FLUX_PROMPT" \
    --method second_order \
    --gradient-norm-scale 0.0 \
    $COMMON_FLUX_ARGS \
    --output-dir "./results/imagereward_2nd_order_unnorm"

# ──────────────────────────────────────────────────────────────────────────────
# 7. FLUX SECOND-ORDER + Bo4 (OURS)
#    Sinh 80 ảnh, lấy top-20 theo reward (best-of-4)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 7. FLUX Second-Order + Bo4 ==="
echo "  [7] Second-Order + Bo4 (80 images, pick top 20)..."
python sample.py \
    --reward imagereward \
    --prompt "$FLUX_PROMPT" \
    --ir-prompt "$FLUX_PROMPT" \
    --method second_order \
    --gradient-norm-scale 50 \
    --num-images 80 --num-steps 28 --height 512 --width 512 --cfg-scale 3.5 \
    --snr-factor 5 --num-guidance-steps 5 --guidance-start-step 1 --reward-scale 1 \
    --output-dir "./results/imagereward_2nd_order_bo4_raw"

# Pick top-20 from 80 images based on saved rewards
echo "  Selecting top 20 images from 80..."
python3 -c "
import os, shutil, numpy as np

src = './results/imagereward_2nd_order_bo4_raw'
dst = './results/imagereward_2nd_order_bo4'
os.makedirs(dst, exist_ok=True)

rewards = np.load(os.path.join(src, 'rewards.npy'))
top_indices = np.argsort(rewards)[-20:][::-1]  # top 20, descending

top_rewards = []
for rank, idx in enumerate(top_indices):
    src_img = os.path.join(src, f'{idx:04d}.png')
    dst_img = os.path.join(dst, f'{rank:04d}.png')
    if os.path.exists(src_img):
        shutil.copy2(src_img, dst_img)
    top_rewards.append(float(rewards[idx]))

np.save(os.path.join(dst, 'rewards.npy'), np.array(top_rewards))

# Copy metadata
meta_src = os.path.join(src, 'metadata.txt')
if os.path.exists(meta_src):
    shutil.copy2(meta_src, os.path.join(dst, 'metadata.txt'))
    with open(os.path.join(dst, 'metadata.txt'), 'a') as f:
        f.write(f'\n--- Bo4 Selection ---\n')
        f.write(f'original_n:             80\n')
        f.write(f'selected_n:             20 (top by reward)\n')
        f.write(f'mean_selected_reward:   {np.mean(top_rewards):+.4f}\n')
        f.write(f'mean_all_reward:        {np.mean(rewards):+.4f}\n')

print(f'Bo4 selection: {len(top_rewards)} images')
print(f'  Mean reward (all 80):   {np.mean(rewards):+.4f}')
print(f'  Mean reward (top 20):   {np.mean(top_rewards):+.4f}')
print(f'  Max reward:             {np.max(top_rewards):+.4f}')
"

cd ..

# ──────────────────────────────────────────────────────────────────────────────
# 8. TỔNG HỢP KẾT QUẢ
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 8. Tổng hợp kết quả ==="

# 8a. Tạo bảng reward summary
python3 << 'PYEOF'
import os, json, numpy as np

summary = {}
for exp_dir in ['flux/results', 'checkerboard/results', 'mode_selection/results']:
    if not os.path.isdir(exp_dir):
        continue
    for sub in sorted(os.listdir(exp_dir)):
        reward_path = os.path.join(exp_dir, sub, 'rewards.npy')
        if os.path.isfile(reward_path):
            r = np.load(reward_path)
            summary[f'{exp_dir}/{sub}'] = {
                'mean': float(np.mean(r)),
                'std': float(np.std(r)),
                'max': float(np.max(r)),
                'min': float(np.min(r)),
                'n': int(len(r)),
            }

# Print table
print(f'{"Condition":<60} {"N":>4} {"Mean":>8} {"Std":>8} {"Max":>8} {"Min":>8}')
print('=' * 100)
for k in sorted(summary):
    s = summary[k]
    print(f'{k:<60} {s["n"]:>4} {s["mean"]:>+8.4f} {s["std"]:>8.4f} {s["max"]:>+8.4f} {s["min"]:>+8.4f}')

# Save JSON
with open('reward_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print()
print('Saved reward_summary.json')
PYEOF

# 8b. Export kết quả
rm -rf "$EXPORT_DIR"
mkdir -p "$EXPORT_DIR"

# Copy rewards.npy + metadata.txt (rất nhẹ) cho mọi thí nghiệm
for results_dir in flux/results checkerboard/results mode_selection/results; do
    if [ ! -d "$results_dir" ]; then continue; fi
    for sub in "$results_dir"/*/; do
        [ -d "$sub" ] || continue
        # Skip the raw Bo4 dir (80 images, only keep the selected one)
        case "$sub" in *bo4_raw*) continue ;; esac
        dest="$EXPORT_DIR/$sub"
        mkdir -p "$dest"
        # Copy file nhẹ: rewards, metadata, csv, npy
        find "$sub" -maxdepth 1 \( -name "*.npy" -o -name "*.txt" -o -name "*.csv" -o -name "*.json" \) \
            -exec cp {} "$dest/" \;
    done
done

# Copy ảnh FLUX (convert PNG → JPG 85% để giữ dung lượng nhỏ)
# Chỉ lấy 4 ảnh đầu tiên mỗi condition để tiết kiệm
python3 << 'PYEOF'
import os
from PIL import Image

export_dir = os.environ.get('EXPORT_DIR', 'export_results')
for results_dir in ['flux/results']:
    if not os.path.isdir(results_dir):
        continue
    for sub in sorted(os.listdir(results_dir)):
        if 'bo4_raw' in sub:
            continue
        sub_path = os.path.join(results_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        dest = os.path.join(export_dir, results_dir, sub)
        os.makedirs(dest, exist_ok=True)
        pngs = sorted([f for f in os.listdir(sub_path) if f.endswith('.png')])[:4]
        for f in pngs:
            img = Image.open(os.path.join(sub_path, f)).convert('RGB')
            img.save(os.path.join(dest, f.replace('.png', '.jpg')), 'JPEG', quality=85)
            print(f'  Exported: {sub}/{f} -> jpg')
PYEOF

# Copy figures (PDF rất nhẹ)
for fig_dir in figures gaussian_mixture mode_selection checkerboard; do
    if [ -d "$fig_dir/figures" ]; then
        mkdir -p "$EXPORT_DIR/$fig_dir/figures"
        find "$fig_dir/figures" -name "*.pdf" -exec cp {} "$EXPORT_DIR/$fig_dir/figures/" \;
        find "$fig_dir/figures" -name "*.png" -exec cp {} "$EXPORT_DIR/$fig_dir/figures/" \;
    fi
done

# Copy reward summary
cp reward_summary.json "$EXPORT_DIR/"

# 8c. Nén
export EXPORT_DIR
tar -czvf export_results.tar.gz "$EXPORT_DIR"/

# 8d. Kiểm tra dung lượng
SIZE=$(du -sm export_results.tar.gz | cut -f1)
echo ""
echo "================================================================="
echo "✅ HOÀN TẤT!"
echo "   File kết quả: export_results.tar.gz ($SIZE MB)"
if [ "$SIZE" -gt 25 ]; then
    echo "   ⚠️  Dung lượng > 25MB. Giảm số ảnh export hoặc chất lượng JPG."
else
    echo "   ✅ Dung lượng OK (< 25MB)"
fi
echo ""
echo "   Nội dung:"
echo "   - reward_summary.json: bảng tổng hợp mean/std/max reward mọi condition"
echo "   - flux/results/*/: rewards.npy + metadata.txt + 4 ảnh sample (JPG)"
echo "   - checkerboard/results/: sampling outputs"
echo "   - figures/: PDF figures cho Gaussian mixture, mode selection, checkerboard"
echo ""
echo "   👉 Kéo file export_results.tar.gz về máy để phân tích!"
echo "================================================================="
