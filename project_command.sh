#!/bin/bash
set -Eeuo pipefail

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
FLUX_COMMON="--num-steps 28 --height 512 --width 512 --cfg-scale 3.5 --snr-factor 5 --num-guidance-steps 5 --guidance-start-step 1 --num-images 20"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLUX_COMMAND_FILE="$REPO_ROOT/flux/flux_commands.txt"
export FLUX_COMMAND_FILE
cd "$REPO_ROOT"

# Mảng chứa các lệnh chạy FLUX để sau đó phân bổ song song cho các GPU
: > "$FLUX_COMMAND_FILE"
trap 'rm -f -- "$FLUX_COMMAND_FILE"' EXIT

# Helper: lưu 1 condition cho 1 figure vào file thay vì chạy ngay
run_flux() {
    local FIGURE=$1; shift
    local CONDITION=$1; shift
    local OUTDIR="../data/${FIGURE}/${CONDITION}"
    local CMD=(python sample.py "$@" --output-dir "$OUTDIR")
    # Preserve prompt/question arguments containing spaces or apostrophes.
    printf '%q ' "${CMD[@]}" >> "$FLUX_COMMAND_FILE"
    printf '\n' >> "$FLUX_COMMAND_FILE"
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. CÀI ĐẶT MÔI TRƯỜNG
# ──────────────────────────────────────────────────────────────────────────────
echo "=== 1. Cài đặt môi trường ==="
if ! command -v uv >/dev/null 2>&1; then
    echo "Không tìm thấy uv." >&2
    echo "Cài uv tại https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi
uv sync
source .venv/bin/activate
python -c "import diffusers, torch, transformers"

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
    --record-trajectories --output-dir results
python sample.py --reward gaussian --num-samples 2000 --max-n 16 --lam 5.0 \
    --record-trajectories --output-dir results
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
python plot.py --plots bon_paper --lam 10.0
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

# ──────────────────────────────────────────────────────────────────────────────
# 6. FLUX SECOND-ORDER (OURS) — apple-to-apple trên 3 prompt ImageReward
#    Chạy ImageReward, một trong các reward hỗ trợ feature-space Hessian.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 6. FLUX Second-Order (ours) ==="

for LABEL_PROMPT in \
    "archaeologist|$ARCH_PROMPT" \
    "miner|$MINER_PROMPT" \
    "market|$MARKET_PROMPT"; do

    IFS='|' read -r LABEL PROMPT <<< "$LABEL_PROMPT"

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

echo "  --- Đang chạy TẤT CẢ $(wc -l < "$FLUX_COMMAND_FILE" | tr -d ' ') thí nghiệm FLUX song song trên các GPU ---"
python3 << 'PYEOF'
import os
import queue
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def get_available_gpus():
    # Chỉ sử dụng GPU 6 và 7
    return ["6", "7"]

available_gpus = get_available_gpus()
num_gpus = len(available_gpus)
if not available_gpus:
    sys.exit("[Dispatcher] Không tìm thấy GPU khả dụng.")
print(f"  [Dispatcher] Tìm thấy {num_gpus} GPU ({','.join(available_gpus)}). Bắt đầu phân bổ lệnh...")

with open(os.environ["FLUX_COMMAND_FILE"], "r") as f:
    commands = [line.strip() for line in f if line.strip()]

gpu_queue = queue.Queue()
for gpu in available_gpus:
    gpu_queue.put(gpu)

log_dir = Path("logs/flux")
log_dir.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd):
    argv = shlex.split(cmd)
    try:
        outdir = argv[argv.index("--output-dir") + 1]
    except (ValueError, IndexError):
        outdir = f"job_{abs(hash(cmd))}"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", outdir).strip("_")
    log_path = log_dir / f"{slug}.log"
    gpu_id = gpu_queue.get()
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"  [GPU {gpu_id}] Đang chạy: {outdir} (log: {log_path})")
        with log_path.open("w") as log_file:
            proc = subprocess.run(
                argv, env=env, stdout=log_file, stderr=subprocess.STDOUT,
                check=False,
            )
        return proc.returncode, outdir, log_path
    finally:
        gpu_queue.put(gpu_id)

with ThreadPoolExecutor(max_workers=num_gpus) as executor:
    results = list(executor.map(run_cmd, commands))

failed = [result for result in results if result[0] != 0]
if failed:
    print(f"\n[Dispatcher] {len(failed)}/{len(results)} lệnh thất bại:", file=sys.stderr)
    for returncode, outdir, log_path in failed:
        print(f"  rc={returncode}  {outdir}  log={log_path}", file=sys.stderr)
        try:
            tail = log_path.read_text(errors="replace").splitlines()[-20:]
            print("\n".join(f"    {line}" for line in tail), file=sys.stderr)
        except OSError:
            pass
    sys.exit(1)
PYEOF

echo "  --- Hoàn thành chạy song song. Đang xử lý Bo4 ---"
# Pick top 20 cho Bo4 sau khi tất cả ảnh đã gen xong
for LABEL in "archaeologist" "miner" "market"; do
python3 << PYEOF
from pathlib import Path
import shutil
import numpy as np

condition = Path('../data/imagereward_${LABEL}/2nd_order_bo4_raw')
run_dirs = [p.parent for p in condition.rglob('rewards.npy')]
if len(run_dirs) != 1:
    raise RuntimeError(f'Expected one completed Bo4 run under {condition}, found {len(run_dirs)}')
src = run_dirs[0]
dst = Path('../data/imagereward_${LABEL}/2nd_order_bo4') / src.name
dst.mkdir(parents=True, exist_ok=True)
rewards = np.load(src / 'rewards.npy')
top_idx = np.argsort(rewards)[-20:][::-1]
top_r = []
for rank, idx in enumerate(top_idx):
    s = src / f'{idx:04d}.png'
    d = dst / f'{rank:04d}.png'
    if not s.exists():
        raise FileNotFoundError(s)
    shutil.copy2(s, d)
    top_r.append(float(rewards[idx]))
np.save(dst / 'rewards.npy', np.array(top_r))
meta = src / 'metadata.txt'
if meta.exists():
    shutil.copy2(meta, dst / 'metadata.txt')
    with (dst / 'metadata.txt').open('a') as f:
        f.write(f'\n--- Bo4 ---\norig={len(rewards)} sel=20 mean_sel={np.mean(top_r):+.4f} mean_all={np.mean(rewards):+.4f}\n')
print(f'  Bo4 {src}: all={np.mean(rewards):+.4f} top20={np.mean(top_r):+.4f}')
PYEOF
done

# Render only after every condition has completed successfully.
echo "  --- Rendering all figures ---"
for fig in ../figures/*/; do
    if [ -f "$fig/regenerate.py" ]; then
        echo "    Rendering $(basename "$fig")..."
        (cd "$fig" && python regenerate.py) || echo "    WARNING: $(basename "$fig") failed"
    fi
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
mkdir -p "$EXPORT_DIR"
find figures gaussian_mixture mode_selection checkerboard -type f \( -name "*.pdf" -o -name "*.png" \) -print0 2>/dev/null | tar -cf - --null -T - | tar -xf - -C "$EXPORT_DIR"

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
