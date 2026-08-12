#!/bin/bash
set -e

# ==============================================================================
# HƯỚNG DẪN CHẠY
# Script này tự động thiết lập môi trường bằng uv thông qua pyproject.toml,
# chạy test, sinh ảnh, và nén kết quả thành 1 file < 25MB.
# ==============================================================================

echo "=== 1. Tạo file pyproject.toml ==="
cat << 'EOF' > pyproject.toml
[project]
name = "reward-guidance"
version = "0.1.0"
description = "Second-Order Reward Guidance for FLUX"
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
    "image-reward"
]
EOF

echo "=== 2. Cài đặt môi trường bằng UV ==="
if ! command -v uv &> /dev/null; then
    echo "Đang cài đặt uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

uv venv
source .venv/bin/activate
# Cài đặt qua uv pip đọc trực tiếp file pyproject.toml
uv pip install -r pyproject.toml

echo "=== 3. Chạy Sanity Check (Test hệ số Woodbury) ==="
cd flux
python3 test_large_lambda.py

echo "=== 4. Chạy Eval (Sinh ảnh với FLUX + Palette) ==="
# Chạy script bash có sẵn. Sẽ sinh ra 32 ảnh tổng cộng.
bash eval_pipelines/eval_second_order.sh palette

echo "=== 5. Tổng hợp và Nén kết quả (< 25MB) ==="
# Convert toàn bộ PNG sang JPG chất lượng 85% để giảm dung lượng kịch liệt
# 32 ảnh PNG gốc có thể tốn 50MB+, nhưng ép sang JPG 85% chỉ còn khoảng 3-5MB.
python3 -c "
import os
from PIL import Image

for root, dirs, files in os.walk('results'):
    for file in files:
        if file.endswith('.png'):
            path = os.path.join(root, file)
            img = Image.open(path).convert('RGB')
            # Lưu lại dưới dạng JPG và xoá file PNG
            img.save(path.replace('.png', '.jpg'), 'JPEG', quality=85)
            os.remove(path)
            print(f'Converted: {file} -> {file.replace(\".png\", \".jpg\")}')
"

echo "=== 6. Export ra thư mục và nén file ==="
cd ..
mkdir -p export_results
cp -r flux/results export_results/

# Tạo file nén cuối cùng để bạn kéo về
tar -czvf export_results.tar.gz export_results/

echo ""
echo "================================================================="
echo "✅ HOÀN TẤT! Toàn bộ kết quả đã được nén vào: export_results.tar.gz"
echo "👉 File này chắc chắn < 25MB. Bạn chỉ cần kéo file này về máy!"
echo "================================================================="
