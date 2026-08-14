# Hướng Dẫn Chạy Second-Order Guidance trên H200

Tài liệu này chứa các hướng dẫn chi tiết để thiết lập môi trường và chạy thử nghiệm **Second-Order Guidance** trên cụm H200. Xin vui lòng làm theo từng bước để đảm bảo cấu hình phần cứng và toán học hoạt động chính xác.

> **Lưu ý về dấu trong công thức:** với target
> `p(x) ∝ p_data(x) exp(+β r(x))`, khai triển bậc hai cho
> `Σ = (σ⁻²I - βH)⁻¹` và mean `μ + βΣ∇r`. Implementation dùng dạng đã sửa
> `(I - cH)⁻¹g`; hai dấu `+H`/`-∇r` trong bản PDF ý tưởng tương ứng với
> `exp(-βr)`, không phải reward maximization.

## 1. Thiết lập môi trường

Chạy từ **root của repository**. Môi trường đã được cố định ở Python 3.11,
PyTorch 2.8.0 và CUDA 12.8:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # bỏ qua nếu đã có uv
uv sync --frozen
source .venv/bin/activate
```

Không dùng `pip install -r requirements.txt`: dependency được quản lý tập trung
trong `pyproject.toml`, còn `uv.lock` cố định phiên bản giữa máy local và H200.
Không bỏ `--frozen` trên máy chạy thí nghiệm; nếu lockfile không khớp project,
hãy cập nhật và review lockfile ở máy phát triển trước.

Đăng nhập Hugging Face trước khi tải FLUX.1-dev:

```bash
huggingface-cli login
```

Kiểm tra nhanh CLI và toán second-order trước khi tải model:

```bash
python flux/h200_preflight.py
python flux/sample.py --help
cd flux && python smoke_test_second_order.py && cd ..
```

`h200_preflight.py` kiểm tra Python/PyTorch/CUDA, chạy một BF16 backward và ép
Triton biên dịch helper liên kết với `libcuda.so.1`. Chỉ chạy batch đầy đủ sau
khi dòng cuối là `H200 preflight: PASS`.

Smoke test end-to-end tối thiểu (một ảnh 256×256) có thể chạy bằng. Đặt
`H200_GPU` nếu tiến trình chưa được scheduler giới hạn sẵn GPU:

```bash
H200_GPU=6 bash flux/eval_pipelines/smoke_h200.sh
```

Khi chạy toàn bộ project, mặc định FLUX dùng GPU 6 và 7. Có thể đổi danh sách;
script sẽ preflight từng H200 trước khi bắt đầu:

```bash
FLUX_GPUS=6,7 bash project_command.sh
```

## 2. Generate thử một ảnh trên H200

Chạy một ảnh với reward `palette` để kiểm tra trọn pipeline mà chưa phải tải
ImageReward. Script có thể được gọi từ bất kỳ thư mục nào:

```bash
NUM_IMAGES=1 bash flux/eval_pipelines/eval_second_order.sh palette
```

Lệnh trên chạy hai condition (normalized và unnormalized). Nếu chỉ muốn một
condition duy nhất:

```bash
cd flux
python sample.py \
  --reward palette --palette cool_ocean \
  --prompt "A small sailboat on a calm ocean at sunrise" \
  --method second_order --gradient-norm-scale 10 \
  --num-images 1 --num-steps 28 --height 512 --width 512 \
  --verbose --output-dir ./results/h200_smoke
```

**Trong quá trình chạy, hãy chú ý Log:**
Khi thanh tiến trình (progress bar) bắt đầu chạy các bước ODE, log sẽ in ra liên tục dòng giám sát hệ số:
```text
k_eig=16 W_cols=16 correction_strength=...
```

Xin hãy **chụp màn hình hoặc copy 3-5 dòng đầu tiên** chứa dòng log này và gửi lại cho chúng tôi. 
- `W_cols` bắt buộc phải là số $\le 16$ (để đảm bảo không bị quá tải VJP).
- `correction_strength` cho biết độ mạnh của hiệu chỉnh bậc hai.

## 3. Chạy ImageReward sau khi smoke test pass

```bash
NUM_IMAGES=1 bash flux/eval_pipelines/eval_second_order.sh \
  imagereward "A cute fluffy cat"
```

## 4. Thu thập kết quả

Sau khi lệnh bash hoàn tất, mã nguồn sẽ sinh ra 2 thư mục kết quả:
1. `flux/results/second_order_norm10/` (chạy với normalization)
2. `flux/results/second_order_unnorm/` (chạy không normalize)

Xin vui lòng nén 2 thư mục này cùng với file log terminal và gửi lại cho chúng tôi để tiến hành phân tích chất lượng ảnh và tính ổn định.

```bash
tar -czvf results_h200.tar.gz \
  flux/results/second_order_norm10/ flux/results/second_order_unnorm/
```

Cảm ơn bạn đã hỗ trợ chạy thí nghiệm!
