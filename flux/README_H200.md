# Hướng Dẫn Chạy Second-Order Guidance trên H200

Tài liệu này chứa các hướng dẫn chi tiết để thiết lập môi trường và chạy thử nghiệm **Second-Order Guidance** trên cụm H200. Xin vui lòng làm theo từng bước để đảm bảo cấu hình phần cứng và toán học hoạt động chính xác.

## 1. Thiết lập Môi trường

Thư mục `flux/` chứa mã nguồn đã được cập nhật mới nhất. Đầu tiên, hãy tạo một virtual environment và cài đặt các dependencies (yêu cầu Python 3.10+):

```bash
cd flux
python3 -m venv venv
source venv/bin/activate

# Cài đặt các thư viện cơ bản
pip install -r requirements.txt
pip install torch transformers accelerate diffusers
```

*(Lưu ý: Nếu cụm H200 đã có sẵn môi trường PyTorch + Diffusers, bạn có thể bỏ qua bước cài đặt hoặc chỉ cài thêm các thư viện còn thiếu).*

## 2. Chạy Đánh giá (Evaluation) Thực Tế

Sau khi môi trường đã sẵn sàng, hãy tiến hành chạy luồng sinh ảnh với reward là `palette` (hàm đánh giá màu sắc cơ bản). Script này sẽ tự động tải model FLUX và chạy sinh ảnh.

```bash
bash eval_pipelines/eval_second_order.sh palette
```

**Trong quá trình chạy, hãy chú ý Log:**
Khi thanh tiến trình (progress bar) bắt đầu chạy các bước ODE, log sẽ in ra liên tục dòng giám sát hệ số:
```text
k_eig=16  W cols=16  c*||W||^2*|mu|=...
```

Xin hãy **chụp màn hình hoặc copy 3-5 dòng đầu tiên** chứa dòng log này và gửi lại cho chúng tôi. 
- `W cols` bắt buộc phải là số $\le 16$ (để đảm bảo không bị quá tải VJP).
- Giá trị `c*||W||^2*|mu|` sẽ quyết định sức mạnh của hiệu chỉnh bậc hai.

## 3. Thu thập Kết quả

Sau khi lệnh bash hoàn tất, mã nguồn sẽ sinh ra 2 thư mục kết quả:
1. `results/second_order_norm10/` (Chạy với Normalization)
2. `results/second_order_unnorm/` (Chạy tự do, thể hiện tính chất damping)

Xin vui lòng nén 2 thư mục này cùng với file log terminal và gửi lại cho chúng tôi để tiến hành phân tích chất lượng ảnh và tính ổn định.

```bash
tar -czvf results_h200.tar.gz results/second_order_norm10/ results/second_order_unnorm/
```

Cảm ơn bạn đã hỗ trợ chạy thí nghiệm!
