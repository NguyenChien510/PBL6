# Hướng Dẫn Đóng Gói và Vận Hành Docker (MFGF TVPR)

Tài liệu hướng dẫn chi tiết cách build và chạy hệ thống **MFGF / DINOv2 Text-to-Video Person Retrieval & Tracking** bên trong môi trường Docker container.

---

## 1. Yêu Cầu Hệ Thống (Prerequisites)

- **Docker**: Phiên bản 20.10 trở lên (khuyên dùng Docker Desktop trên Windows).
- **Docker Compose**: Phiên bản v2 trở lên (đã tích hợp sẵn trong Docker Desktop).
- **GPU (Khuyên dùng)**: NVIDIA GPU có VRAM >= 4GB kèm theo **NVIDIA Container Toolkit** (được Docker Desktop hỗ trợ tự động trên Windows qua WSL2).
- **CPU**: Hệ thống vẫn hỗ trợ chạy hoàn toàn trên CPU (tốc độ xử lý sẽ chậm hơn GPU).

---

## 2. Kiến Trúc Đóng Gói (Docker Architecture)

- **Base Image**: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (Python 3.10 + PyTorch 2.5.1 + CUDA 12.1 + cuDNN 9).
- **Dependencies**: Cài đặt OpenCV, FFmpeg, PhoBERT v2, DINOv2, Gradio, PyVi tiếng Việt.
- **Tối ưu Layer Caching**: NLTK datasets và pip dependencies được nạp trước để build nhanh và chạy được offline.
- **Volume Mappings**:
  - `dataset/`: Mount trực tiếp tập video và chú thích từ máy tính (không bake 10GB+ vào image).
  - `src/checkpoints/`: Mount checkpoint đã train (`best.pth`, `latest.pth`) để nạp và lưu trữ ngoài container.
  - `src/outputs/`: Lưu video sau khi theo vết.
  - `tvpr_hf_cache` & `tvpr_torch_cache`: Giữ lại trọng lượng pretrained HuggingFace và PyTorch Hub, tránh tải lại sau khi tắt container.

---

## 3. Hướng Dẫn Khởi Động Nhanh (Quick Start)

### Cách 1: Sử dụng Docker Compose (Khuyên Dùng)

#### 1. Build Image:
```bash
docker compose build
```

#### 2. Khởi chạy Giao diện Web UI (GPU):
```bash
docker compose up
```
Sau khi khởi động, mở trình duyệt truy cập:
👉 **http://localhost:7860**

#### 3. Chạy Web UI ở chế độ chạy nền (Detached mode):
```bash
docker compose up -d
```

#### 4. Xem nhật ký (Logs):
```bash
docker compose logs -f
```

#### 5. Dừng container:
```bash
docker compose down
```

---

## 4. Các Lệnh Vận Hành Nâng Cao (CLI & Evaluation)

### 4.1. Theo vết đối tượng qua dòng lệnh (Tracking CLI)
Chạy theo vết một video cụ thể bằng câu mô tả tiếng Việt:
```bash
docker compose run --rm tvpr track --video dataset/videos/duke_person0001.mp4 --text "người phụ nữ mặc áo đen đeo túi xách" --top_k 1
```

Các tham số hỗ trợ:
- `--video` hoặc `-v`: Đường dẫn file video (.mp4).
- `--text` hoặc `-t`: Câu mô tả bằng tiếng Việt hoặc tiếng Anh.
- `--top_k` hoặc `-k`: Số lượng đối tượng theo vết (1 đến 5).
- `--output` hoặc `-o`: Đường dẫn file video xuất ra (mặc định lưu vào `src/outputs/`).
- `--max_frames`: Giới hạn số khung hình cần xử lý (ví dụ: `--max_frames 200`).

### 4.2. Đánh giá mô hình trên tập Test (Evaluation)
```bash
docker compose run --rm tvpr test
```

### 4.3. Truy cập vào Terminal bên trong Container (Interactive Bash)
```bash
docker compose run --rm tvpr bash
```

---

## 5. Dành Cho Máy Không Có GPU NVIDIA (CPU Mode)

Nếu máy tính của bạn không có card đồ họa rời NVIDIA hoặc chưa cài đặt driver CUDA:

Khởi chạy Web UI ở chế độ CPU:
```bash
docker compose --profile cpu run --rm --service-ports tvpr-cpu
```

---

## 6. Xây Dựng và Chạy Thủ Công (Native Docker CLI)

Nếu không sử dụng Docker Compose, bạn có thể build và run trực tiếp bằng Docker:

```bash
# Build image
docker build -t tvpr-mfgf:latest .

# Chạy Web UI với GPU
docker run --gpus all -it --rm \
  -p 7860:7860 \
  -v "%cd%/dataset:/app/dataset:ro" \
  -v "%cd%/src/checkpoints:/app/src/checkpoints" \
  -v "%cd%/src/outputs:/app/src/outputs" \
  tvpr-mfgf:latest web
```
*(Trên Linux/macOS, thay thế `%cd%` bằng `$(pwd)`)*

---

## 7. Cấu Trúc Các Tệp Đóng Gói Trong Dự Án

```
PBL6/
├── Dockerfile              # Cấu hình container image (CUDA 12.1 + PyTorch + Web UI)
├── docker-compose.yml      # Cấu hình dịch vụ, volume, GPU và port
├── .dockerignore           # Loại bỏ checkpoint nặng, dataset, cache khi build
├── entrypoint.sh           # Điều hướng linh hoạt giữa Web UI, Tracking CLI, Test, Train
├── DOCKER_README.md        # Hướng dẫn chi tiết sử dụng Docker
└── src/
    ├── web_app.py          # Giao diện Web Gradio phục vụ theo dõi trực tiếp
    ├── pipeline_tracking.py# Engine theo vết tích hợp CLI __main__
    ├── requirements.txt    # Danh sách thư viện Python
    └── ...
```
