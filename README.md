# DINOv2-TVPR: Text-to-Video Person Retrieval & Continuous Person Tracking

Hệ thống truy vấn và theo vết đối tượng người liên tục trong video bằng văn bản mô tả tiếng Việt (**Text-to-Video Person Retrieval & Tracking - TVPR**).
Dự án kết hợp mô hình thị giác nền tảng **Meta DINOv2 (Foundation Vision)** cùng mô hình ngôn ngữ tiếng Việt chuyên sâu **VinAI PhoBERT v2**, tích hợp bộ theo vết liên tục đa đối tượng (**Robust Continuous Multi-Person Tracker**).

---

## 1. Điểm Nổi Bật & Cải Tiến Kiến Trúc

1. **Thị giác nền tảng Meta DINOv2 (Vision Backbone)**:
   - Sử dụng pre-trained Meta DINOv2 (`facebook/dinov2-small`, 384 chiều đặc trưng, 21M tham số).
   - Cơ chế đóng băng chọn lọc: đóng băng các tầng trích xuất đặc trưng cơ bản và fine-tune 2 tầng Transformer cuối cùng để thích ứng sâu với đặc điểm trang phục và diện mạo người.

2. **Cơ chế tổng hợp thời gian Temporal Attention Aggregator**:
   - Sử dụng Multi-Head Self-Attention với Temporal Positional Embeddings để nắm bắt mối tương quan động giữa các khung hình (frames) theo trục thời gian, thay thế hoàn toàn các mô hình 3D CNN nặng nề (S3D).

3. **Ngôn ngữ tiếng Việt chuyên sâu VinAI PhoBERT v2 (Text Backbone)**:
   - Tích hợp VinAI PhoBERT v2 (`vinai/phobert-base-v2`) kết hợp công cụ tách từ tiếng Việt tự động (`pyvi`).
   - Tối ưu hóa biểu diễn ngữ nghĩa của các câu mô tả tiếng Việt phức tạp (màu sắc quần áo, phụ kiện, dáng điệu, giới tính).

4. **Không gian biểu diễn chung (Common Embedding Space & InfoNCE Loss)**:
   - Chiếu đặc trưng hình ảnh và văn bản vào không gian chung 256 chiều chuẩn hóa L2.
   - Sử dụng hàm mất mát đối chiếu hai chiều đối xứng (Symmetric InfoNCE Contrastive Loss) với tham số nhiệt độ học được (learnable temperature) theo phong cách CLIP.

5. **Bộ theo vết người liên tục (Robust Continuous Person Tracker)**:
   - Cơ chế quản lý vòng đời định danh đối tượng (SORT / ByteTrack style) với khả năng ghi nhớ qua các pha che khuất (occlusion) và chập chờn phát hiện (lên đến 30 frames).
   - Nội suy quỹ đạo mượt mà và thuật toán phát hiện đối tượng rời khỏi góc máy quay (Out-of-Camera Detection).
   - Hỗ trợ linh hoạt các backend: YOLOv8, Torchvision Faster R-CNN MobileNetV3-FPN, hoặc OpenCV.

6. **Tương thích ngược với kiến trúc MFGF (ViT + S3D + Guided Tips Learning)**:
   - Hệ thống hỗ trợ chuyển đổi linh hoạt giữa kiến trúc DINOv2 và MFGF gốc thông qua file cấu hình `configs/default.yaml`.

7. **Chỉ số đánh giá độ ổn định**:
   - Rank@1, Rank@5, Rank@10, Rank@50, mAP, MRR.
   - Median Rank (MdR): thước đo độ ổn định hệ thống trong môi trường an ninh và giám sát thực tế.

---

## 2. Cấu Trúc Dự Án

```
PBL6/
├── configs/
│   └── default.yaml          # File cấu hình: DINOv2, PhoBERT v2, learning rates, frames, batch size
├── data/
│   ├── dataset.py            # DataLoader hỗ trợ định dạng JSON và video
│   └── transforms.py         # Biến đổi không gian - thời gian cho khung hình video
├── models/
│   ├── dinov2_tvpr.py        # Kiến trúc chính: DINOv2 + Temporal Aggregator + PhoBERT v2
│   ├── mfgf_main.py          # Kiến trúc MFGF ViT + S3D (tương thích ngược)
│   ├── visual_encoder.py     # ViT Spatio-Temporal Encoder
│   ├── motion_encoder.py     # S3D Motion Separable Convolutions
│   ├── text_encoder.py       # BERT/PhoBERT Feature Projector
│   └── spaces.py             # Feature Aggregator & Projection Heads
├── loss/
│   └── criterion.py          # InfoNCE Contrastive Loss, D2 Distillation Loss
├── utils/
│   ├── metrics.py            # Rank@K, Median Rank (MdR), MRR, mAP
│   └── paths.py              # Centralized Path Management (tự động nhận diện đường dẫn dự án)
├── train.py                  # Huấn luyện mô hình với Warm-up & Cosine Annealing LR
├── test.py                   # Đánh giá benchmark trên tập test & demo truy vấn đơn
├── app.py                    # Giao diện Desktop GUI (Tkinter) với video player tích hợp
├── web_app.py                # Giao diện Web UI hiện đại (Gradio) phục vụ chạy cục bộ & Docker
├── pipeline_tracking.py      # Engine theo vết đối tượng liên tục & Text-to-Video Matching (hỗ trợ CLI)
├── requirements.txt          # Danh sách các thư viện phụ thuộc
└── README.md                 # Tài liệu hướng dẫn sử dụng
```

---

## 3. Cấu Hình Đường Dẫn Tập Trung (Centralized Path Management)

Dự án sử dụng cơ chế quản lý đường dẫn tập trung tại **`src/utils/paths.py`**:
* **Tự động nhận diện** toàn bộ thư mục dữ liệu `dataset/` (hoặc `data/`), `checkpoints/`, `configs/`, `outputs/`, `reports/` mà không cần hardcode đường dẫn tuyệt đối.
* Nếu muốn đổi đường dẫn dữ liệu `data_root`, bạn chỉ cần chỉnh sửa **duy nhất 1 dòng** trong `src/configs/default.yaml` (`data.data_root`). Mặc định là `"auto"` để tự động tìm kiếm.
* Tất cả các lệnh có thể chạy trực tiếp từ thư mục gốc dự án (`D:\PBL6`) hoặc từ thư mục `D:\PBL6\src`.

---

## 4. Hướng Dẫn Chạy Thử Nghiệm

Kích hoạt môi trường conda trước khi chạy:
```bash
conda activate tvpr
```

### 4.1. Giao diện Desktop GUI (Continuous Person Tracking)
```bash
python src/app.py
# hoặc nếu đang ở thư mục src:
python app.py
```

### 4.2. Giao diện Web UI (Gradio trên trình duyệt)
```bash
python src/web_app.py
# Mở trình duyệt tại: http://localhost:7860
```

### 4.3. Theo vết đối tượng qua dòng lệnh (CLI Tracking)
```bash
python src/pipeline_tracking.py --video dataset/videos/duke_person0001.mp4 --text "người phụ nữ mặc áo đen mang túi xách" --top_k 1
```

### 4.4. Huấn luyện (Training)
```bash
python src/train.py
# hoặc chỉ định file cấu hình:
python src/train.py --config src/configs/default.yaml
```

### 4.5. Đánh giá trên Test Set (Evaluation)
```bash
python src/test.py
# hoặc chỉ định checkpoint cụ thể:
python src/test.py --checkpoint src/checkpoints/best.pth
```

### 4.6. Truy vấn Trực quan (Inference Demo)
```bash
python src/test.py --query "Người phụ nữ mặc áo khoác trắng và quần xanh dương."
```

---

## 5. Chạy Với Docker (Container Deployment)

Chi tiết xem tại tài liệu [DOCKER_README.md](../DOCKER_README.md).

```bash
# 1. Build image
docker compose build

# 2. Khởi chạy Web UI (GPU) tại http://localhost:7860
docker compose up

# 3. Chạy theo vết CLI bên trong Docker
docker compose run --rm tvpr track --video dataset/videos/duke_person0001.mp4 --text "người phụ nữ mặc áo đen mang túi xách" --top_k 1
```
