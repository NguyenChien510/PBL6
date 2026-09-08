# MFGF TVPR: Multielement Feature Guided Fragments Learning for Text-to-Video Person Retrieval

Triển khai hoàn chỉnh kiến trúc mô hình **MFGF (Multielement Feature Guided Fragments Learning)** bằng PyTorch cho bài toán **Text-to-Video Person Retrieval (TVPR)**.

---

## 1. Điểm nổi bật & Cải tiến Kiến trúc

1. **Khắc phục hiện tượng che khuất (Occlusion) & Đứt gãy thông tin tĩnh**:
   - Tích hợp chuỗi không-thời gian (spatio-temporal) từ $L_1=4$ khung hình ngẫu nhiên (Visual ViT) và $L_2=16$ khung hình liên tiếp (Motion S3D).
2. **Text Prompter & Guided Tips Learning ($f_{tips} = W_h \odot f^O_{tips}$)**:
   - Dùng NLTK lọc các mảnh ngữ nghĩa tinh khiết: Nouns, Verbs, Adjectives.
   - Trọng số không huấn luyện: $W = c_n / C$ (tần suất từ khóa trong tập dữ liệu).
3. **Modified Visual Encoder (ViT)**:
   - Patch size $16 \times 16$, độ phân giải $224 \times 224$.
   - Cộng gộp Spatial Embeddings ($E_S$) và Temporal Embeddings ($E_T$):
     $$e_{l, n} = e^O_{l, n} + E_{S_n} + E_{T_l}$$
4. **Motion Encoder (S3D)**:
   - Separable Convolutions: tách 3D Conv thành 2D Spatial $[1, k, k]$ và 1D Temporal $[k, 1, 1]$.
   - Inception blocks đa quy mô và cơ chế Gating:
     $$y_i = \text{sigmoid}(\sigma \cdot y^p_i + b) \odot y^p_i$$
5. **Feature Aggregator & Dual-Distilled ($D^2$) Space**:
   - Feature Aggregator (FC + LayerNorm + GELU) khử thông tin dư thừa (information redundancy).
   - FeatureConvertor (`Linear -> BatchNorm1d -> Sigmoid`) ánh xạ vào không gian $D_{tips}$.
6. **Hàm mất mát và cơ chế Blending Alpha ($\alpha$)**:
   $$\mathcal{L}_{MFGF} = \alpha \mathcal{L}_{common} + (1 - \alpha) \mathcal{L}_{D2}$$
   - $\mathcal{L}_{common}$: InfoNCE Contrastive Softmax Loss ($\theta = 0.05$).
   - $\mathcal{L}_{D2} = \mathcal{L}(T^{D2}, V^{D2}) + B \times [\text{BCE}(Tips, T^{D2}) + \text{BCE}(Tips, V^{D2})]$ với hệ số quy mô $B$ (Batch Size multiplier).
   - $\alpha$ là tham số học được khởi tạo trong khoảng $0.1 \sim 0.2$ ($\alpha_0 = 0.15$).
7. **Chỉ số đánh giá độ ổn định**:
   - Rank@1, Rank@5, Rank@10, Rank@50.
   - **Median Rank (MdR)**: thước đo độ ổn định hệ thống trong môi trường an ninh thực tế.

---

## 2. Cấu trúc Dự án

```
MFGF_TVPR/
├── configs/
│   └── default.yaml          # Hyperparameters: patch_size=16, res=224, L1=4, L2=16, θ=0.05, lr=1e-4
├── data/
│   ├── dataset.py            # TVPReid (PRID-2011, iLIDS, DukeMTMC) DataLoader
│   └── transforms.py         # Spatio-temporal augmentation & NLTK preprocessing
├── models/
│   ├── text_prompter.py      # Trích xuất Verbs/Nouns/Adjectives, Ma trận W = c_n/C không train, Tips
│   ├── text_encoder.py       # BERT [CLS] extraction + Positional projection
│   ├── visual_encoder.py     # ViT (4 random frames) + Additive E_S & E_T Embeddings
│   ├── motion_encoder.py     # S3D Separable Convs + Inception + Gating Mechanism
│   ├── spaces.py             # FeatureConvertor (Linear + BN + Sigmoid) & Common Space Projectors
│   └── mfgf_main.py          # Feature Aggregator + Alpha-blending + Complete MFGF Pipeline
├── loss/
│   └── criterion.py          # L_common (InfoNCE), L_D2 (Distillation + Scaled BCE), L_MFGF
├── utils/
│   ├── metrics.py            # Rank@1, 5, 10, 50, Median Rank (MdR), MRR, mAP
│   └── paths.py              # Centralized Path Management (Tự động nhận diện đường dẫn dự án)
├── train.py                  # Training loop với Warm-up (3 epochs) & Cosine Annealing
├── test.py                   # Benchmark Evaluation & Single-Query Retrieval Demo
├── app.py                    # Modern Desktop GUI Tracking Application
├── pipeline_tracking.py      # Tracking Engine & Text-to-Video Matching Pipeline
├── requirements.txt          # Thư viện phụ thuộc
└── README.md                 # Tài liệu hướng dẫn
```

---

## 3. Cấu hình Đường dẫn Tập trung (Centralized Path Management)

Dự án sử dụng cơ chế quản lý đường dẫn tập trung tại **`src/utils/paths.py`**:
* **Tự động nhận diện** toàn bộ thư mục dữ liệu `data/` (`captions/`, `videos/`), `checkpoints/`, `configs/`, `outputs/`, `reports/` mà không cần hardcode đường dẫn tuyệt đối.
* Nếu muốn đổi đường dẫn dữ liệu `data_root`, bạn chỉ cần chỉnh sửa **duy nhất 1 dòng** trong `src/configs/default.yaml` (`data.data_root`). Mặc định là `"auto"` để tự động tìm kiếm.
* Tất cả các lệnh có thể chạy trực tiếp từ thư mục gốc dự án (`D:\PBL6`) hoặc từ thư mục `D:\PBL6\src`.

---

## 4. Hướng dẫn Chạy Thử nghiệm

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

### 4.2. Huấn luyện (Training)
```bash
python src/train.py
# hoặc:
python src/train.py --config src/configs/default.yaml
```

### 4.3. Đánh giá trên Test Set (Evaluation)
```bash
python src/test.py
# hoặc chỉ định checkpoint:
python src/test.py --checkpoint src/checkpoints/best.pth
```

### 4.4. Truy vấn Trực quan (Inference Demo)
```bash
python src/test.py --query "Người phụ nữ mặc áo khoác trắng và quần xanh dương."
```

### 4.5. Giao diện Web UI (Gradio)
```bash
python src/web_app.py
# Mở trình duyệt tại: http://localhost:7860
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
