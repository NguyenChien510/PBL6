#!/usr/bin/env bash
set -e

# Đảm bảo các thư mục cần thiết luôn tồn tại
mkdir -p /app/src/outputs /app/src/checkpoints /app/src/reports /app/.cache/huggingface /app/.cache/torch

# Hiển thị thông tin môi trường và GPU nếu có
echo "======================================================================"
echo "          MFGF TVPR - Text-to-Video Person Tracking Docker            "
echo "======================================================================"
python -c "import torch; print(f'[*] Python: PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()}', flush=True)"
if command -v nvidia-smi &> /dev/null; then
    echo "[*] GPU Device Info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi
echo "======================================================================"

case "$1" in
  web)
    shift
    echo "[*] Khởi động Gradio Web App tại cổng 7860..."
    exec python src/web_app.py --server-name 0.0.0.0 --server-port 7860 "$@"
    ;;

  track)
    shift
    echo "[*] Chạy chế độ Theo vết đối tượng qua CLI (pipeline_tracking.py)..."
    exec python src/pipeline_tracking.py "$@"
    ;;

  test)
    shift
    echo "[*] Chạy chế độ Đánh giá mô hình (test.py)..."
    exec python src/test.py "$@"
    ;;

  train)
    shift
    echo "[*] Chạy chế độ Huấn luyện mô hình (train.py)..."
    exec python src/train.py "$@"
    ;;

  bash|sh)
    echo "[*] Truy cập Interactive Shell..."
    exec /bin/bash
    ;;

  *)
    # Nếu truyền lệnh tùy ý khác
    exec "$@"
    ;;
esac
