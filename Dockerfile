# ==============================================================================
# MFGF TVPR - Production Dockerfile (GPU CUDA 12.1 & CPU Supported)
# Base: PyTorch official CUDA 12.1 runtime with cuDNN 9 & Python 3.10
# ==============================================================================

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Metadata
LABEL maintainer="MFGF-TVPR Project Team"
LABEL description="Text-to-Video Person Retrieval & Continuous Tracking using DINOv2 and PhoBERT v2"

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH="/app/src:/app" \
    TORCH_HOME="/app/.cache/torch" \
    HF_HOME="/app/.cache/huggingface" \
    TRANSFORMERS_CACHE="/app/.cache/huggingface" \
    NLTK_DATA="/app/.cache/nltk_data"

# Install essential system dependencies (OpenCV, FFmpeg, build tools for pyvi/CRF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create cache and runtime directories
RUN mkdir -p /app/.cache/torch \
             /app/.cache/huggingface \
             /app/.cache/nltk_data \
             /app/src/outputs \
             /app/src/checkpoints \
             /app/src/reports

# Copy requirements first for optimal layer caching
COPY src/requirements.txt /app/requirements.txt

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Pre-download required NLTK datasets to prevent runtime downloads and enable offline execution
RUN python -c "import nltk; \
    nltk.download('punkt', download_dir='/app/.cache/nltk_data'); \
    nltk.download('stopwords', download_dir='/app/.cache/nltk_data'); \
    nltk.download('averaged_perceptron_tagger', download_dir='/app/.cache/nltk_data'); \
    nltk.download('averaged_perceptron_tagger_eng', download_dir='/app/.cache/nltk_data')"

# Copy source code and entrypoint
COPY src/ /app/src/
COPY entrypoint.sh /app/entrypoint.sh

# Fix Windows CRLF line endings on entrypoint script and make executable
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose ports for Web UI (Gradio: 7860, API: 8000)
EXPOSE 7860 8000

# Set default entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command launches Gradio Web UI
CMD ["web"]
