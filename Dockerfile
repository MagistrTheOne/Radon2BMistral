# RADON Mistral Transformer Dockerfile - Optimized for Yandex Cloud
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt

# Install Flash Attention (optional, for better performance)
RUN pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed, continuing without it"

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0
ENV USE_FLASH_ATTENTION=true
ENV MODEL_CACHE_DIR=/app/models
ENV TOKENIZER_CACHE_DIR=/app/tokenizer

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set work directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs outputs checkpoints models tokenizer

# Set permissions
RUN chmod +x scripts/*.py

# Pre-download model (if available)
# COPY artifacts/ ./artifacts/
# RUN python scripts/preload_model.py --model_path ./artifacts

# Environment for clean model initialization
ENV CLEAN_MODEL=true
ENV MODEL_INIT_FROM_SCRATCH=true
ENV CORPUS_PATH=/app/data/raw_corpus

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with optimizations
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
