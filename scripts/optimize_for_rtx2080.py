"""
Оптимизация RADON для RTX 2080 (8GB VRAM)
"""

import torch
import os
from models.config import ModelConfig


def optimize_for_rtx2080():
    """Оптимизация конфигурации для RTX 2080"""
    
    print("🔧 Optimizing RADON for RTX 2080 (8GB VRAM)")
    
    # Создать оптимизированную конфигурацию
    config = ModelConfig(
        model_name="radon",
        model_type="mistral",
        vocab_size=32000,
        hidden_size=1536,      # Уменьшено с 2048
        num_layers=20,         # Уменьшено с 24
        num_attention_heads=24, # Уменьшено с 32
        num_kv_heads=6,         # Уменьшено с 8
        intermediate_size=4096, # Уменьшено с 5632
        max_position_embeddings=8192,  # Уменьшено с 32768
        sliding_window=2048,    # Уменьшено с 4096
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        dropout=0.1,
        attention_dropout=0.1,
        activation_function="silu",
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        torch_dtype="float16",  # FP16 вместо FP32
        output_attentions=False,
        output_hidden_states=False
    )
    
    # Сохранить конфигурацию
    config.to_json("configs/model_config_mistral_rtx2080.json")
    
    # Оценить размер модели
    estimated_params = (
        config.vocab_size * config.hidden_size +  # Embeddings
        config.num_layers * (
            config.hidden_size * config.hidden_size * 4 +  # Attention
            config.hidden_size * config.intermediate_size * 2  # FFN
        ) +
        config.hidden_size * config.vocab_size  # LM Head
    )
    
    # Размер в байтах (FP16)
    model_size_fp16 = estimated_params * 2 / (1024**3)  # GB
    model_size_fp32 = estimated_params * 4 / (1024**3)  # GB
    
    print(f"📊 Model size estimation:")
    print(f"   - Parameters: {estimated_params:,}")
    print(f"   - FP16: {model_size_fp16:.2f} GB")
    print(f"   - FP32: {model_size_fp32:.2f} GB")
    
    # KV Cache estimation
    kv_cache_size = (
        config.num_layers * 
        config.num_kv_heads * 
        config.max_position_embeddings * 
        (config.hidden_size // config.num_attention_heads) * 2  # K + V
    ) * 2 / (1024**3)  # FP16
    
    print(f"   - KV Cache (8K): {kv_cache_size:.2f} GB")
    print(f"   - Total (FP16): {model_size_fp16 + kv_cache_size:.2f} GB")
    
    return config


def create_rtx2080_dockerfile():
    """Создать оптимизированный Dockerfile для RTX 2080"""
    
    dockerfile_content = '''# RADON Mistral - Optimized for RTX 2080
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0
ENV USE_FLASH_ATTENTION=false  # Отключено для RTX 2080
ENV TORCH_DTYPE=float16
ENV MODEL_CONFIG_PATH=configs/model_config_mistral_rtx2080.json

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3.9 \\
    python3.9-dev \\
    python3-pip \\
    build-essential \\
    curl \\
    git \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (без Flash Attention)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs outputs checkpoints models tokenizer

# Set permissions
RUN chmod +x scripts/*.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run with memory optimizations
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
'''
    
    with open("Dockerfile.rtx2080", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Created Dockerfile.rtx2080")


def create_rtx2080_docker_compose():
    """Создать docker-compose для RTX 2080"""
    
    compose_content = '''version: '3.8'

services:
  radon-rtx2080:
    build: 
      context: .
      dockerfile: Dockerfile.rtx2080
    container_name: radon-rtx2080
    ports:
      - "8000:8000"
    environment:
      - MODEL_CONFIG_PATH=configs/model_config_mistral_rtx2080.json
      - DEVICE=cuda
      - TORCH_DTYPE=float16
      - USE_FLASH_ATTENTION=false
      - MEMORY_LIMIT=7GB
      - MAX_CONTEXT_LENGTH=8192
      - VK_ACCESS_TOKEN=${VK_ACCESS_TOKEN:-}
      - VK_CONFIRMATION_CODE=${VK_CONFIRMATION_CODE:-default_confirmation_code}
    volumes:
      - ./logs:/app/logs
      - ./outputs:/app/outputs
      - ./checkpoints:/app/checkpoints
      - ./models:/app/models
      - ./tokenizer:/app/tokenizer
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
'''
    
    with open("docker-compose.rtx2080.yml", "w") as f:
        f.write(compose_content)
    
    print("✅ Created docker-compose.rtx2080.yml")


def main():
    """Основная функция оптимизации"""
    
    print("🚀 RADON RTX 2080 Optimization")
    print("=" * 40)
    
    # Создать оптимизированную конфигурацию
    config = optimize_for_rtx2080()
    
    # Создать Docker файлы
    create_rtx2080_dockerfile()
    create_rtx2080_docker_compose()
    
    print("\n✅ Optimization complete!")
    print("\n📋 Next steps:")
    print("1. Build: docker-compose -f docker-compose.rtx2080.yml build")
    print("2. Run: docker-compose -f docker-compose.rtx2080.yml up")
    print("3. Test: curl http://localhost:8000/health")
    
    print("\n⚠️  Recommendations:")
    print("- Use FP16 for memory efficiency")
    print("- Disable Flash Attention (not supported on RTX 2080)")
    print("- Limit context to 8K tokens")
    print("- Monitor VRAM usage with nvidia-smi")


if __name__ == "__main__":
    main()
