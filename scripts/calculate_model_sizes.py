"""
Calculate model sizes for different RADON configurations
"""

import json
import torch


def calculate_model_size(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    max_position_embeddings: int,
    dtype: str = "float16"
):
    """Рассчитать размер модели"""
    
    # Определяем размер элемента данных
    dtype_sizes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    element_size = dtype_sizes.get(dtype, 2)
    
    # Подсчитываем параметры
    parameters = 0
    
    # 1. Embeddings
    embed_params = vocab_size * hidden_size
    parameters += embed_params
    print(f"📊 Embeddings: {embed_params:,} parameters")
    
    # 2. Layers
    for layer_idx in range(num_layers):
        layer_params = 0
        
        # Attention weights
        q_proj = hidden_size * hidden_size
        k_proj = num_kv_heads * (hidden_size // num_attention_heads) * hidden_size
        v_proj = num_kv_heads * (hidden_size // num_attention_heads) * hidden_size
        o_proj = hidden_size * hidden_size
        
        attention_params = q_proj + k_proj + v_proj + o_proj
        layer_params += attention_params
        
        # MLP weights
        gate_proj = intermediate_size * hidden_size
        up_proj = intermediate_size * hidden_size
        down_proj = hidden_size * intermediate_size
        
        mlp_params = gate_proj + up_proj + down_proj
        layer_params += mlp_params
        
        # Layer norms
        layer_norm_params = hidden_size * 2  # input_layernorm + post_attention_layernorm
        layer_params += layer_norm_params
        
        parameters += layer_params
        print(f"📊 Layer {layer_idx + 1}: {layer_params:,} parameters")
    
    # 3. Final layer norm
    final_norm_params = hidden_size
    parameters += final_norm_params
    print(f"📊 Final norm: {final_norm_params:,} parameters")
    
    # 4. LM head
    lm_head_params = vocab_size * hidden_size
    parameters += lm_head_params
    print(f"📊 LM head: {lm_head_params:,} parameters")
    
    # Рассчитываем размеры
    size_bytes = parameters * element_size
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    
    return {
        "parameters": parameters,
        "size_bytes": size_bytes,
        "size_mb": size_mb,
        "size_gb": size_gb,
        "dtype": dtype,
        "element_size": element_size
    }


def analyze_radon_configs():
    """Анализировать размеры разных конфигураций RADON"""
    
    print("🔍 RADON Model Size Analysis")
    print("=" * 50)
    
    # Загружаем конфигурации
    configs = {
        "Small (50M)": {
            "vocab_size": 8192,
            "hidden_size": 512,
            "num_layers": 6,
            "num_attention_heads": 8,
            "num_kv_heads": 2,
            "intermediate_size": 1024,
            "max_position_embeddings": 2048
        },
        "Medium (500M)": {
            "vocab_size": 16384,
            "hidden_size": 1024,
            "num_layers": 12,
            "num_attention_heads": 16,
            "num_kv_heads": 4,
            "intermediate_size": 2048,
            "max_position_embeddings": 4096
        },
        "Large (2B)": {
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_layers": 24,
            "num_attention_heads": 32,
            "num_kv_heads": 8,
            "intermediate_size": 5632,
            "max_position_embeddings": 8192
        },
        "XL (7B)": {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_kv_heads": 8,
            "intermediate_size": 11008,
            "max_position_embeddings": 16384
        }
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n📊 {config_name} Configuration:")
        print("-" * 30)
        
        # Рассчитываем для разных типов данных
        for dtype in ["float16", "float32", "int8", "int4"]:
            result = calculate_model_size(dtype=dtype, **config)
            results[f"{config_name}_{dtype}"] = result
            
            print(f"   {dtype.upper()}: {result['parameters']:,} params, {result['size_gb']:.2f} GB")
    
    return results


def estimate_training_requirements():
    """Оценить требования для тренировки"""
    
    print("\n🚀 Training Requirements Estimation")
    print("=" * 50)
    
    # Примерные требования для тренировки
    training_multipliers = {
        "optimizer_states": 2,  # AdamW хранит momentum и variance
        "gradients": 1,         # Градиенты
        "activations": 0.5,     # Активации (зависит от batch size)
        "mixed_precision": 0.5, # FP16 training
        "gradient_checkpointing": 0.3  # Экономия памяти
    }
    
    configs = {
        "2B Model": {
            "parameters": 2_000_000_000,
            "base_memory_gb": 4.0  # FP16
        },
        "7B Model": {
            "parameters": 7_000_000_000,
            "base_memory_gb": 14.0  # FP16
        }
    }
    
    for model_name, config in configs.items():
        print(f"\n📊 {model_name} Training Requirements:")
        
        base_memory = config["base_memory_gb"]
        
        # Стандартная тренировка
        standard_memory = base_memory * (1 + training_multipliers["optimizer_states"] + training_multipliers["gradients"])
        print(f"   Standard training: {standard_memory:.1f} GB")
        
        # С mixed precision
        mixed_precision_memory = base_memory * (1 + training_multipliers["optimizer_states"] + training_multipliers["gradients"] + training_multipliers["mixed_precision"])
        print(f"   Mixed precision: {mixed_precision_memory:.1f} GB")
        
        # С gradient checkpointing
        checkpointing_memory = base_memory * (1 + training_multipliers["optimizer_states"] + training_multipliers["gradients"] + training_multipliers["gradient_checkpointing"])
        print(f"   + Gradient checkpointing: {checkpointing_memory:.1f} GB")
        
        # Рекомендации по GPU
        print(f"   Recommended GPUs:")
        if standard_memory <= 24:
            print(f"     - RTX 4090 (24GB): ✅")
        if standard_memory <= 40:
            print(f"     - A100 (40GB): ✅")
        if standard_memory <= 80:
            print(f"     - A100 (80GB): ✅")
        if standard_memory > 80:
            print(f"     - Multi-GPU setup required")


def main():
    """Основная функция"""
    
    # Анализируем размеры
    results = analyze_radon_configs()
    
    # Оцениваем требования для тренировки
    estimate_training_requirements()
    
    print("\n📋 Summary:")
    print("=" * 50)
    print("🔹 Small (50M): ~100MB - для разработки и тестирования")
    print("🔹 Medium (500M): ~1GB - для экспериментов")
    print("🔹 Large (2B): ~4GB - основная модель")
    print("🔹 XL (7B): ~14GB - максимальная производительность")
    
    print("\n💡 Recommendations:")
    print("🔹 RTX 2080 (8GB): Small + Medium модели")
    print("🔹 RTX 4070 (12GB): Medium + Large модели")
    print("🔹 RTX 4090 (24GB): Large + XL модели")
    print("🔹 A100 (40GB+): Все модели + тренировка")


if __name__ == "__main__":
    main()
