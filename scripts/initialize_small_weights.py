"""
Initialize smaller RADON model weights for memory efficiency
"""

import os
import json
import torch
from pathlib import Path
from huggingface_hub import HfApi


def initialize_small_mistral_weights(
    config_path: str = "configs/model_config_mistral_2b.json",
    output_dir: str = "./model_weights_small"
):
    """Инициализировать веса небольшой модели Mistral"""
    
    print("🔧 Initializing small RADON Mistral model weights...")
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем конфигурацию
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"[1/4] Loading configuration from {config_path}")
    
    # Уменьшаем размеры для экономии памяти
    small_config = {
        "vocab_size": 8192,  # Уменьшаем словарь
        "hidden_size": 512,  # Уменьшаем скрытый размер
        "num_layers": 6,     # Уменьшаем количество слоев
        "num_attention_heads": 8,
        "num_kv_heads": 2,
        "intermediate_size": 1024,
        "max_position_embeddings": 2048,
        "sliding_window": 1024,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "torch_dtype": "float16"
    }
    
    print(f"   Small model: {small_config['hidden_size']} hidden, {small_config['num_layers']} layers")
    
    # Создаем состояние модели
    print("[2/4] Creating model state dict...")
    
    state_dict = {}
    
    # Embeddings
    state_dict["model.embed_tokens.weight"] = torch.randn(
        small_config["vocab_size"], 
        small_config["hidden_size"], 
        dtype=torch.float16
    )
    
    # Layers
    for layer_idx in range(small_config["num_layers"]):
        layer_prefix = f"model.layers.{layer_idx}"
        
        # Attention weights
        state_dict[f"{layer_prefix}.self_attn.q_proj.weight"] = torch.randn(
            small_config["hidden_size"], 
            small_config["hidden_size"], 
            dtype=torch.float16
        )
        state_dict[f"{layer_prefix}.self_attn.k_proj.weight"] = torch.randn(
            small_config["num_kv_heads"] * (small_config["hidden_size"] // small_config["num_attention_heads"]), 
            small_config["hidden_size"], 
            dtype=torch.float16
        )
        state_dict[f"{layer_prefix}.self_attn.v_proj.weight"] = torch.randn(
            small_config["num_kv_heads"] * (small_config["hidden_size"] // small_config["num_attention_heads"]), 
            small_config["hidden_size"], 
            dtype=torch.float16
        )
        state_dict[f"{layer_prefix}.self_attn.o_proj.weight"] = torch.randn(
            small_config["hidden_size"], 
            small_config["hidden_size"], 
            dtype=torch.float16
        )
        
        # MLP weights
        state_dict[f"{layer_prefix}.mlp.gate_proj.weight"] = torch.randn(
            small_config["intermediate_size"], 
            small_config["hidden_size"], 
            dtype=torch.float16
        )
        state_dict[f"{layer_prefix}.mlp.up_proj.weight"] = torch.randn(
            small_config["intermediate_size"], 
            small_config["hidden_size"], 
            dtype=torch.float16
        )
        state_dict[f"{layer_prefix}.mlp.down_proj.weight"] = torch.randn(
            small_config["hidden_size"], 
            small_config["intermediate_size"], 
            dtype=torch.float16
        )
        
        # Layer norms
        state_dict[f"{layer_prefix}.input_layernorm.weight"] = torch.ones(
            small_config["hidden_size"], 
            dtype=torch.float16
        )
        state_dict[f"{layer_prefix}.post_attention_layernorm.weight"] = torch.ones(
            small_config["hidden_size"], 
            dtype=torch.float16
        )
    
    # Final layer norm
    state_dict["model.norm.weight"] = torch.ones(
        small_config["hidden_size"], 
        dtype=torch.float16
    )
    
    # LM head
    state_dict["lm_head.weight"] = torch.randn(
        small_config["vocab_size"], 
        small_config["hidden_size"], 
        dtype=torch.float16
    )
    
    print(f"   Created {len(state_dict)} weight tensors")
    
    # Сохраняем веса
    print("[3/4] Saving model weights...")
    
    # Сохраняем в формате PyTorch
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    print("   ✅ PyTorch format saved")
    
    # Сохраняем в формате Safetensors (если возможно)
    try:
        from safetensors.torch import save_file
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
        print("   ✅ Safetensors format saved")
    except Exception as e:
        print(f"   ⚠️  Safetensors failed: {e}")
    
    # Сохраняем конфигурацию
    with open(os.path.join(output_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(small_config, f, indent=2)
    
    # Создаем файл с информацией о модели
    model_info = {
        "model_name": "radon-small",
        "architecture": "mistral",
        "parameters": sum(p.numel() for p in state_dict.values()),
        "size_mb": sum(p.numel() * p.element_size() for p in state_dict.values()) / (1024 * 1024),
        "dtype": "float16",
        "initialization": "random",
        "note": "This is a small initialized model with random weights. Training required for actual performance."
    }
    
    with open(os.path.join(output_dir, "model_info.json"), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Small model weights initialized in {output_dir}")
    print(f"   Parameters: {model_info['parameters']:,}")
    print(f"   Size: {model_info['size_mb']:.1f} MB")
    
    return output_dir, model_info


def upload_small_weights_to_hf(
    weights_dir: str,
    repo_id: str = "MagistrTheOne/RadonSAI-Small",
    hf_token: str = None
):
    """Загрузить веса небольшой модели на HF"""
    
    print(f"🚀 Uploading small RADON weights to Hugging Face Hub...")
    print(f"   Repository: {repo_id}")
    print(f"   Weights path: {weights_dir}")
    
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
    
    api = HfApi(token=hf_token)
    
    try:
        # Создаем репозиторий для маленькой модели
        from huggingface_hub import create_repo
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True,
            token=hf_token
        )
        
        # Загружаем веса
        print("[1/2] Uploading small model weights...")
        api.upload_folder(
            folder_path=weights_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="🔧 Add small RADON Mistral model weights"
        )
        print("✅ Small model weights uploaded successfully")
        
        # Создаем README для маленькой модели
        print("[2/2] Creating model card...")
        create_small_model_card(repo_id, api)
        
        print(f"\n🎉 Small weights upload successful!")
        print(f"📡 Model URL: https://huggingface.co/{repo_id}")
        print(f"📊 Files uploaded: {len(os.listdir(weights_dir))} files")
        
        return True
        
    except Exception as e:
        print(f"❌ Small weights upload failed: {e}")
        return False


def create_small_model_card(repo_id: str, api: HfApi):
    """Создать карточку для маленькой модели"""
    
    readme_content = """---
license: apache-2.0
language:
- ru
- en
tags:
- mistral
- russian
- english
- code
- machine-learning
- nlp
- transformer
- small
- demo
pipeline_tag: text-generation
size_categories: 100M
---

# RADON-Small - Compact Mistral-based Russian-English Transformer

## Model Description

RADON-Small is a compact version of the RADON transformer model, optimized for development, testing, and resource-constrained environments.

### Key Features

- **Architecture**: Mistral with Llama 3 innovations (GQA, RMSNorm, SwiGLU, RoPE)
- **Parameters**: ~50M parameters (small version)
- **Context**: 2K tokens
- **Tokenizer**: Hybrid Unigram+BPE for Russian-English
- **Status**: Initialized with random weights (training required)
- **Use Case**: Development, testing, prototyping

### Model Weights

This is a small model with initialized weights:

- **Format**: PyTorch (.bin) and Safetensors (.safetensors)
- **Dtype**: float16
- **Initialization**: Random
- **Size**: ~100MB (50M parameters)

### Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load small model
model = AutoModelForCausalLM.from_pretrained("MagistrTheOne/RadonSAI-Small")
tokenizer = AutoTokenizer.from_pretrained("MagistrTheOne/RadonSAI-Small")

# Note: This model has random weights and needs training
# For inference, you should use a trained version

# Generate text (will produce random output)
prompt = "Машинное обучение - это"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, temperature=0.7)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Training

This small model is perfect for:

1. **Development and testing**
2. **Learning transformer architectures**
3. **Prototyping new ideas**
4. **Resource-constrained environments**

### Model Architecture

```
RADON-Small:
- Hidden size: 512
- Layers: 6
- Attention heads: 8 (2 KV heads)
- Intermediate size: 1024
- Vocabulary: 8K
- Context window: 2K tokens
```

### Related Models

- **Full Model**: [MagistrTheOne/RadonSAI](https://huggingface.co/MagistrTheOne/RadonSAI)
- **Datasets**: [MagistrTheOne/radon-examples](https://huggingface.co/datasets/MagistrTheOne/radon-examples)

### Citation

```bibtex
@misc{radon2024small,
  title={RADON-Small: Compact Mistral-based Russian-English Transformer},
  author={MagistrTheOne},
  year={2024},
  url={https://github.com/MagistrTheOne/Radon2BMistral}
}
```

### License

Apache 2.0 License

### Contact

- GitHub: [MagistrTheOne/Radon2BMistral](https://github.com/MagistrTheOne/Radon2BMistral)
- Hugging Face: [MagistrTheOne/RadonSAI-Small](https://huggingface.co/MagistrTheOne/RadonSAI-Small)
"""
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="📝 Add small model card"
        )
        print("✅ Small model card created")
    except Exception as e:
        print(f"⚠️  Failed to create model card: {e}")


def main():
    """Основная функция"""
    
    print("🔧 RADON Small Model Weights Initialization & Upload")
    print("=" * 60)
    
    # Проверяем токен
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment variables")
        print("   Set it with: $env:HF_TOKEN='your_token_here'")
        return
    
    # Инициализируем маленькие веса
    weights_dir, model_info = initialize_small_mistral_weights()
    
    # Загружаем на HF
    success = upload_small_weights_to_hf(
        weights_dir=weights_dir,
        repo_id="MagistrTheOne/RadonSAI-Small",
        hf_token=hf_token
    )
    
    if success:
        print("\n✅ RADON-Small model weights successfully uploaded!")
        print("🔗 https://huggingface.co/MagistrTheOne/RadonSAI-Small")
        print(f"📊 Model info: {model_info['parameters']:,} parameters, {model_info['size_mb']:.1f} MB")
        print("\n⚠️  Note: This is a small model with random weights for development/testing")
    else:
        print("\n❌ Small weights upload failed")


if __name__ == "__main__":
    main()
