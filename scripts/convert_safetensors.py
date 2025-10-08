"""
Convert model weights to Safetensors format
Supports Mistral, GPT-2, and T5 models
"""

import os
import torch
from safetensors.torch import save_file
from models.hybrid_model import HybridTransformerModel
from models.config import ModelConfig
from models.mistral_model import MistralForCausalLM
from models.transformer_gpt2 import CustomGPT2Model
from models.transformer_t5 import CustomT5Model

# Configuration
CFG_PATH = os.environ.get("RADON_CONFIG", "configs/model_config_mistral_2b.json")
OUT_PATH = os.environ.get("OUT", "artifacts/model.safetensors")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "mistral")  # mistral, gpt2, t5, hybrid

def build_model_from_config(config_path: str, model_type: str = "mistral"):
    """
    Build model from configuration
    
    Args:
        config_path: Path to model configuration
        model_type: Type of model to build
        
    Returns:
        Initialized model
    """
    config = ModelConfig.from_json(config_path)
    
    if model_type == "mistral":
        return MistralForCausalLM(config)
    elif model_type == "gpt2":
        return CustomGPT2Model(config)
    elif model_type == "t5":
        return CustomT5Model(config)
    elif model_type == "hybrid":
        return HybridTransformerModel(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def convert_to_safetensors(
    config_path: str,
    output_path: str,
    model_type: str = "mistral",
    quantize: bool = False,
    dtype: str = "float32"
):
    """
    Convert model to Safetensors format
    
    Args:
        config_path: Path to model configuration
        output_path: Output path for Safetensors file
        model_type: Type of model to convert
        quantize: Whether to quantize the model
        dtype: Data type for conversion
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load model config
    print(f"[+] Loading config from {config_path}")
    config = ModelConfig.from_json(config_path)
    
    # Initialize model
    print(f"[+] Initializing {model_type} model")
    model = build_model_from_config(config_path, model_type)
    
    # Get state dict
    state_dict = model.state_dict()
    
    # Convert dtype if needed
    if dtype != "float32":
        target_dtype = getattr(torch, dtype)
        state_dict = {k: v.to(target_dtype) for k, v in state_dict.items()}
        print(f"[+] Converted to {dtype}")
    
    # Quantize if requested
    if quantize:
        print("[+] Quantizing model to INT8...")
        # Simple quantization (in production, use proper quantization)
        quantized_state_dict = {}
        for name, tensor in state_dict.items():
            if tensor.dtype == torch.float32:
                # Quantize to INT8
                scale = tensor.abs().max() / 127.0
                quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
                quantized_state_dict[name] = quantized
                quantized_state_dict[f"{name}_scale"] = scale
            else:
                quantized_state_dict[name] = tensor
        state_dict = quantized_state_dict
        print("[+] Quantization completed")
    
    # Save to safetensors
    print(f"[+] Saving to {output_path}")
    save_file(state_dict, output_path)
    
    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] Done! Saved {size_mb:.2f} MB → {output_path}")
    
    # Also save config alongside
    config_out = output_path.replace('.safetensors', '_config.json')
    config.to_json(config_out)
    print(f"[+] Config saved → {config_out}")
    
    return output_path

def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert model to Safetensors format")
    parser.add_argument("--config", default=CFG_PATH, help="Path to model config")
    parser.add_argument("--output", default=OUT_PATH, help="Output path")
    parser.add_argument("--model_type", default=MODEL_TYPE, choices=["mistral", "gpt2", "t5", "hybrid"], help="Model type")
    parser.add_argument("--quantize", action="store_true", help="Quantize to INT8")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Data type")
    
    args = parser.parse_args()
    
    convert_to_safetensors(
        config_path=args.config,
        output_path=args.output,
        model_type=args.model_type,
        quantize=args.quantize,
        dtype=args.dtype
    )

if __name__ == "__main__":
    main()

