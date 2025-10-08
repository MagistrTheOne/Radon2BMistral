"""
Initialize Mistral model for deployment
"""

import os
import json
import torch
from pathlib import Path
from models.config import ModelConfig
from models.mistral_model import MistralForCausalLM
from tokenizer.hybrid_tokenizer import HybridTokenizer


def initialize_mistral_model(
    config_path: str,
    output_dir: str,
    model_size: str = "2b",
    device: str = "cpu"
):
    """Initialize Mistral model with proper weights"""
    
    print(f"[+] Initializing Mistral-{model_size} model...")
    
    # Load configuration
    config = ModelConfig.from_json(config_path)
    print(f"[+] Config loaded: {config.model_name} ({config.model_type})")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = MistralForCausalLM(config)
    
    # Initialize tokenizer
    tokenizer = HybridTokenizer()
    
    # Initialize weights (Xavier/He initialization)
    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
    
    model.apply(init_weights)
    print("[+] Model weights initialized")
    
    # Move to device
    model = model.to(device)
    
    # Save model
    model.save_pretrained(output_dir)
    print(f"[+] Model saved to {output_dir}")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"[+] Tokenizer saved to {output_dir}")
    
    # Save config
    config.to_json(os.path.join(output_dir, "config.json"))
    print(f"[+] Config saved to {output_dir}/config.json")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[+] Model info:")
    print(f"    - Total parameters: {total_params:,}")
    print(f"    - Trainable parameters: {trainable_params:,}")
    print(f"    - Model size: {total_params * 4 / 1024**3:.2f} GB (FP32)")
    
    return model, tokenizer


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Mistral model")
    parser.add_argument("--config_path", required=True, help="Path to model config")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--model_size", default="2b", choices=["2b", "7b"], help="Model size")
    parser.add_argument("--device", default="cpu", help="Device to use")
    
    args = parser.parse_args()
    
    initialize_mistral_model(
        config_path=args.config_path,
        output_dir=args.output_dir,
        model_size=args.model_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
