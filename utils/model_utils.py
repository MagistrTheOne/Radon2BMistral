"""
Model utilities for saving, loading, and converting models
"""

import os
import torch
import json
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import shutil

try:
    from safetensors import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Install with: pip install safetensors")

from huggingface_hub import HfApi, Repository
from transformers import PreTrainedModel, PreTrainedTokenizer


def save_model(
    model: PreTrainedModel,
    save_directory: str,
    model_name: Optional[str] = None,
    save_format: str = "pytorch",
    config: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    create_model_card: bool = True
) -> Dict[str, str]:
    """
    Save model in specified format
    
    Args:
        model: Model to save
        save_directory: Directory to save model
        model_name: Optional model name
        save_format: Format to save ("pytorch", "safetensors", "both")
        config: Optional configuration dictionary
        tokenizer: Optional tokenizer to save
        create_model_card: Whether to create model card
    
    Returns:
        Dictionary with saved file paths
    """
    
    os.makedirs(save_directory, exist_ok=True)
    saved_files = {}
    
    # Save model configuration
    if config:
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        saved_files["config"] = config_path
    
    # Save model weights
    if save_format in ["pytorch", "both"]:
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model.state_dict(), model_path)
        saved_files["pytorch_model"] = model_path
    
    if save_format in ["safetensors", "both"] and SAFETENSORS_AVAILABLE:
        safetensors_path = os.path.join(save_directory, "model.safetensors")
        # Convert state dict to safetensors format
        state_dict = model.state_dict()
        save_file(state_dict, safetensors_path)
        saved_files["safetensors_model"] = safetensors_path
    elif save_format == "safetensors" and not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors not available. Install with: pip install safetensors")
    
    # Save tokenizer
    if tokenizer:
        tokenizer.save_pretrained(save_directory)
        saved_files["tokenizer"] = os.path.join(save_directory, "tokenizer")
    
    # Create model card
    if create_model_card:
        model_card_path = create_model_card_file(
            save_directory,
            model_name=model_name,
            config=config
        )
        saved_files["model_card"] = model_card_path
    
    return saved_files


def load_model(
    model_class,
    model_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float32
) -> PreTrainedModel:
    """
    Load model from saved checkpoint
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to model file
        config_path: Optional path to config file
        device: Device to load model on
        torch_dtype: Torch data type
    
    Returns:
        Loaded model
    """
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    else:
        # Try to find config in same directory
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Create model instance
    model = model_class.from_pretrained(
        model_path,
        config=config_dict,
        torch_dtype=torch_dtype
    )
    
    # Move to device
    model.to(device)
    
    return model


def convert_to_safetensors(
    model_path: str,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None
) -> str:
    """
    Convert PyTorch model to Safetensors format
    
    Args:
        model_path: Path to PyTorch model file
        output_path: Optional output path for safetensors file
        config_path: Optional path to config file
    
    Returns:
        Path to converted safetensors file
    """
    
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors not available. Install with: pip install safetensors")
    
    # Load PyTorch model
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Determine output path
    if output_path is None:
        base_path = os.path.splitext(model_path)[0]
        output_path = f"{base_path}.safetensors"
    
    # Convert to safetensors
    save_file(state_dict, output_path)
    
    return output_path


def convert_from_safetensors(
    safetensors_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Convert Safetensors model to PyTorch format
    
    Args:
        safetensors_path: Path to safetensors file
        output_path: Optional output path for PyTorch file
    
    Returns:
        Path to converted PyTorch file
    """
    
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors not available. Install with: pip install safetensors")
    
    # Load safetensors model
    state_dict = load_file(safetensors_path)
    
    # Determine output path
    if output_path is None:
        base_path = os.path.splitext(safetensors_path)[0]
        output_path = f"{base_path}.bin"
    
    # Convert to PyTorch
    torch.save(state_dict, output_path)
    
    return output_path


def get_model_size(model: PreTrainedModel) -> Dict[str, int]:
    """
    Get model size information
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with size information
    """
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "parameter_size_bytes": param_size,
        "buffer_size_bytes": buffer_size,
        "total_size_bytes": param_size + buffer_size,
        "total_size_mb": (param_size + buffer_size) / (1024 * 1024),
        "total_size_gb": (param_size + buffer_size) / (1024 * 1024 * 1024)
    }


def create_model_card_file(
    save_directory: str,
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    usage_example: Optional[str] = None
) -> str:
    """
    Create model card README.md file
    
    Args:
        save_directory: Directory to save model card
        model_name: Optional model name
        config: Optional model configuration
        description: Optional model description
        usage_example: Optional usage example
    
    Returns:
        Path to created model card file
    """
    
    model_card_path = os.path.join(save_directory, "README.md")
    
    # Default description
    if description is None:
        description = f"Custom transformer model{' - ' + model_name if model_name else ''}"
    
    # Default usage example
    if usage_example is None:
        usage_example = """
```python
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
model = AutoModel.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Generate text
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```
"""
    
    # Create model card content
    model_card_content = f"""# {model_name or 'Custom Transformer Model'}

{description}

## Model Information

"""
    
    # Add configuration if available
    if config:
        model_card_content += "### Configuration\n\n"
        model_card_content += "```json\n"
        model_card_content += json.dumps(config, indent=2, ensure_ascii=False)
        model_card_content += "\n```\n\n"
    
    # Add usage example
    model_card_content += "## Usage\n\n"
    model_card_content += usage_example
    
    # Add additional sections
    model_card_content += """
## Model Details

- **Model Type**: Custom Transformer
- **Architecture**: GPT-2 / T5 compatible
- **Training**: Custom training pipeline
- **Tokenizer**: SentencePiece BPE

## Files

- `config.json`: Model configuration
- `pytorch_model.bin`: PyTorch model weights
- `model.safetensors`: Safetensors model weights (if available)
- `tokenizer/`: Tokenizer files
- `README.md`: This model card

## License

This model is provided for research and educational purposes.
"""
    
    # Write model card
    with open(model_card_path, 'w', encoding='utf-8') as f:
        f.write(model_card_content)
    
    return model_card_path


def push_to_huggingface_hub(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    repo_name: str,
    hf_token: str,
    private: bool = False,
    commit_message: str = "Add custom transformer model"
) -> str:
    """
    Push model and tokenizer to Hugging Face Hub
    
    Args:
        model: Model to upload
        tokenizer: Tokenizer to upload
        repo_name: Repository name on HF Hub
        hf_token: Hugging Face token
        private: Whether repository should be private
        commit_message: Commit message
    
    Returns:
        Repository URL
    """
    
    # Initialize HF API
    api = HfApi()
    
    # Create repository
    try:
        api.create_repo(
            repo_id=repo_name,
            token=hf_token,
            private=private,
            exist_ok=True
        )
    except Exception as e:
        print(f"Repository creation failed: {e}")
        raise
    
    # Create local repository
    repo = Repository(
        local_dir=f"./temp_{repo_name}",
        clone_from=repo_name,
        use_auth_token=hf_token
    )
    
    # Save model and tokenizer
    model.save_pretrained(repo.local_dir)
    tokenizer.save_pretrained(repo.local_dir)
    
    # Create model card
    create_model_card_file(
        repo.local_dir,
        model_name=repo_name,
        config=model.config.to_dict() if hasattr(model, 'config') else None
    )
    
    # Push to hub
    try:
        repo.push_to_hub(commit_message=commit_message)
    except Exception as e:
        print(f"Push to hub failed: {e}")
        raise
    finally:
        # Clean up local repository
        if os.path.exists(repo.local_dir):
            shutil.rmtree(repo.local_dir)
    
    return f"https://huggingface.co/{repo_name}"


def optimize_model_for_inference(
    model: PreTrainedModel,
    device: str = "cpu",
    optimize_memory: bool = True,
    use_half_precision: bool = False
) -> PreTrainedModel:
    """
    Optimize model for inference
    
    Args:
        model: Model to optimize
        device: Target device
        optimize_memory: Whether to optimize memory usage
        use_half_precision: Whether to use half precision
    
    Returns:
        Optimized model
    """
    
    # Move to device
    model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    # Half precision optimization
    if use_half_precision and device != "cpu":
        model.half()
    
    # Memory optimization
    if optimize_memory:
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Clear cache if using CUDA
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    
    return model


def benchmark_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_prompts: list,
    device: str = "cpu",
    num_runs: int = 5
) -> Dict[str, float]:
    """
    Benchmark model performance
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        test_prompts: List of test prompts
        device: Device to run on
        num_runs: Number of benchmark runs
    
    Returns:
        Benchmark results
    """
    
    import time
    
    model.eval()
    model.to(device)
    
    total_time = 0
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(num_runs):
            for prompt in test_prompts:
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # Measure generation time
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    do_sample=True,
                    temperature=0.7
                )
                end_time = time.time()
                
                # Calculate metrics
                generation_time = end_time - start_time
                generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
                
                total_time += generation_time
                total_tokens += generated_tokens
    
    # Calculate average metrics
    avg_time = total_time / (num_runs * len(test_prompts))
    avg_tokens = total_tokens / (num_runs * len(test_prompts))
    tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
    
    return {
        "average_generation_time": avg_time,
        "average_tokens_generated": avg_tokens,
        "tokens_per_second": tokens_per_second,
        "total_runs": num_runs * len(test_prompts)
    }
