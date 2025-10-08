"""
Quantization utilities for model optimization
Supports INT8, INT4, and dynamic quantization
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import os


class QuantizedModel:
    """
    Wrapper for quantized models
    """
    
    def __init__(self, model: nn.Module, quantization_config: Dict[str, Any]):
        self.model = model
        self.quantization_config = quantization_config
        self.is_quantized = True
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.model, name)


def quantize_model_int8(
    model: nn.Module,
    calibration_data: Optional[torch.Tensor] = None,
    per_channel: bool = True
) -> QuantizedModel:
    """
    Quantize model to INT8 using PyTorch quantization
    
    Args:
        model: Model to quantize
        calibration_data: Data for calibration (optional)
        per_channel: Whether to use per-channel quantization
        
    Returns:
        Quantized model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Prepare model for quantization
    if per_channel:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    else:
        model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # Prepare model
    prepared_model = torch.quantization.prepare(model)
    
    # Calibrate if data provided
    if calibration_data is not None:
        print("Calibrating model...")
        with torch.no_grad():
            prepared_model(calibration_data)
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(prepared_model)
    
    return QuantizedModel(quantized_model, {
        "type": "int8",
        "per_channel": per_channel,
        "backend": "fbgemm" if per_channel else "qnnpack"
    })


def quantize_model_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8
) -> QuantizedModel:
    """
    Apply dynamic quantization to model
    
    Args:
        model: Model to quantize
        dtype: Quantization dtype
        
    Returns:
        Quantized model
    """
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=dtype
    )
    
    return QuantizedModel(quantized_model, {
        "type": "dynamic",
        "dtype": str(dtype)
    })


def quantize_weights_int4(
    weights: torch.Tensor,
    group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to INT4 using GPTQ-style quantization
    
    Args:
        weights: Weight tensor to quantize
        group_size: Group size for quantization
        
    Returns:
        Tuple of (quantized_weights, scales)
    """
    # Reshape weights for group-wise quantization
    original_shape = weights.shape
    weights_flat = weights.view(-1, group_size)
    
    # Calculate scales for each group
    scales = weights_flat.abs().max(dim=1, keepdim=True)[0] / 7.0  # INT4 range: -8 to 7
    
    # Quantize weights
    quantized_weights = torch.round(weights_flat / scales).clamp(-8, 7).to(torch.int8)
    
    # Reshape back to original shape
    quantized_weights = quantized_weights.view(original_shape)
    scales = scales.view(original_shape[0], -1)
    
    return quantized_weights, scales


def dequantize_weights_int4(
    quantized_weights: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    """
    Dequantize INT4 weights back to float
    
    Args:
        quantized_weights: Quantized weight tensor
        scales: Scale tensor
        group_size: Group size used for quantization
        
    Returns:
        Dequantized weights
    """
    # Reshape for dequantization
    original_shape = quantized_weights.shape
    quantized_flat = quantized_weights.view(-1, group_size)
    scales_flat = scales.view(-1, 1)
    
    # Dequantize
    dequantized_weights = quantized_flat.float() * scales_flat
    
    # Reshape back
    return dequantized_weights.view(original_shape)


def apply_gptq_quantization(
    model: nn.Module,
    calibration_data: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.1
) -> QuantizedModel:
    """
    Apply GPTQ quantization to model
    
    Args:
        model: Model to quantize
        calibration_data: Data for calibration
        bits: Number of bits for quantization
        group_size: Group size for quantization
        damp_percent: Damping percentage
        
    Returns:
        Quantized model
    """
    # This is a simplified GPTQ implementation
    # In production, use the official GPTQ library
    
    model.eval()
    quantized_layers = {}
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get layer weights
                weights = module.weight.data
                
                # Apply GPTQ-style quantization
                if bits == 4:
                    quantized_weights, scales = quantize_weights_int4(weights, group_size)
                else:
                    # Fallback to INT8
                    quantized_weights = torch.round(weights / weights.abs().max() * 127).clamp(-128, 127).to(torch.int8)
                    scales = weights.abs().max()
                
                # Store quantized weights
                quantized_layers[name] = {
                    'weight': quantized_weights,
                    'scales': scales,
                    'bias': module.bias
                }
    
    # Create quantized model
    quantized_model = create_quantized_model(model, quantized_layers, bits)
    
    return QuantizedModel(quantized_model, {
        "type": "gptq",
        "bits": bits,
        "group_size": group_size,
        "damp_percent": damp_percent
    })


def create_quantized_model(
    original_model: nn.Module,
    quantized_layers: Dict[str, Dict[str, torch.Tensor]],
    bits: int
) -> nn.Module:
    """
    Create a model with quantized layers
    
    Args:
        original_model: Original model
        quantized_layers: Dictionary of quantized layers
        bits: Number of bits used for quantization
        
    Returns:
        Model with quantized layers
    """
    # Create a copy of the model
    quantized_model = type(original_model)(original_model.config)
    
    # Replace layers with quantized versions
    for name, module in quantized_model.named_modules():
        if name in quantized_layers:
            layer_data = quantized_layers[name]
            
            if bits == 4:
                # INT4 quantization
                dequantized_weight = dequantize_weights_int4(
                    layer_data['weight'],
                    layer_data['scales']
                )
                module.weight.data = dequantized_weight
            else:
                # INT8 quantization
                module.weight.data = layer_data['weight'].float()
            
            if layer_data['bias'] is not None:
                module.bias.data = layer_data['bias']
    
    return quantized_model


def save_quantized_model(
    model: QuantizedModel,
    save_path: str,
    format: str = "safetensors"
):
    """
    Save quantized model
    
    Args:
        model: Quantized model
        save_path: Path to save model
        format: Save format (safetensors, torch)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if format == "safetensors":
        from safetensors.torch import save_file
        state_dict = model.model.state_dict()
        save_file(state_dict, save_path)
    else:
        torch.save(model.model.state_dict(), save_path)
    
    # Save quantization config
    config_path = save_path.replace('.safetensors', '_quant_config.json').replace('.pt', '_quant_config.json')
    import json
    with open(config_path, 'w') as f:
        json.dump(model.quantization_config, f, indent=2)


def load_quantized_model(
    model_path: str,
    config_path: str,
    model_class,
    **kwargs
) -> QuantizedModel:
    """
    Load quantized model
    
    Args:
        model_path: Path to quantized model
        config_path: Path to quantization config
        model_class: Model class to load
        **kwargs: Additional arguments for model loading
        
    Returns:
        Loaded quantized model
    """
    # Load quantization config
    import json
    with open(config_path, 'r') as f:
        quant_config = json.load(f)
    
    # Load model
    if model_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')
    
    # Create model and load weights
    model = model_class(**kwargs)
    model.load_state_dict(state_dict)
    
    return QuantizedModel(model, quant_config)


def benchmark_quantized_model(
    model: QuantizedModel,
    test_data: torch.Tensor,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark quantized model performance
    
    Args:
        model: Quantized model
        test_data: Test data for benchmarking
        num_runs: Number of benchmark runs
        
    Returns:
        Benchmark results
    """
    import time
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_data)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(test_data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = test_data.size(0) / avg_time
    
    return {
        "avg_inference_time": avg_time,
        "throughput_samples_per_sec": throughput,
        "quantization_type": model.quantization_config.get("type", "unknown")
    }
