"""
Configuration classes for RADON models
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Base configuration for transformer models"""
    
    # Model architecture
    model_name: str = "radon"    # Model name
    model_type: str = "mistral"  # mistral, gpt2, t5, hybrid
    vocab_size: int = 32000
    hidden_size: int = 2048  # Updated for Mistral-2B
    num_layers: int = 24     # Updated for Mistral-2B
    num_attention_heads: int = 32
    num_kv_heads: int = 8    # GQA: 32 query heads, 8 key/value heads
    intermediate_size: int = 5632  # Updated for Mistral-2B
    max_position_embeddings: int = 32768  # Support for 32K context
    
    # Mistral-specific parameters
    sliding_window: int = 4096  # Sliding window attention size
    rope_theta: float = 10000.0  # RoPE base frequency
    rms_norm_eps: float = 1e-6   # RMSNorm epsilon
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Activation and normalization
    activation_function: str = "silu"  # SwiGLU uses SiLU
    layer_norm_eps: float = 1e-6       # Updated for RMSNorm
    
    # Initialization
    initializer_range: float = 0.02
    
    # Generation
    use_cache: bool = True
    torch_dtype: str = "float32"
    
    # Additional parameters
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Output configuration
    output_attentions: bool = False
    output_hidden_states: bool = False
    
    @classmethod
    def from_json(cls, config_path: str) -> "ModelConfig":
        """Load configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def update(self, **kwargs) -> "ModelConfig":
        """Update configuration with new values"""
        config_dict = asdict(self)
        config_dict.update(kwargs)
        return ModelConfig(**config_dict)


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # Training parameters
    learning_rate: float = 5e-4
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Optimization
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Logging and saving
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    output_dir: str = "./outputs"
    
    # System
    seed: int = 42
    fp16: bool = False
    dataloader_num_workers: int = 2
    
    @classmethod
    def from_json(cls, config_path: str) -> "TrainingConfig":
        """Load training configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, config_path: str) -> None:
        """Save training configuration to JSON file"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
