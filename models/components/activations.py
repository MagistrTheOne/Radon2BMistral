"""
Modern activation functions for Mistral+Llama3 architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Swish(nn.Module):
    """
    Swish activation function: x * sigmoid(x)
    Used in SwiGLU and other modern architectures
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU activation: Swish-Gated Linear Unit
    
    Formula: Swish(xW + b) ⊙ (xV + c)
    Where ⊙ is element-wise multiplication
    
    More effective than GELU/ReLU:
    - Better gradient flow
    - Used in Llama, PaLM, Mistral
    - Gated mechanism improves expressiveness
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Gate projection (Swish branch)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        # Up projection (linear branch)  
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        # Down projection (output)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        # Swish activation
        self.activation = Swish()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Gate branch: Swish(xW + b)
        gate = self.activation(self.gate_proj(x))
        
        # Up branch: xV + c
        up = self.up_proj(x)
        
        # Element-wise multiplication and down projection
        return self.down_proj(gate * up)


class SiLU(nn.Module):
    """
    Sigmoid Linear Unit (SiLU) - same as Swish
    Alternative name used in some papers
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit
    Standard activation in many transformer models
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


def get_activation_fn(activation: str) -> nn.Module:
    """
    Get activation function by name
    
    Args:
        activation: Name of activation function
        
    Returns:
        Activation module
    """
    activations = {
        "swish": Swish(),
        "silu": SiLU(),
        "swiglu": SwiGLU,  # Class, not instance
        "gelu": GELU(),
        "relu": nn.ReLU(),
    }
    
    if activation not in activations:
        raise ValueError(f"Unknown activation: {activation}")
    
    return activations[activation]
