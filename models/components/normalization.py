"""
RMSNorm (Root Mean Square Layer Normalization) implementation
More efficient than LayerNorm - no mean computation, only RMS
"""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    Formula: x * rsqrt(mean(x²) + ε) * weight
    
    More efficient than LayerNorm:
    - No mean computation (only RMS)
    - Better numerical stability
    - Used in Llama, Mistral, PaLM
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Normalized tensor of same shape
        """
        input_dtype = hidden_states.dtype
        
        # Compute RMS: sqrt(mean(x²))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        
        # Apply learnable weight if enabled
        if self.elementwise_affine:
            hidden_states = hidden_states.to(self.weight.dtype) * self.weight
            hidden_states = hidden_states.to(input_dtype)
        
        return hidden_states


class PreNorm(nn.Module):
    """
    Pre-normalization wrapper (RMSNorm before sublayer)
    Used in Mistral/Llama architecture
    """
    
    def __init__(
        self,
        hidden_size: int,
        sublayer: nn.Module,
        eps: float = 1e-6
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.sublayer = sublayer
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Apply normalization before sublayer
        
        Args:
            x: Input tensor
            *args, **kwargs: Arguments for sublayer
            
        Returns:
            Output from sublayer
        """
        return self.sublayer(self.norm(x), *args, **kwargs)
