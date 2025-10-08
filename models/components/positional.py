"""
Rotary Position Embeddings (RoPE) implementation
Relative positional encoding for better long context handling
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RoPE(nn.Module):
    """
    Rotary Position Embeddings (RoPE)
    
    Key advantages:
    - Relative positional encoding
    - Better extrapolation to longer sequences
    - Used in Llama, Mistral, PaLM
    - Scales well to 8K-32K context lengths
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        theta: float = 10000.0,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_factor = scaling_factor
        
        # Precompute frequency matrix
        self._precompute_freqs_cis()
    
    def _precompute_freqs_cis(self):
        """Precompute frequency matrix for efficiency"""
        # Create frequency matrix
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2)[:self.dim // 2].float() / self.dim))
        
        # Create position matrix
        t = torch.arange(self.max_seq_len, device=freqs.device)
        
        # Outer product to get all combinations
        freqs = torch.outer(t, freqs).float()
        
        # Create complex numbers: cos + i*sin
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs))
    
    def _get_freqs_cis(self, seq_len: int) -> torch.Tensor:
        """Get frequency matrix for given sequence length"""
        if seq_len > self.max_seq_len:
            # Extrapolate for longer sequences
            freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2)[:self.dim // 2].float() / self.dim))
            t = torch.arange(seq_len, device=freqs.device)
            freqs = torch.outer(t, freqs).float()
            return torch.polar(torch.ones_like(freqs), freqs)
        
        return self.freqs_cis[:seq_len]
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors
        
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            seq_len: Sequence length (if None, inferred from q)
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        if seq_len is None:
            seq_len = q.shape[-2]
        
        # Get frequency matrix
        freqs_cis = self._get_freqs_cis(seq_len)
        
        # Reshape for complex multiplication
        q_reshaped = q.float().reshape(*q.shape[:-1], -1, 2)
        k_reshaped = k.float().reshape(*k.shape[:-1], -1, 2)
        
        # Convert to complex numbers
        q_complex = torch.view_as_complex(q_reshaped)
        k_complex = torch.view_as_complex(k_reshaped)
        
        # Apply rotation
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        q_rotated = q_complex * freqs_cis
        k_rotated = k_complex * freqs_cis
        
        # Convert back to real numbers
        q_rotated = torch.view_as_real(q_rotated).flatten(-2)
        k_rotated = torch.view_as_real(k_rotated).flatten(-2)
        
        return q_rotated.to(q.dtype), k_rotated.to(k.dtype)


class RoPE1D(nn.Module):
    """
    1D RoPE implementation (simpler version)
    Used when complex numbers are not available
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        theta: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute cos and sin matrices
        self._precompute_rotary_matrices()
    
    def _precompute_rotary_matrices(self):
        """Precompute rotation matrices"""
        # Create frequency matrix
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Create position matrix
        t = torch.arange(self.max_seq_len)
        
        # Outer product
        freqs = torch.outer(t, freqs)
        
        # Separate cos and sin
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 1D RoPE to query and key tensors
        """
        if seq_len is None:
            seq_len = q.shape[-2]
        
        # Get rotation matrices
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        
        # Reshape for rotation
        q_reshaped = q.reshape(*q.shape[:-1], -1, 2)
        k_reshaped = k.reshape(*k.shape[:-1], -1, 2)
        
        # Apply rotation
        cos = cos.unsqueeze(0).unsqueeze(0)  # Add batch and head dims
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        q_rotated = torch.cat([
            q_reshaped[..., 0] * cos - q_reshaped[..., 1] * sin,
            q_reshaped[..., 0] * sin + q_reshaped[..., 1] * cos
        ], dim=-1)
        
        k_rotated = torch.cat([
            k_reshaped[..., 0] * cos - k_reshaped[..., 1] * sin,
            k_reshaped[..., 0] * sin + k_reshaped[..., 1] * cos
        ], dim=-1)
        
        return q_rotated, k_rotated
