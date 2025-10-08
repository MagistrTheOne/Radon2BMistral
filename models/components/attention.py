"""
Grouped Query Attention (GQA) and Sliding Window Attention
Modern attention mechanisms for efficient inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from .positional import RoPE


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    
    Key innovation:
    - Multiple query heads share key/value heads
    - Reduces KV-cache memory usage
    - Speeds up inference 2-3x with minimal quality loss
    - Used in Llama 2, Mistral, PaLM-2
    
    Example: 32 query heads, 8 key/value heads (ratio 4:1)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 32768
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5
        
        # Ensure num_heads is divisible by num_kv_heads
        assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # RoPE for positional encoding
        self.rope = RoPE(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta
        )
    
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value heads to match query heads
        
        Args:
            x: Key/value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            n_rep: Number of repetitions (num_queries_per_kv)
            
        Returns:
            Repeated tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        if n_rep == 1:
            return x
        
        # Repeat along head dimension
        x = x[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass of Grouped Query Attention
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position indices for RoPE
            past_key_value: Cached key/value from previous steps
            use_cache: Whether to cache key/value
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, past_key_value, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if position_ids is not None:
            query_states, key_states = self.rope(query_states, key_states, seq_len)
        
        # Handle past key/value (for generation)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        # Repeat key/value heads to match query heads
        key_states = self._repeat_kv(key_states, self.num_queries_per_kv)
        value_states = self._repeat_kv(value_states, self.num_queries_per_kv)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        # Prepare past key/value for next step
        past_key_value = (key_states, value_states) if use_cache else None
        
        return attn_output, past_key_value, attn_weights if output_attentions else None


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention
    Key innovation in Mistral architecture
    
    Benefits:
    - Fixed attention window (e.g., 4096 tokens)
    - Linear complexity O(n) instead of O(n²)
    - Better for long sequences
    - Rolling buffer cache for efficiency
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        window_size: int = 4096,
        dropout: float = 0.0,
        bias: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 32768
    ):
        super().__init__()
        self.window_size = window_size
        
        # Use GQA as base attention mechanism
        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
            bias=bias,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len
        )
    
    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Create sliding window attention mask
        
        Args:
            seq_len: Sequence length
            device: Device for tensor
            dtype: Data type for tensor
            
        Returns:
            Attention mask of shape (1, 1, seq_len, seq_len)
        """
        if seq_len <= self.window_size:
            # No mask needed for short sequences
            return None
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
        
        # Apply sliding window constraint
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, :start] = 1  # Mask positions outside window
        
        # Convert to attention mask (0 = attend, -inf = mask)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass with sliding window attention
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Create sliding window mask
        sliding_mask = self._create_sliding_window_mask(seq_len, hidden_states.device, hidden_states.dtype)
        
        # Combine with provided attention mask
        if attention_mask is not None and sliding_mask is not None:
            attention_mask = attention_mask + sliding_mask
        elif sliding_mask is not None:
            attention_mask = sliding_mask
        
        # Apply base attention
        return self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
