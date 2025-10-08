"""
Custom T5 Transformer Model
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Tuple, Union, List

from .config import ModelConfig


class T5LayerNorm(nn.Module):
    """T5-style layer normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class T5Attention(nn.Module):
    """Multi-head attention for T5"""
    
    def __init__(self, config: ModelConfig, has_relative_attention_bias: bool = False):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.key_value_proj_dim = config.hidden_size // config.num_attention_heads
        self.n_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        
        # Linear projections
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        
        # Relative attention bias
        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
    
    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """Compute relative position buckets"""
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / torch.log(torch.tensor(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
    def compute_bias(self, query_length, key_length):
        """Compute attention bias for relative positions"""
        if not self.has_relative_attention_bias:
            return None
        
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values
    
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        use_cache=False,
    ):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Linear projections
        query_states = self.q(hidden_states)
        if key_value_states is None:
            key_states = self.k(hidden_states)
            value_states = self.v(hidden_states)
        else:
            key_states = self.k(key_value_states)
            value_states = self.v(key_value_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim)
        key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)
        value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)
        
        # Transpose for attention computation
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        
        # Add position bias
        if position_bias is None:
            position_bias = self.compute_bias(seq_length, key_states.size(2))
        
        if position_bias is not None:
            scores = scores + position_bias
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.inner_dim)
        attn_output = self.o(attn_output)
        
        return attn_output, attn_weights, position_bias


class T5LayerSelfAttention(nn.Module):
    """T5 self-attention layer"""
    
    def __init__(self, config: ModelConfig, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        return (hidden_states,) + attention_output[1:]


class T5LayerCrossAttention(nn.Module):
    """T5 cross-attention layer"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        return (hidden_states,) + attention_output[1:]


class T5LayerFF(nn.Module):
    """T5 feed-forward layer"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.DenseReluDense = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        self.layer_norm = T5LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states):
        forwarded_states = self.DenseReluDense(hidden_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Block(nn.Module):
    """T5 transformer block"""
    
    def __init__(self, config: ModelConfig, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias))
        self.layer.append(T5LayerFF(config))
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        for i, layer_module in enumerate(self.layer):
            if i == 0:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                    layer_head_mask=layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            else:
                layer_outputs = layer_module(layer_outputs[0])
        
        return layer_outputs


class CustomT5Model(PreTrainedModel):
    """Custom T5 model implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        
        # Shared embeddings
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Encoder
        self.encoder = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=(i == 0))
            for i in range(config.num_layers)
        ])
        self.encoder_norm = T5LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Decoder
        self.decoder = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=(i == 0))
            for i in range(config.num_layers)
        ])
        self.decoder_norm = T5LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    
    def get_input_embeddings(self):
        return self.shared
    
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Encoder
        encoder_outputs = self._encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        # Decoder
        decoder_outputs = self._decode(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        
        # Language modeling head
        logits = self.lm_head(decoder_outputs[0])
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values if use_cache else None,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def _encode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        """Encode input sequence"""
        if input_ids is None:
            return None
        
        # Embeddings
        hidden_states = self.shared(input_ids)
        
        # Encoder layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.encoder:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        hidden_states = self.encoder_norm(hidden_states)
        
        return (hidden_states, all_hidden_states, all_attentions)
    
    def _decode(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ):
        """Decode sequence"""
        if decoder_input_ids is None:
            return None
        
        # Embeddings
        hidden_states = self.shared(decoder_input_ids)
        
        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.decoder:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=decoder_attention_mask,
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        hidden_states = self.decoder_norm(hidden_states)
        
        return (hidden_states, all_hidden_states, all_attentions)
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate text using the model"""
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        # Encode input
        encoder_outputs = self._encode(input_ids=input_ids)
        encoder_hidden_states = encoder_outputs[0]
        
        # Initialize decoder
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Start with BOS token or first token
        decoder_input_ids = torch.full((batch_size, 1), pad_token_id, device=device)
        if self.config.bos_token_id is not None:
            decoder_input_ids[:, 0] = self.config.bos_token_id
        
        # Generate tokens
        for _ in range(max_length - 1):
            # Decode
            decoder_outputs = self._decode(
                decoder_input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
            )
            
            # Get logits for last token
            logits = self.lm_head(decoder_outputs[0][:, -1, :])
            
            # Sample next token
            if do_sample:
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Add new token
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return decoder_input_ids
