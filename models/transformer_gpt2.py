"""
Custom GPT-2 Transformer Model
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple, Union, List

from .config import ModelConfig


class GPT2Attention(nn.Module):
    """Multi-head attention for GPT-2"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        # Scale attention weights
        attn_weights = attn_weights / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply head mask
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
            
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        batch_size, seq_len, embed_dim = hidden_states.size()
        
        # Linear projections
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=2)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, attn_weights


class GPT2MLP(nn.Module):
    """Feed-forward network for GPT-2"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    """Single transformer block for GPT-2"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GPT2MLP(config)
        
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs, attn_weights = self.attn(hidden_states, attention_mask, head_mask)
        hidden_states = attn_outputs + residual
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states, attn_weights


class CustomGPT2Model(PreTrainedModel):
    """Custom GPT-2 model implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer blocks
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_layers)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
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
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if position_ids is None:
            position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Transformer blocks
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            outputs = block(hidden_states, attention_mask, head_mask[i] if head_mask is not None else None)
            hidden_states = outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)
        
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_hidden_states:
                output = output + (all_hidden_states,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    
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
            
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - input_ids.size(1)):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(generated)
                logits = outputs.logits[:, -1, :]  # Last token logits
                
                if do_sample:
                    # Apply temperature
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
                    
                    # Sample from filtered distribution
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Add new token
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if eos_token_id is not None:
                    finished = finished | (next_token.squeeze(1) == eos_token_id)
                    if finished.all():
                        break
        
        return generated
