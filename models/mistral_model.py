"""
Mistral-based transformer model implementation
Modern architecture with sliding window attention and efficient components
"""

import torch
import torch.nn as nn
import json
import os
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import ModelConfig
from .components import (
    RMSNorm, SwiGLU, RoPE, GroupedQueryAttention, 
    SlidingWindowAttention, PreNorm
)


class MistralDecoderLayer(nn.Module):
    """
    Single Mistral decoder layer
    
    Components:
    - Pre-normalization (RMSNorm before each sublayer)
    - Sliding Window Attention with GQA
    - SwiGLU feed-forward network
    - Residual connections
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Self-attention with sliding window
        self.self_attn = SlidingWindowAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            window_size=config.sliding_window,
            dropout=config.attention_dropout,
            rope_theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings
        )
        
        # Feed-forward network with SwiGLU
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
        
        # Pre-normalization layers
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
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
        Forward pass of Mistral decoder layer
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position indices for RoPE
            past_key_value: Cached key/value from previous steps
            use_cache: Whether to cache key/value
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (hidden_states, past_key_value, attention_weights)
        """
        residual = hidden_states
        
        # Self-attention with pre-normalization
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        hidden_states = residual + hidden_states
        
        # Feed-forward with pre-normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value, self_attn_weights


class MistralModel(PreTrainedModel):
    """
    Mistral-based transformer model
    
    Key features:
    - Sliding window attention (4096 tokens)
    - Grouped Query Attention (GQA)
    - RMSNorm instead of LayerNorm
    - SwiGLU activation
    - RoPE positional encoding
    - Pre-normalization architecture
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            MistralDecoderLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of Mistral model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_values: Cached key/value pairs from previous steps
            inputs_embeds: Input embeddings (alternative to input_ids)
            use_cache: Whether to cache key/value pairs
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_cache
        
        # Get input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Get position ids
        if position_ids is None:
            position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min
        
        # Initialize past key values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Forward through decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        hidden_states = inputs_embeds
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[2],)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return CausalLMOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLM(PreTrainedModel):
    """
    Mistral model with language modeling head
    
    Adds a linear layer on top of the base model for next token prediction
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        
        # Base model
        self.model = MistralModel(config)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Load system prompt for RADON identity
        self.system_prompt = self._load_system_prompt()
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def _load_system_prompt(self):
        """Load RADON system prompt for model identity"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'radon_system_prompt.json')
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
                return prompt_data.get('system_prompt', '')
            return ''
        except Exception:
            return ''
    
    def get_system_prompt(self):
        """Get RADON system prompt"""
        return self.system_prompt
    
    def get_model_identity(self):
        """Get RADON model identity information"""
        try:
            prompt_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'radon_system_prompt.json')
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
                return prompt_data.get('identity', {})
            return {}
        except Exception:
            return {}
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with language modeling head
        """
        return_dict = return_dict if return_dict is not None else self.config.use_cache
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
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
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
        """
        Generate text using the Mistral model
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # KV cache for efficiency
        past_key_values = None
        
        for _ in range(max_length - input_ids.size(1)):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits[:, -1, :]  # Last token logits
                past_key_values = outputs.past_key_values
                
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
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample from filtered distribution
                    probs = torch.softmax(logits, dim=-1)
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
