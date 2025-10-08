"""
Hybrid Transformer Model with Mistral, GPT-2 and T5 support
"""

import torch
from typing import Optional, Union, Dict, Any
from transformers import PreTrainedModel

from .config import ModelConfig
from .transformer_gpt2 import CustomGPT2Model
from .transformer_t5 import CustomT5Model
from .mistral_model import MistralForCausalLM


class HybridTransformerModel(PreTrainedModel):
    """Hybrid model that can switch between Mistral, GPT-2 and T5 architectures"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.model_type = config.model_type
        
        # Initialize the appropriate model based on config
        if config.model_type == "mistral":
            self.model = MistralForCausalLM(config)
        elif config.model_type == "gpt2":
            self.model = CustomGPT2Model(config)
        elif config.model_type == "t5":
            self.model = CustomT5Model(config)
        elif config.model_type == "hybrid":
            # For hybrid mode, default to Mistral but allow switching
            self.model = MistralForCausalLM(config)
            self._mistral_model = MistralForCausalLM(config)
            self._gpt2_model = CustomGPT2Model(config)
            self._t5_model = CustomT5Model(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    def switch_to_mistral(self):
        """Switch to Mistral architecture"""
        if self.config.model_type == "hybrid":
            self.model = self._mistral_model
            self.model_type = "mistral"
        else:
            raise ValueError("Model switching only available in hybrid mode")
    
    def switch_to_gpt2(self):
        """Switch to GPT-2 architecture"""
        if self.config.model_type == "hybrid":
            self.model = self._gpt2_model
            self.model_type = "gpt2"
        else:
            raise ValueError("Model switching only available in hybrid mode")
    
    def switch_to_t5(self):
        """Switch to T5 architecture"""
        if self.config.model_type == "hybrid":
            self.model = self._t5_model
            self.model_type = "t5"
        else:
            raise ValueError("Model switching only available in hybrid mode")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.config.model_name,
            "model_type": self.model_type,
            "config": self.config.to_dict(),
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
    
    def forward(self, *args, **kwargs):
        """Forward pass through the current model"""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate text using the current model"""
        return self.model.generate(*args, **kwargs)
    
    def get_input_embeddings(self):
        """Get input embeddings from the current model"""
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, new_embeddings):
        """Set input embeddings for the current model"""
        self.model.set_input_embeddings(new_embeddings)
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model and configuration"""
        # Save the current model
        self.model.save_pretrained(save_directory, **kwargs)
        
        # Save hybrid configuration if applicable
        if self.config.model_type == "hybrid":
            import json
            import os
            
            hybrid_config = {
                "model_type": self.model_type,
                "base_config": self.config.to_dict(),
                "available_models": ["gpt2", "t5"]
            }
            
            config_path = os.path.join(save_directory, "hybrid_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(hybrid_config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load a hybrid model from pretrained checkpoint"""
        import json
        import os
        
        # Load base configuration
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config = ModelConfig(**config_dict)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Check for hybrid configuration
        hybrid_config_path = os.path.join(pretrained_model_name_or_path, "hybrid_config.json")
        if os.path.exists(hybrid_config_path):
            with open(hybrid_config_path, 'r', encoding='utf-8') as f:
                hybrid_config = json.load(f)
            model_type = hybrid_config.get("model_type", config.model_type)
        else:
            model_type = config.model_type
        
        # Create model instance
        model = cls(config)
        
        # Load the appropriate model weights
        if model_type == "mistral":
            model.model = MistralForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif model_type == "gpt2":
            model.model = CustomGPT2Model.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif model_type == "t5":
            model.model = CustomT5Model.from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            raise ValueError(f"Unsupported model type for loading: {model_type}")
        
        model.model_type = model_type
        return model
    
    def train(self, mode: bool = True):
        """Set training mode for the model"""
        super().train(mode)
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode for the model"""
        super().eval()
        self.model.eval()
        return self
    
    def to(self, device):
        """Move model to device"""
        super().to(device)
        self.model.to(device)
        return self
    
    def cuda(self, device=None):
        """Move model to CUDA"""
        super().cuda(device)
        self.model.cuda(device)
        return self
    
    def cpu(self):
        """Move model to CPU"""
        super().cpu()
        self.model.cpu()
        return self
    
    def half(self):
        """Convert model to half precision"""
        super().half()
        self.model.half()
        return self
    
    def float(self):
        """Convert model to float precision"""
        super().float()
        self.model.float()
        return self
    
    def double(self):
        """Convert model to double precision"""
        super().double()
        self.model.double()
        return self
