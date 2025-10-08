#!/usr/bin/env python3
"""
RADON Model Inference Script for SageMaker
"""

import os
import json
import torch
import logging
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RADONInference:
    """RADON Model Inference Handler"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def model_fn(self, model_dir: str):
        """Load model for inference"""
        logger.info(f"Loading model from {model_dir}...")
        
        try:
            # Import RADON model
            import sys
            sys.path.append('/opt/ml/code')
            
            from models.mistral_model import MistralForCausalLM
            from models.config import ModelConfig
            
            # Load config
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Create model config
            model_config = ModelConfig(**config_data)
            
            # Initialize model
            self.model = MistralForCausalLM(model_config)
            
            # Load weights
            weights_path = os.path.join(model_dir, "pytorch_model.bin")
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("Loaded model weights")
            else:
                logger.warning("No weights found, using random initialization")
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("MagistrTheOne/RadonSAI")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("✅ Model loaded successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def input_fn(self, request_body: str, content_type: str = "application/json"):
        """Parse input data"""
        if content_type == "application/json":
            return json.loads(request_body)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    
    def predict_fn(self, input_data: Dict[str, Any], model):
        """Generate predictions"""
        try:
            # Extract parameters
            prompt = input_data.get("prompt", "Hello!")
            max_length = input_data.get("max_length", 100)
            temperature = input_data.get("temperature", 0.7)
            top_p = input_data.get("top_p", 0.9)
            top_k = input_data.get("top_k", 50)
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Prepare response
            response = {
                "generated_text": generated_text,
                "prompt": prompt,
                "model_name": "RADON",
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "error": str(e),
                "generated_text": "",
                "prompt": input_data.get("prompt", "")
            }
    
    def output_fn(self, prediction: Dict[str, Any], accept: str = "application/json"):
        """Format output"""
        if accept == "application/json":
            return json.dumps(prediction, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported accept type: {accept}")

# Global inference handler
inference_handler = RADONInference()

def model_fn(model_dir: str):
    """SageMaker model loading function"""
    return inference_handler.model_fn(model_dir)

def input_fn(request_body: str, content_type: str = "application/json"):
    """SageMaker input parsing function"""
    return inference_handler.input_fn(request_body, content_type)

def predict_fn(input_data: Dict[str, Any], model):
    """SageMaker prediction function"""
    return inference_handler.predict_fn(input_data, model)

def output_fn(prediction: Dict[str, Any], accept: str = "application/json"):
    """SageMaker output formatting function"""
    return inference_handler.output_fn(prediction, accept)
