"""
LangChain Integration for RADON
OpenAI-compatible interface for LangChain applications
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Iterator
from pydantic import BaseModel, Field

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Generation, LLMResult
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class RADONLLM(LLM):
    """
    LangChain LLM wrapper for RADON
    OpenAI-compatible interface
    """
    
    model_name: str = "MagistrTheOne/RadonSAI-Pretrained"
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    device: str = "auto"
    torch_dtype: str = "float16"
    
    # Internal attributes
    _model: Optional[Any] = None
    _tokenizer: Optional[Any] = None
    
    class Config:
        """Configuration for Pydantic model"""
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load RADON model and tokenizer"""
        try:
            print(f"Loading RADON model: {self.model_name}")
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Determine torch dtype
            torch_dtype = torch.float16 if self.torch_dtype == "float16" else torch.float32
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            print(f"✅ RADON model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load RADON model: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        """Return LLM type"""
        return "radon"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using RADON"""
        
        if self._model is None or self._tokenizer is None:
            raise ValueError("Model not loaded. Call _load_model() first.")
        
        # Override parameters with kwargs
        max_length = kwargs.get("max_length", self.max_length)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        repetition_penalty = kwargs.get("repetition_penalty", self.repetition_penalty)
        do_sample = kwargs.get("do_sample", self.do_sample)
        
        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id
                )
            
            # Decode response
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove original prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Apply stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0]
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return f"Error: {str(e)}"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate text for multiple prompts"""
        
        generations = []
        
        for prompt in prompts:
            try:
                # Generate text
                text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
                
                # Create generation object
                generation = Generation(text=text)
                generations.append([generation])
                
            except Exception as e:
                # Handle errors
                error_generation = Generation(text=f"Error: {str(e)}")
                generations.append([error_generation])
        
        return LLMResult(generations=generations)
    
    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text generation"""
        
        if self._model is None or self._tokenizer is None:
            raise ValueError("Model not loaded")
        
        # Override parameters
        max_length = kwargs.get("max_length", self.max_length)
        temperature = kwargs.get("temperature", self.temperature)
        
        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            generated_tokens = []
            
            with torch.no_grad():
                for _ in range(max_length):
                    # Forward pass
                    outputs = self._model(**inputs)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Sample next token
                    if temperature > 0:
                        probs = torch.softmax(next_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Check for EOS
                    if next_token.item() == self._tokenizer.eos_token_id:
                        break
                    
                    # Add to generated tokens
                    generated_tokens.append(next_token.item())
                    
                    # Update inputs for next iteration
                    inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
                    
                    # Yield partial result
                    partial_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    yield partial_text
                    
                    # Check for stop sequences
                    if stop:
                        for stop_seq in stop:
                            if stop_seq in partial_text:
                                return
            
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get RADON model information"""
        if self._model is None:
            return {"error": "Model not loaded"}
        
        # Get model identity if available
        model_identity = {}
        if hasattr(self._model, 'get_model_identity'):
            model_identity = self._model.get_model_identity()
        
        return {
            "model_name": "radon",
            "model_type": "mistral",
            "device": str(next(self._model.parameters()).device),
            "dtype": str(next(self._model.parameters()).dtype),
            "parameters": sum(p.numel() for p in self._model.parameters()),
            "model_identity": model_identity,
            "langchain_type": self._llm_type
        }


class RADONChatModel:
    """
    Chat model interface for RADON
    OpenAI ChatCompletion compatible
    """
    
    def __init__(self, model_name: str = "MagistrTheOne/RadonSAI-Pretrained"):
        self.llm = RADONLLM(model_name=model_name)
    
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "radon",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Create chat completion using RADON"""
        
        # Extract prompt from messages
        prompt = self._format_messages(messages)
        
        # Generate response
        if stream:
            response_text = ""
            for partial in self.llm.stream(
                prompt, 
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p
            ):
                response_text = partial
        else:
            response_text = self.llm._call(
                prompt,
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
        
        # Format response
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single prompt"""
        formatted_prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
        
        formatted_prompt += "Assistant:"
        return formatted_prompt


# Example usage functions
def create_radon_llm(**kwargs) -> RADONLLM:
    """Create RADON LLM instance"""
    return RADONLLM(**kwargs)


def create_radon_chat_model(model_name: str = "MagistrTheOne/RadonSAI-Pretrained") -> RADONChatModel:
    """Create RADON chat model instance"""
    return RADONChatModel(model_name=model_name)


# LangChain integration examples
def example_usage():
    """Example usage of RADON with LangChain"""
    
    # Create RADON LLM
    radon_llm = create_radon_llm(
        max_length=100,
        temperature=0.7,
        top_p=0.9
    )
    
    # Basic text generation
    prompt = "Машинное обучение - это"
    response = radon_llm(prompt)
    print(f"Prompt: {prompt}")
    print(f"RADON Response: {response}")
    
    # Chat completion
    chat_model = create_radon_chat_model()
    
    messages = [
        {"role": "user", "content": "Объясни что такое нейронные сети"}
    ]
    
    completion = chat_model.create_chat_completion(
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )
    
    print(f"Chat Response: {completion['choices'][0]['message']['content']}")
    
    # Get model info
    model_info = radon_llm.get_model_info()
    print(f"Model Info: {model_info}")


if __name__ == "__main__":
    example_usage()
