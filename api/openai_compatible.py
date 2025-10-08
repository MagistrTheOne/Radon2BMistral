"""
OpenAI-Compatible API for RADON
Drop-in replacement for OpenAI API
"""

import os
import time
import uuid
import json
from typing import Dict, List, Any, Optional, Union, Iterator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Request/Response Models
class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""
    model: str = Field(default="radon", description="Model name")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: Optional[int] = Field(default=100, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(default=50, description="Top-k sampling parameter")
    stream: Optional[bool] = Field(default=False, description="Whether to stream response")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    user: Optional[str] = Field(default=None, description="User identifier")


class CompletionRequest(BaseModel):
    """Text completion request model"""
    model: str = Field(default="radon", description="Model name")
    prompt: str = Field(..., description="Input prompt")
    max_tokens: Optional[int] = Field(default=100, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(default=50, description="Top-k sampling parameter")
    stream: Optional[bool] = Field(default=False, description="Whether to stream response")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    user: Optional[str] = Field(default=None, description="User identifier")


class ChatChoice(BaseModel):
    """Chat choice model"""
    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Generated message")
    finish_reason: str = Field(default="stop", description="Finish reason")


class CompletionChoice(BaseModel):
    """Completion choice model"""
    index: int = Field(..., description="Choice index")
    text: str = Field(..., description="Generated text")
    finish_reason: str = Field(default="stop", description="Finish reason")


class Usage(BaseModel):
    """Usage statistics model"""
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")
    total_tokens: int = Field(..., description="Total number of tokens")


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""
    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[ChatChoice] = Field(..., description="Generated choices")
    usage: Usage = Field(..., description="Usage statistics")


class CompletionResponse(BaseModel):
    """Text completion response model"""
    id: str = Field(..., description="Response ID")
    object: str = Field(default="text_completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model name")
    choices: List[CompletionChoice] = Field(..., description="Generated choices")
    usage: Usage = Field(..., description="Usage statistics")


class ModelInfo(BaseModel):
    """Model information model"""
    id: str = Field(..., description="Model ID")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    owned_by: str = Field(..., description="Model owner")
    permission: List[Dict] = Field(default=[], description="Model permissions")
    root: str = Field(default="radon", description="Root model")
    parent: Optional[str] = Field(default=None, description="Parent model")


class ModelsResponse(BaseModel):
    """Models list response model"""
    object: str = Field(default="list", description="Object type")
    data: List[ModelInfo] = Field(..., description="Available models")


# RADON OpenAI-Compatible API
class RADONOpenAI:
    """OpenAI-compatible API for RADON"""
    
    def __init__(self, model_name: str = "MagistrTheOne/RadonSAI-Pretrained"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_model()
    
    def _load_model(self):
        """Load RADON model and tokenizer"""
        try:
            print(f"Loading RADON model: {self.model_name}")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ RADON model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load RADON model: {e}")
            raise
    
    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a single prompt"""
        formatted_prompt = ""
        
        for message in messages:
            role = message.role
            content = message.content
            
            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
        
        formatted_prompt += "Assistant:"
        return formatted_prompt
    
    def _generate_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop: Optional[Union[str, List[str]]] = None
    ) -> str:
        """Generate text using RADON"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove original prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Apply stop sequences
            if stop:
                if isinstance(stop, str):
                    stop = [stop]
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0]
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _stream_generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop: Optional[Union[str, List[str]]] = None
    ) -> Iterator[str]:
        """Stream text generation"""
        
        if self.model is None or self.tokenizer is None:
            yield f"Error: Model not loaded"
            return
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            generated_tokens = []
            
            with torch.no_grad():
                for _ in range(max_tokens):
                    # Forward pass
                    outputs = self.model(**inputs)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Sample next token
                    if temperature > 0:
                        probs = torch.softmax(next_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Check for EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Add to generated tokens
                    generated_tokens.append(next_token.item())
                    
                    # Update inputs for next iteration
                    inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
                    
                    # Yield partial result
                    partial_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Check for stop sequences
                    if stop:
                        if isinstance(stop, str):
                            stop = [stop]
                        for stop_seq in stop:
                            if stop_seq in partial_text:
                                return
                    
                    yield partial_text
            
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def create_chat_completion(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        user: Optional[str] = None
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionResponse]]:
        """Create chat completion"""
        
        # Format messages into prompt
        prompt = self._format_messages(messages)
        
        # Generate response
        if stream:
            return self._stream_chat_completion(
                prompt, messages, max_tokens, temperature, top_p, top_k, stop
            )
        else:
            response_text = self._generate_text(
                prompt, max_tokens, temperature, top_p, top_k, stop
            )
            
            # Create response
            completion_id = f"chatcmpl-{uuid.uuid4()}"
            
            return ChatCompletionResponse(
                id=completion_id,
                created=int(time.time()),
                model="radon",
                choices=[ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )],
                usage=Usage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(response_text.split()),
                    total_tokens=len(prompt.split()) + len(response_text.split())
                )
            )
    
    def _stream_chat_completion(
        self,
        prompt: str,
        messages: List[ChatMessage],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop: Optional[Union[str, List[str]]]
    ) -> Iterator[ChatCompletionResponse]:
        """Stream chat completion"""
        
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        
        for partial_text in self._stream_generate(
            prompt, max_tokens, temperature, top_p, top_k, stop
        ):
            yield ChatCompletionResponse(
                id=completion_id,
                created=int(time.time()),
                model="radon",
                choices=[ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=partial_text),
                    finish_reason="stop" if partial_text else "length"
                )],
                usage=Usage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(partial_text.split()),
                    total_tokens=len(prompt.split()) + len(partial_text.split())
                )
            )
    
    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        user: Optional[str] = None
    ) -> Union[CompletionResponse, Iterator[CompletionResponse]]:
        """Create text completion"""
        
        if stream:
            return self._stream_completion(
                prompt, max_tokens, temperature, top_p, top_k, stop
            )
        else:
            response_text = self._generate_text(
                prompt, max_tokens, temperature, top_p, top_k, stop
            )
            
            # Create response
            completion_id = f"completion-{uuid.uuid4()}"
            
            return CompletionResponse(
                id=completion_id,
                created=int(time.time()),
                model="radon",
                choices=[CompletionChoice(
                    index=0,
                    text=response_text,
                    finish_reason="stop"
                )],
                usage=Usage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(response_text.split()),
                    total_tokens=len(prompt.split()) + len(response_text.split())
                )
            )
    
    def _stream_completion(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stop: Optional[Union[str, List[str]]]
    ) -> Iterator[CompletionResponse]:
        """Stream text completion"""
        
        completion_id = f"completion-{uuid.uuid4()}"
        
        for partial_text in self._stream_generate(
            prompt, max_tokens, temperature, top_p, top_k, stop
        ):
            yield CompletionResponse(
                id=completion_id,
                created=int(time.time()),
                model="radon",
                choices=[CompletionChoice(
                    index=0,
                    text=partial_text,
                    finish_reason="stop" if partial_text else "length"
                )],
                usage=Usage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(partial_text.split()),
                    total_tokens=len(prompt.split()) + len(partial_text.split())
                )
            )
    
    def list_models(self) -> ModelsResponse:
        """List available models"""
        
        models = [
            ModelInfo(
                id="radon",
                created=int(time.time()),
                owned_by="MagistrTheOne",
                permission=[],
                root="radon"
            ),
            ModelInfo(
                id="radon-2b",
                created=int(time.time()),
                owned_by="MagistrTheOne",
                permission=[],
                root="radon",
                parent="radon"
            ),
            ModelInfo(
                id="radon-7b",
                created=int(time.time()),
                owned_by="MagistrTheOne",
                permission=[],
                root="radon",
                parent="radon"
            )
        ]
        
        return ModelsResponse(data=models)


# FastAPI Application
app = FastAPI(
    title="RADON OpenAI-Compatible API",
    description="OpenAI-compatible API for RADON Mistral-based transformer",
    version="1.0.0"
)

# Initialize RADON
radon_api = RADONOpenAI()


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return radon_api.list_models()


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion"""
    try:
        response = radon_api.create_chat_completion(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream=request.stream,
            stop=request.stop,
            user=request.user
        )
        
        if request.stream:
            def generate():
                for chunk in response:
                    yield f"data: {chunk.json()}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            return response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Create text completion"""
    try:
        response = radon_api.create_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream=request.stream,
            stop=request.stop,
            user=request.user
        )
        
        if request.stream:
            def generate():
                for chunk in response:
                    yield f"data: {chunk.json()}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            return response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "radon",
        "device": radon_api.device,
        "timestamp": int(time.time())
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RADON OpenAI-Compatible API",
        "version": "1.0.0",
        "creator": "MagistrTheOne",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "completions": "/v1/completions",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
