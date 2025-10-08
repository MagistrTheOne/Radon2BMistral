"""
API routes for text generation and model interaction
"""

import time
import uuid
from typing import Optional, Dict, Any, List
import torch

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field

from utils.logging_utils import log_request, log_error, log_generation, log_metrics
from models.hybrid_model import HybridTransformerModel
from tokenizer.custom_tokenizer import CustomTokenizer


# Import global variables
from .app import model, tokenizer, logger

router = APIRouter()


# Request/Response models
class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input prompt for text generation")
    max_length: int = Field(100, ge=1, le=32000, description="Maximum length of generated text (up to 32K for Mistral)")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="Top-p sampling parameter")
    do_sample: bool = Field(True, description="Whether to use sampling")
    num_return_sequences: int = Field(1, ge=1, le=5, description="Number of sequences to generate")
    sliding_window_size: Optional[int] = Field(None, ge=1024, le=8192, description="Sliding window size for Mistral attention")
    use_flash_attention: bool = Field(False, description="Whether to use Flash Attention (if available)")
    repetition_penalty: float = Field(1.0, ge=0.1, le=2.0, description="Repetition penalty for generation quality")
    user_id: Optional[str] = Field(None, description="Optional user ID for logging")
    request_id: Optional[str] = Field(None, description="Optional request ID")


class GenerationResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original prompt")
    generation_time: float = Field(..., description="Generation time in seconds")
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type used")
    request_id: str = Field(..., description="Request ID")
    model_identity: Optional[Dict[str, Any]] = Field(None, description="RADON model identity information")
    system_prompt: Optional[str] = Field(None, description="RADON system prompt")


class TokenizeRequest(BaseModel):
    """Request model for tokenization"""
    text: str = Field(..., description="Text to tokenize")
    add_special_tokens: bool = Field(True, description="Whether to add special tokens")
    return_tokens: bool = Field(True, description="Whether to return token strings")
    user_id: Optional[str] = Field(None, description="Optional user ID for logging")


class TokenizeResponse(BaseModel):
    """Response model for tokenization"""
    input_ids: List[int] = Field(..., description="Token IDs")
    tokens: Optional[List[str]] = Field(None, description="Token strings")
    text: str = Field(..., description="Original text")
    token_count: int = Field(..., description="Number of tokens")


class ModelSwitchRequest(BaseModel):
    """Request model for switching model architecture"""
    model_type: str = Field(..., description="Model type to switch to (mistral, gpt2, t5)")
    user_id: Optional[str] = Field(None, description="Optional user ID for logging")


class ModelSwitchResponse(BaseModel):
    """Response model for model switching"""
    success: bool = Field(..., description="Whether switch was successful")
    model_name: str = Field(..., description="Model name")
    current_model_type: str = Field(..., description="Current model type")
    message: str = Field(..., description="Status message")


@router.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate text using the loaded model"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    
    # Generate request ID if not provided
    request_id = request.request_id or str(uuid.uuid4())
    
    try:
        # Log request
        if logger:
            log_request(
                logger,
                request_data=request.dict(),
                endpoint="/generate",
                user_id=request.user_id,
                request_id=request_id
            )
        
        # Start timing
        start_time = time.time()
        
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            # Prepare generation parameters
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "do_sample": request.do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            
            # Add Mistral-specific parameters if available
            if hasattr(model, 'model') and hasattr(model.model, 'config'):
                if hasattr(model.model.config, 'sliding_window') and request.sliding_window_size:
                    # Note: sliding_window is set in model config, not generation
                    pass
                
                # Add repetition penalty if supported
                if request.repetition_penalty != 1.0:
                    generation_kwargs["repetition_penalty"] = request.repetition_penalty
            
            outputs = model.generate(**generation_kwargs)
        
        # Decode generated text
        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Remove original prompt from generated text
        if generated_text.startswith(request.prompt):
            generated_text = generated_text[len(request.prompt):].strip()
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Log generation
        if logger:
            log_generation(
                logger,
                prompt=request.prompt,
                generated_text=generated_text,
                generation_time=generation_time,
                model_name=model.model_type,
                user_id=request.user_id,
                request_id=request_id,
                generation_params=request.dict()
            )
        
        # Log metrics
        if logger:
            metrics = {
                "generation_time": generation_time,
                "input_length": len(request.prompt),
                "output_length": len(generated_text),
                "tokens_per_second": len(generated_text.split()) / generation_time if generation_time > 0 else 0
            }
            log_metrics(logger, metrics, model_name=model.model_type, user_id=request.user_id, request_id=request_id)
        
        # Get RADON identity information if available
        model_identity = None
        system_prompt = None
        if hasattr(model, 'get_model_identity'):
            model_identity = model.get_model_identity()
        if hasattr(model, 'get_system_prompt'):
            system_prompt = model.get_system_prompt()
        
        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            generation_time=generation_time,
            model_name=model.config.model_name,
            model_type=model.model_type,
            request_id=request_id,
            model_identity=model_identity,
            system_prompt=system_prompt
        )
        
    except Exception as e:
        if logger:
            log_error(logger, e, user_id=request.user_id, request_id=request_id)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize_text(request: TokenizeRequest):
    """Tokenize text using the loaded tokenizer"""
    
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    
    try:
        # Log request
        if logger:
            log_request(
                logger,
                request_data=request.dict(),
                endpoint="/tokenize",
                user_id=request.user_id
            )
        
        # Tokenize text
        inputs = tokenizer(
            request.text,
            add_special_tokens=request.add_special_tokens,
            return_tensors="pt"
        )
        
        # Get token IDs
        input_ids = inputs["input_ids"].squeeze().tolist()
        
        # Get token strings if requested
        tokens = None
        if request.return_tokens:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Log metrics
        if logger:
            metrics = {
                "text_length": len(request.text),
                "token_count": len(input_ids),
                "tokens_per_character": len(input_ids) / len(request.text) if len(request.text) > 0 else 0
            }
            log_metrics(logger, metrics, user_id=request.user_id)
        
        return TokenizeResponse(
            input_ids=input_ids,
            tokens=tokens,
            text=request.text,
            token_count=len(input_ids)
        )
        
    except Exception as e:
        if logger:
            log_error(logger, e, user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")


@router.post("/model/switch", response_model=ModelSwitchResponse)
async def switch_model_architecture(request: ModelSwitchRequest):
    """Switch model architecture (GPT-2 <-> T5)"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if model.config.model_type != "hybrid":
        raise HTTPException(status_code=400, detail="Model switching only available in hybrid mode")
    
    try:
        # Log request
        if logger:
            log_request(
                logger,
                request_data=request.dict(),
                endpoint="/model/switch",
                user_id=request.user_id
            )
        
        # Switch model architecture
        if request.model_type == "mistral":
            model.switch_to_mistral()
            message = "Switched to Mistral architecture"
        elif request.model_type == "gpt2":
            model.switch_to_gpt2()
            message = "Switched to GPT-2 architecture"
        elif request.model_type == "t5":
            model.switch_to_t5()
            message = "Switched to T5 architecture"
        else:
            raise HTTPException(status_code=400, detail="Invalid model type. Use 'mistral', 'gpt2' or 't5'")
        
        # Log metrics
        if logger:
            metrics = {
                "new_model_type": request.model_type,
                "previous_model_type": model.model_type
            }
            log_metrics(logger, metrics, model_name=model.model_type, user_id=request.user_id)
        
        return ModelSwitchResponse(
            success=True,
            model_name=model.config.model_name,
            current_model_type=model.model_type,
            message=message
        )
        
    except Exception as e:
        if logger:
            log_error(logger, e, user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Model switching failed: {str(e)}")


@router.get("/model/info")
async def get_model_info():
    """Get detailed model information"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get model information
        model_info = model.get_model_info()
        
        # Add tokenizer information
        tokenizer_info = {
            "vocab_size": tokenizer.vocab_size if tokenizer else None,
            "special_tokens": {
                "unk_token": tokenizer.unk_token if tokenizer else None,
                "bos_token": tokenizer.bos_token if tokenizer else None,
                "eos_token": tokenizer.eos_token if tokenizer else None,
                "pad_token": tokenizer.pad_token if tokenizer else None,
            }
        } if tokenizer else None
        
        # Get device information
        device_info = {
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype)
        }
        
        return {
            "model_info": model_info,
            "tokenizer_info": tokenizer_info,
            "device_info": device_info,
            "status": "loaded"
        }
        
    except Exception as e:
        if logger:
            log_error(logger, e)
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.get("/model/benchmark")
async def benchmark_model():
    """Benchmark model performance"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    
    try:
        from utils.model_utils import benchmark_model
        
        # Test prompts
        test_prompts = [
            "Hello, world!",
            "The quick brown fox",
            "In a galaxy far, far away",
            "Машинное обучение",
            "Привет, мир!"
        ]
        
        # Run benchmark
        device = str(next(model.parameters()).device)
        results = benchmark_model(model, tokenizer, test_prompts, device=device, num_runs=3)
        
        # Log metrics
        if logger:
            log_metrics(logger, results, model_name=model.model_type)
        
        return {
            "benchmark_results": results,
            "model_type": model.model_type,
            "device": device
        }
        
    except Exception as e:
        if logger:
            log_error(logger, e)
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.post("/model/reload")
async def reload_model():
    """Reload model and tokenizer"""
    
    try:
        # Log request
        if logger:
            log_request(logger, {"action": "reload"}, endpoint="/model/reload")
        
        # Reload model and tokenizer
        from .app import load_model_and_tokenizer
        await load_model_and_tokenizer()
        
        if logger:
            log_metrics(logger, {"action": "model_reloaded"}, model_name=model.model_type if model else None)
        
        return {
            "success": True,
            "message": "Model and tokenizer reloaded successfully",
            "model_type": model.model_type if model else None
        }
        
    except Exception as e:
        if logger:
            log_error(logger, e)
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")
