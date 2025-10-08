"""
FastAPI application for RADON Custom Transformer
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import torch

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import router as api_router
from .vk_webhook import router as vk_router
from utils.logging_utils import setup_logger, log_request, log_error
from models.hybrid_model import HybridTransformerModel
from models.config import ModelConfig
from tokenizer.custom_tokenizer import CustomTokenizer


# Global variables for model and tokenizer
model: Optional[HybridTransformerModel] = None
tokenizer: Optional[CustomTokenizer] = None
logger: Optional[logging.Logger] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model, tokenizer, logger
    
    # Startup
    logger = setup_logger("radon_api", log_file="logs/api.log")
    logger.info("Starting RADON API server...")
    
    try:
        # Load model and tokenizer
        await load_model_and_tokenizer()
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model and tokenizer: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RADON API server...")


async def load_model_and_tokenizer():
    """Load model and tokenizer"""
    global model, tokenizer
    
    # Model configuration
    model_config_path = os.getenv("MODEL_CONFIG_PATH", "configs/model_config_small.json")
    model_path = os.getenv("MODEL_PATH", "./models/checkpoint")
    tokenizer_path = os.getenv("TOKENIZER_PATH", "./tokenizer/checkpoint")
    clean_model = os.getenv("CLEAN_MODEL", "false").lower() == "true"
    corpus_path = os.getenv("CORPUS_PATH", "./data/raw_corpus")
    
    try:
        # Load model configuration
        config = ModelConfig.from_json(model_config_path)
        
        if clean_model:
            logger.info("🧹 Clean model mode: Initializing from scratch with clean corpus")
            
            # Prepare clean corpus if needed
            if not os.path.exists(os.path.join(corpus_path, "combined_corpus.txt")):
                logger.warning(f"No clean corpus found at {corpus_path}")
                logger.info("Creating model with default initialization")
            
            # Initialize model from scratch
            model = HybridTransformerModel(config)
            logger.info("✅ Clean model initialized from scratch")
            
            # Initialize clean tokenizer
            tokenizer = CustomTokenizer()
            logger.info("✅ Clean tokenizer initialized")
            
        else:
            # Load existing model
            if os.path.exists(model_path):
                model = HybridTransformerModel.from_pretrained(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                # Create new model if checkpoint doesn't exist
                model = HybridTransformerModel(config)
                logger.info("Created new model instance")
            
            # Load tokenizer
            if os.path.exists(tokenizer_path):
                tokenizer = CustomTokenizer.from_pretrained(tokenizer_path)
                logger.info(f"Tokenizer loaded from {tokenizer_path}")
            else:
                # Create default tokenizer if not found
                tokenizer = CustomTokenizer()
                logger.info("Created default tokenizer")
        
        # Move model to appropriate device
        device = os.getenv("DEVICE", "cpu")
        model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Log model info
        if logger:
            model_info = model.get_model_info()
            logger.info(f"Model info: {model_info['model_name']} ({model_info['model_type']}) - {model_info['num_parameters']:,} parameters")
        
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {e}")
        raise


# Create FastAPI application
app = FastAPI(
    title="RADON Custom Transformer API",
    description="API for custom transformer models with GPT-2 and T5 support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Logging middleware for requests"""
    global logger
    
    # Log request
    if logger:
        log_request(
            logger,
            request_data={"url": str(request.url), "method": request.method},
            endpoint=request.url.path,
            method=request.method
        )
    
    # Process request
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Log error
        if logger:
            log_error(logger, e, context={"url": str(request.url), "method": request.method})
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )


# Include routers
app.include_router(api_router, prefix="/api/v1")
app.include_router(vk_router, prefix="/vk")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RADON Custom Transformer API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global model, tokenizer
    
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(next(model.parameters()).device) if model else None
    }
    
    if model is None or tokenizer is None:
        status["status"] = "unhealthy"
        return JSONResponse(status_code=503, content=status)
    
    return status


@app.get("/model/info")
async def model_info():
    """Get model information"""
    global model, tokenizer
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": model.model_type,
        "vocab_size": tokenizer.vocab_size if tokenizer else None,
        "device": str(next(model.parameters()).device),
        "model_info": model.get_model_info()
    }
    
    return info


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    global logger
    
    if logger:
        log_error(logger, exc, context={"url": str(request.url), "method": request.method})
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    global logger
    
    if logger:
        log_error(logger, exc, context={"url": str(request.url), "method": request.method})
    
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run server
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
