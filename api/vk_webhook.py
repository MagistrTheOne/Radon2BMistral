"""
VK Webhook integration for RADON Custom Transformer
"""

import json
import time
import uuid
from typing import Optional, Dict, Any, List
import requests

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field

from utils.logging_utils import log_request, log_error, log_generation
from models.hybrid_model import HybridTransformerModel
from tokenizer.custom_tokenizer import CustomTokenizer


# Import global variables
from .app import model, tokenizer, logger

router = APIRouter()


# VK API configuration
VK_API_URL = "https://api.vk.com/method"
VK_API_VERSION = "5.131"


# Request/Response models
class VKCallbackEvent(BaseModel):
    """VK callback event model"""
    type: str = Field(..., description="Event type")
    object: Dict[str, Any] = Field(..., description="Event object")
    group_id: int = Field(..., description="Group ID")


class VKMessage(BaseModel):
    """VK message model"""
    id: int = Field(..., description="Message ID")
    from_id: int = Field(..., description="Sender ID")
    peer_id: int = Field(..., description="Peer ID")
    text: str = Field(..., description="Message text")
    date: int = Field(..., description="Message timestamp")


class VKWebhookRequest(BaseModel):
    """VK webhook request model"""
    type: str = Field(..., description="Event type")
    object: Dict[str, Any] = Field(..., description="Event object")
    group_id: int = Field(..., description="Group ID")


class VKWebhookResponse(BaseModel):
    """VK webhook response model"""
    response: str = Field("ok", description="Response status")


class VKMessageRequest(BaseModel):
    """VK message request model"""
    message: str = Field(..., description="Message text")
    peer_id: int = Field(..., description="Peer ID")
    access_token: str = Field(..., description="VK access token")
    group_id: Optional[int] = Field(None, description="Group ID")


class VKMessageResponse(BaseModel):
    """VK message response model"""
    success: bool = Field(..., description="Whether message was sent successfully")
    message_id: Optional[int] = Field(None, description="Sent message ID")
    error: Optional[str] = Field(None, description="Error message if any")


def get_vk_access_token() -> Optional[str]:
    """Get VK access token from environment"""
    import os
    return os.getenv("VK_ACCESS_TOKEN")


def send_vk_message(
    message: str,
    peer_id: int,
    access_token: str,
    group_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Send message via VK API
    
    Args:
        message: Message text
        peer_id: Peer ID (user or chat ID)
        access_token: VK access token
        group_id: Optional group ID
    
    Returns:
        API response dictionary
    """
    
    url = f"{VK_API_URL}/messages.send"
    
    params = {
        "access_token": access_token,
        "v": VK_API_VERSION,
        "peer_id": peer_id,
        "message": message,
        "random_id": int(time.time() * 1000)
    }
    
    if group_id:
        params["group_id"] = group_id
    
    try:
        response = requests.post(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to send message: {str(e)}"}


def process_vk_message(message_text: str, user_id: int) -> str:
    """
    Process VK message and generate response
    
    Args:
        message_text: Incoming message text
        user_id: User ID
    
    Returns:
        Generated response text
    """
    
    if model is None or tokenizer is None:
        return "Извините, модель не загружена. Попробуйте позже."
    
    try:
        # Generate response using the model
        inputs = tokenizer(
            message_text,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with model.eval():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Remove original message from generated text
        if generated_text.startswith(message_text):
            generated_text = generated_text[len(message_text):].strip()
        
        # Limit response length
        if len(generated_text) > 1000:
            generated_text = generated_text[:1000] + "..."
        
        return generated_text
        
    except Exception as e:
        if logger:
            log_error(logger, e, context={"user_id": user_id, "message": message_text})
        return "Извините, произошла ошибка при генерации ответа."


@router.post("/webhook", response_model=VKWebhookResponse)
async def vk_webhook(request: VKWebhookRequest, background_tasks: BackgroundTasks):
    """Handle VK webhook events"""
    
    try:
        # Log request
        if logger:
            log_request(
                logger,
                request_data=request.dict(),
                endpoint="/vk/webhook",
                user_id=str(request.group_id)
            )
        
        # Handle different event types
        if request.type == "confirmation":
            # Return confirmation code
            confirmation_code = get_vk_confirmation_code()
            return VKWebhookResponse(response=confirmation_code)
        
        elif request.type == "message_new":
            # Handle new message
            await handle_new_message(request.object, background_tasks)
            return VKWebhookResponse(response="ok")
        
        else:
            # Unknown event type
            if logger:
                log_error(
                    logger,
                    Exception(f"Unknown event type: {request.type}"),
                    context={"event": request.dict()}
                )
            return VKWebhookResponse(response="ok")
        
    except Exception as e:
        if logger:
            log_error(logger, e, context={"webhook_request": request.dict()})
        return VKWebhookResponse(response="error")


async def handle_new_message(message_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Handle new message event"""
    
    try:
        # Extract message information
        message = message_data.get("message", {})
        message_id = message.get("id")
        from_id = message.get("from_id")
        peer_id = message.get("peer_id")
        text = message.get("text", "")
        
        # Skip messages from bots
        if from_id < 0:
            return
        
        # Skip empty messages
        if not text.strip():
            return
        
        # Log message
        if logger:
            log_request(
                logger,
                request_data={"message": text, "from_id": from_id},
                endpoint="/vk/message",
                user_id=str(from_id)
            )
        
        # Process message in background
        background_tasks.add_task(
            process_and_respond_to_message,
            message_id=message_id,
            from_id=from_id,
            peer_id=peer_id,
            text=text
        )
        
    except Exception as e:
        if logger:
            log_error(logger, e, context={"message_data": message_data})


async def process_and_respond_to_message(
    message_id: int,
    from_id: int,
    peer_id: int,
    text: str
):
    """Process message and send response"""
    
    try:
        # Generate response
        response_text = process_vk_message(text, from_id)
        
        # Get access token
        access_token = get_vk_access_token()
        if not access_token:
            if logger:
                log_error(
                    logger,
                    Exception("VK access token not configured"),
                    context={"message_id": message_id, "from_id": from_id}
                )
            return
        
        # Send response
        result = send_vk_message(
            message=response_text,
            peer_id=peer_id,
            access_token=access_token
        )
        
        # Log response
        if logger:
            log_generation(
                logger,
                prompt=text,
                generated_text=response_text,
                generation_time=0.0,  # Background task, no timing
                model_name=model.model_type if model else None,
                user_id=str(from_id),
                request_id=str(message_id)
            )
        
        # Check for errors
        if "error" in result:
            if logger:
                log_error(
                    logger,
                    Exception(f"VK API error: {result['error']}"),
                    context={"message_id": message_id, "from_id": from_id}
                )
        
    except Exception as e:
        if logger:
            log_error(logger, e, context={"message_id": message_id, "from_id": from_id})


@router.post("/message/send", response_model=VKMessageResponse)
async def send_message(request: VKMessageRequest):
    """Send message via VK API"""
    
    try:
        # Log request
        if logger:
            log_request(
                logger,
                request_data={"message": request.message, "peer_id": request.peer_id},
                endpoint="/vk/message/send",
                user_id=str(request.peer_id)
            )
        
        # Send message
        result = send_vk_message(
            message=request.message,
            peer_id=request.peer_id,
            access_token=request.access_token,
            group_id=request.group_id
        )
        
        # Check for errors
        if "error" in result:
            return VKMessageResponse(
                success=False,
                error=result["error"]
            )
        
        # Extract message ID from response
        message_id = result.get("response", {}).get("message_id")
        
        return VKMessageResponse(
            success=True,
            message_id=message_id
        )
        
    except Exception as e:
        if logger:
            log_error(logger, e, user_id=str(request.peer_id))
        return VKMessageResponse(
            success=False,
            error=str(e)
        )


@router.get("/webhook/info")
async def get_webhook_info():
    """Get webhook configuration information"""
    
    return {
        "webhook_url": "/vk/webhook",
        "supported_events": ["confirmation", "message_new"],
        "confirmation_code": get_vk_confirmation_code(),
        "status": "active"
    }


def get_vk_confirmation_code() -> str:
    """Get VK confirmation code from environment"""
    import os
    return os.getenv("VK_CONFIRMATION_CODE", "default_confirmation_code")


@router.post("/webhook/test")
async def test_webhook():
    """Test webhook functionality"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    
    try:
        # Test message processing
        test_message = "Привет! Как дела?"
        test_user_id = 12345
        
        # Generate response
        response_text = process_vk_message(test_message, test_user_id)
        
        return {
            "success": True,
            "test_message": test_message,
            "generated_response": response_text,
            "model_type": model.model_type,
            "tokenizer_loaded": tokenizer is not None
        }
        
    except Exception as e:
        if logger:
            log_error(logger, e)
        raise HTTPException(status_code=500, detail=f"Webhook test failed: {str(e)}")


@router.get("/webhook/status")
async def get_webhook_status():
    """Get webhook status and configuration"""
    
    return {
        "webhook_active": True,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "vk_token_configured": get_vk_access_token() is not None,
        "confirmation_code": get_vk_confirmation_code(),
        "supported_events": ["confirmation", "message_new"],
        "api_version": VK_API_VERSION
    }
