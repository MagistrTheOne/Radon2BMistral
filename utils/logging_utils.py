"""
Logging utilities for RADON project
"""

import logging
import sys
from typing import Optional, Dict, Any, Union
from datetime import datetime
import json
import traceback


def setup_logger(
    name: str = "radon",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    include_level: bool = True,
    include_module: bool = True,
    include_line_number: bool = True
) -> logging.Logger:
    """
    Setup logger with custom configuration
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
        include_timestamp: Include timestamp in logs
        include_level: Include log level in logs
        include_module: Include module name in logs
        include_line_number: Include line number in logs
    
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create format string
    if format_string is None:
        format_parts = []
        if include_timestamp:
            format_parts.append("%(asctime)s")
        if include_level:
            format_parts.append("%(levelname)s")
        if include_module:
            format_parts.append("%(name)s")
        if include_line_number:
            format_parts.append("%(filename)s:%(lineno)d")
        format_parts.append("%(message)s")
        format_string = " - ".join(format_parts)
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_request(
    logger: logging.Logger,
    request_data: Dict[str, Any],
    endpoint: str,
    method: str = "POST",
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    response_time: Optional[float] = None
) -> None:
    """
    Log API request
    
    Args:
        logger: Logger instance
        request_data: Request data dictionary
        endpoint: API endpoint
        method: HTTP method
        user_id: Optional user ID
        request_id: Optional request ID
        response_time: Optional response time in seconds
    """
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "request",
        "endpoint": endpoint,
        "method": method,
        "user_id": user_id,
        "request_id": request_id,
        "response_time": response_time,
        "data": request_data
    }
    
    logger.info(f"API Request: {json.dumps(log_data, ensure_ascii=False)}")


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    include_traceback: bool = True
) -> None:
    """
    Log error with context
    
    Args:
        logger: Logger instance
        error: Exception instance
        context: Optional context dictionary
        user_id: Optional user ID
        request_id: Optional request ID
        include_traceback: Include full traceback
    """
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "user_id": user_id,
        "request_id": request_id,
        "context": context
    }
    
    if include_traceback:
        log_data["traceback"] = traceback.format_exc()
    
    logger.error(f"Error occurred: {json.dumps(log_data, ensure_ascii=False)}")


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Union[int, float, str]],
    model_name: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> None:
    """
    Log model metrics
    
    Args:
        logger: Logger instance
        metrics: Metrics dictionary
        model_name: Optional model name
        user_id: Optional user ID
        request_id: Optional request ID
    """
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "metrics",
        "model_name": model_name,
        "user_id": user_id,
        "request_id": request_id,
        "metrics": metrics
    }
    
    logger.info(f"Metrics: {json.dumps(log_data, ensure_ascii=False)}")


def log_model_loading(
    logger: logging.Logger,
    model_name: str,
    model_path: str,
    load_time: float,
    model_size: Optional[int] = None,
    device: Optional[str] = None
) -> None:
    """
    Log model loading event
    
    Args:
        logger: Logger instance
        model_name: Model name
        model_path: Model path
        load_time: Loading time in seconds
        model_size: Optional model size in bytes
        device: Optional device name
    """
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "model_loading",
        "model_name": model_name,
        "model_path": model_path,
        "load_time": load_time,
        "model_size": model_size,
        "device": device
    }
    
    logger.info(f"Model loaded: {json.dumps(log_data, ensure_ascii=False)}")


def log_generation(
    logger: logging.Logger,
    prompt: str,
    generated_text: str,
    generation_time: float,
    model_name: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    generation_params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log text generation event
    
    Args:
        logger: Logger instance
        prompt: Input prompt
        generated_text: Generated text
        generation_time: Generation time in seconds
        model_name: Optional model name
        user_id: Optional user ID
        request_id: Optional request ID
        generation_params: Optional generation parameters
    """
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "generation",
        "model_name": model_name,
        "user_id": user_id,
        "request_id": request_id,
        "prompt": prompt,
        "generated_text": generated_text,
        "generation_time": generation_time,
        "generation_params": generation_params
    }
    
    logger.info(f"Text generated: {json.dumps(log_data, ensure_ascii=False)}")


def log_training(
    logger: logging.Logger,
    epoch: int,
    step: int,
    loss: float,
    learning_rate: float,
    model_name: Optional[str] = None,
    additional_metrics: Optional[Dict[str, Union[int, float]]] = None
) -> None:
    """
    Log training event
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        step: Current step
        loss: Current loss
        learning_rate: Current learning rate
        model_name: Optional model name
        additional_metrics: Optional additional metrics
    """
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "training",
        "model_name": model_name,
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "learning_rate": learning_rate,
        "additional_metrics": additional_metrics
    }
    
    logger.info(f"Training step: {json.dumps(log_data, ensure_ascii=False)}")


def log_deployment(
    logger: logging.Logger,
    deployment_type: str,
    model_name: str,
    deployment_url: Optional[str] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    deployment_time: Optional[float] = None
) -> None:
    """
    Log deployment event
    
    Args:
        logger: Logger instance
        deployment_type: Type of deployment (hf, docker, etc.)
        model_name: Model name
        deployment_url: Optional deployment URL
        status: Deployment status
        error_message: Optional error message
        deployment_time: Optional deployment time in seconds
    """
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "type": "deployment",
        "deployment_type": deployment_type,
        "model_name": model_name,
        "deployment_url": deployment_url,
        "status": status,
        "error_message": error_message,
        "deployment_time": deployment_time
    }
    
    if status == "success":
        logger.info(f"Deployment successful: {json.dumps(log_data, ensure_ascii=False)}")
    else:
        logger.error(f"Deployment failed: {json.dumps(log_data, ensure_ascii=False)}")


def create_request_logger(
    name: str = "radon_requests",
    log_file: str = "logs/requests.log"
) -> logging.Logger:
    """
    Create specialized logger for API requests
    
    Args:
        name: Logger name
        log_file: Log file path
    
    Returns:
        Request logger
    """
    
    return setup_logger(
        name=name,
        level=logging.INFO,
        log_file=log_file,
        format_string="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_error_logger(
    name: str = "radon_errors",
    log_file: str = "logs/errors.log"
) -> logging.Logger:
    """
    Create specialized logger for errors
    
    Args:
        name: Logger name
        log_file: Log file path
    
    Returns:
        Error logger
    """
    
    return setup_logger(
        name=name,
        level=logging.ERROR,
        log_file=log_file,
        format_string="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )


def create_metrics_logger(
    name: str = "radon_metrics",
    log_file: str = "logs/metrics.log"
) -> logging.Logger:
    """
    Create specialized logger for metrics
    
    Args:
        name: Logger name
        log_file: Log file path
    
    Returns:
        Metrics logger
    """
    
    return setup_logger(
        name=name,
        level=logging.INFO,
        log_file=log_file,
        format_string="%(asctime)s - %(message)s"
    )
