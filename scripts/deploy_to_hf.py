"""
Deployment script for Hugging Face Hub
"""

import os
import json
import argparse
from typing import Dict, Any, Optional
import torch

from huggingface_hub import HfApi, Repository, login
from transformers import AutoModel, AutoTokenizer

from models.hybrid_model import HybridTransformerModel
from models.config import ModelConfig
from tokenizer.custom_tokenizer import CustomTokenizer
from utils.logging_utils import setup_logger, log_deployment
from utils.model_utils import push_to_huggingface_hub, create_model_card_file


def deploy_to_huggingface(
    model_path: str,
    tokenizer_path: str,
    repo_name: str,
    hf_token: str,
    private: bool = False,
    commit_message: str = "Add custom transformer model",
    logger: Optional[Any] = None
) -> Dict[str, Any]:
    """Deploy model to Hugging Face Hub"""
    
    try:
        # Login to Hugging Face
        login(token=hf_token)
        
        if logger:
            logger.info(f"Logged in to Hugging Face Hub")
        
        # Load model and tokenizer
        model = HybridTransformerModel.from_pretrained(model_path)
        tokenizer = CustomTokenizer.from_pretrained(tokenizer_path)
        
        if logger:
            logger.info(f"Loaded model from {model_path}")
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        
        # Deploy to Hub
        start_time = time.time()
        repo_url = push_to_huggingface_hub(
            model=model,
            tokenizer=tokenizer,
            repo_name=repo_name,
            hf_token=hf_token,
            private=private,
            commit_message=commit_message
        )
        deployment_time = time.time() - start_time
        
        # Log deployment
        if logger:
            log_deployment(
                logger,
                deployment_type="huggingface",
                model_name=repo_name,
                deployment_url=repo_url,
                status="success",
                deployment_time=deployment_time
            )
        
        return {
            "success": True,
            "repo_url": repo_url,
            "deployment_time": deployment_time,
            "model_type": model.model_type,
            "private": private
        }
        
    except Exception as e:
        if logger:
            log_deployment(
                logger,
                deployment_type="huggingface",
                model_name=repo_name,
                status="failed",
                error_message=str(e)
            )
        raise


def create_model_card(
    model_path: str,
    output_path: str,
    model_name: str,
    description: Optional[str] = None,
    usage_example: Optional[str] = None,
    logger: Optional[Any] = None
) -> str:
    """Create model card for the deployed model"""
    
    try:
        # Load model configuration
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = None
        
        # Create model card
        model_card_path = create_model_card_file(
            output_path,
            model_name=model_name,
            config=config,
            description=description,
            usage_example=usage_example
        )
        
        if logger:
            logger.info(f"Model card created: {model_card_path}")
        
        return model_card_path
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to create model card: {e}")
        raise


def verify_deployment(
    repo_name: str,
    hf_token: str,
    logger: Optional[Any] = None
) -> Dict[str, Any]:
    """Verify deployment on Hugging Face Hub"""
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Get repository info
        repo_info = api.repo_info(repo_id=repo_name, token=hf_token)
        
        # Check if model files exist
        files = api.list_repo_files(repo_id=repo_name, token=hf_token)
        
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        missing_files = [f for f in required_files if f not in files]
        
        verification_result = {
            "repo_exists": True,
            "repo_url": f"https://huggingface.co/{repo_name}",
            "files": files,
            "missing_files": missing_files,
            "verification_passed": len(missing_files) == 0
        }
        
        if logger:
            if verification_result["verification_passed"]:
                logger.info(f"Deployment verification passed for {repo_name}")
            else:
                logger.warning(f"Deployment verification failed for {repo_name}: missing {missing_files}")
        
        return verification_result
        
    except Exception as e:
        if logger:
            logger.error(f"Deployment verification failed: {e}")
        return {
            "repo_exists": False,
            "error": str(e),
            "verification_passed": False
        }


def main():
    """Main deployment function"""
    
    parser = argparse.ArgumentParser(description="Deploy RADON model to Hugging Face Hub")
    parser.add_argument("--model_path", required=True, help="Path to model directory")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer directory")
    parser.add_argument("--repo_name", required=True, help="Repository name on HF Hub")
    parser.add_argument("--hf_token", required=True, help="Hugging Face token")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--commit_message", default="Add custom transformer model", help="Commit message")
    parser.add_argument("--verify", action="store_true", help="Verify deployment after upload")
    parser.add_argument("--model_card", action="store_true", help="Create model card")
    parser.add_argument("--description", help="Model description for model card")
    parser.add_argument("--usage_example", help="Usage example for model card")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("radon_deployment", log_file="logs/deployment.log")
    logger.info(f"Starting deployment with args: {args}")
    
    try:
        # Deploy to Hugging Face Hub
        deployment_result = deploy_to_huggingface(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            repo_name=args.repo_name,
            hf_token=args.hf_token,
            private=args.private,
            commit_message=args.commit_message,
            logger=logger
        )
        
        logger.info(f"Deployment successful: {deployment_result}")
        
        # Create model card if requested
        if args.model_card:
            model_card_path = create_model_card(
                model_path=args.model_path,
                output_path=args.model_path,
                model_name=args.repo_name,
                description=args.description,
                usage_example=args.usage_example,
                logger=logger
            )
            logger.info(f"Model card created: {model_card_path}")
        
        # Verify deployment if requested
        if args.verify:
            verification_result = verify_deployment(
                repo_name=args.repo_name,
                hf_token=args.hf_token,
                logger=logger
            )
            logger.info(f"Verification result: {verification_result}")
            
            if not verification_result["verification_passed"]:
                logger.error("Deployment verification failed!")
                return 1
        
        logger.info("Deployment completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1


if __name__ == "__main__":
    import time
    exit(main())
