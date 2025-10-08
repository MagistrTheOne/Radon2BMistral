#!/usr/bin/env python3
"""
AWS Deployment Script for RADON Models
Deploy trained models to production endpoints
"""

import os
import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.predictor import Predictor
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RADONDeployer:
    """RADON Model Deployer for AWS"""
    
    def __init__(self, region="us-east-1"):
        self.region = region
        self.sagemaker_session = sagemaker.Session()
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Configuration
        self.bucket = os.getenv("S3_BUCKET", "radon-models")
        self.role = os.getenv("SAGEMAKER_ROLE", "arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole")
        
    def create_training_job(self, model_config: str, instance_type: str = "ml.p3.2xlarge"):
        """Create SageMaker training job"""
        logger.info(f"Creating training job for {model_config}...")
        
        # Training configuration
        training_config = {
            "entry_point": "sagemaker_entrypoint.py",
            "source_dir": "/opt/ml/code",
            "role": self.role,
            "instance_count": 1,
            "instance_type": instance_type,
            "framework_version": "1.12.1",
            "py_version": "py38",
            "hyperparameters": {
                "config_path": f"s3://{self.bucket}/configs/{model_config}",
                "output_dir": "/opt/ml/model"
            }
        }
        
        # Create estimator
        estimator = PyTorch(
            **training_config,
            sagemaker_session=self.sagemaker_session
        )
        
        # Start training
        estimator.fit({
            "training": f"s3://{self.bucket}/data/training/",
            "validation": f"s3://{self.bucket}/data/validation/"
        })
        
        logger.info("✅ Training job completed")
        return estimator
    
    def deploy_model(self, model_artifact: str, endpoint_name: str = None):
        """Deploy trained model to endpoint"""
        if not endpoint_name:
            endpoint_name = f"radon-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"Deploying model to endpoint: {endpoint_name}")
        
        # Create model
        model = PyTorchModel(
            model_data=model_artifact,
            role=self.role,
            entry_point="inference.py",
            framework_version="1.12.1",
            py_version="py38",
            sagemaker_session=self.sagemaker_session
        )
        
        # Deploy to endpoint
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.g4dn.xlarge",
            endpoint_name=endpoint_name
        )
        
        logger.info(f"✅ Model deployed to {endpoint_name}")
        return predictor
    
    def create_inference_endpoint(self, model_name: str):
        """Create inference endpoint for specific model"""
        logger.info(f"Creating inference endpoint for {model_name}...")
        
        # Model configurations
        model_configs = {
            "radon-355m": {
                "instance_type": "ml.t3.medium",
                "model_artifact": f"s3://{self.bucket}/models/radon-355m/model.tar.gz"
            },
            "radon-7b": {
                "instance_type": "ml.g4dn.xlarge", 
                "model_artifact": f"s3://{self.bucket}/models/radon-7b/model.tar.gz"
            },
            "radon-70b": {
                "instance_type": "ml.p3.2xlarge",
                "model_artifact": f"s3://{self.bucket}/models/radon-70b/model.tar.gz"
            }
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = model_configs[model_name]
        
        # Deploy model
        predictor = self.deploy_model(
            model_artifact=config["model_artifact"],
            endpoint_name=f"radon-{model_name}-endpoint"
        )
        
        return predictor
    
    def test_endpoint(self, endpoint_name: str):
        """Test deployed endpoint"""
        logger.info(f"Testing endpoint: {endpoint_name}")
        
        # Create predictor
        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=self.sagemaker_session
        )
        
        # Test data
        test_data = {
            "prompt": "Привет! Как дела?",
            "max_length": 100,
            "temperature": 0.7
        }
        
        # Make prediction
        response = predictor.predict(test_data)
        logger.info(f"Test response: {response}")
        
        return response
    
    def create_batch_transform_job(self, model_name: str, input_data: str, output_data: str):
        """Create batch transform job for large-scale inference"""
        logger.info(f"Creating batch transform job for {model_name}...")
        
        # Create model for batch transform
        model = PyTorchModel(
            model_data=f"s3://{self.bucket}/models/{model_name}/model.tar.gz",
            role=self.role,
            entry_point="batch_inference.py",
            framework_version="1.12.1",
            py_version="py38",
            sagemaker_session=self.sagemaker_session
        )
        
        # Create transformer
        transformer = model.transformer(
            instance_count=2,
            instance_type="ml.m5.large",
            output_path=output_data
        )
        
        # Start batch transform
        transformer.transform(
            data=input_data,
            content_type="application/json",
            split_type="Line"
        )
        
        logger.info("✅ Batch transform job started")
        return transformer

def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy RADON models to AWS")
    parser.add_argument("--model", required=True, help="Model name (radon-355m, radon-7b, radon-70b)")
    parser.add_argument("--action", required=True, help="Action (train, deploy, test)")
    parser.add_argument("--endpoint", help="Endpoint name for testing")
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = RADONDeployer()
    
    if args.action == "train":
        # Start training
        deployer.create_training_job(f"{args.model}.json")
        
    elif args.action == "deploy":
        # Deploy model
        deployer.create_inference_endpoint(args.model)
        
    elif args.action == "test":
        # Test endpoint
        if not args.endpoint:
            args.endpoint = f"radon-{args.model}-endpoint"
        deployer.test_endpoint(args.endpoint)
    
    else:
        print(f"Unknown action: {args.action}")

if __name__ == "__main__":
    main()
