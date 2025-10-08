#!/usr/bin/env python3
"""
SageMaker Entry Point for RADON Training
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def setup_environment():
    """Setup training environment"""
    print("🔧 Setting up RADON training environment...")
    
    # Install dependencies
    subprocess.run([
        "pip", "install", "-r", "/opt/ml/code/requirements.txt"
    ], check=True)
    
    # Install additional AWS dependencies
    subprocess.run([
        "pip", "install", 
        "boto3", "sagemaker", "s3fs"
    ], check=True)
    
    print("✅ Environment setup complete")

def main():
    """SageMaker entry point"""
    print("🚀 Starting RADON training on SageMaker...")
    
    # Setup environment
    setup_environment()
    
    # Get training parameters
    config_path = os.getenv("SM_MODEL_DIR", "/opt/ml/code/configs/model_config_mistral_balanced_tier.json")
    
    # Run training
    try:
        from training_script import main as train_main
        train_main()
        
        print("🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
