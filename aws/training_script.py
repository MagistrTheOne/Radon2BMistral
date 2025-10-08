#!/usr/bin/env python3
"""
AWS Training Script for RADON Models
Optimized for AWS SageMaker/EC2 with GPU instances
"""

import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoConfig, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import logging
from pathlib import Path
import boto3
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RADONTrainer:
    """RADON Model Trainer for AWS"""
    
    def __init__(self, config_path: str, output_dir: str = "/opt/ml/model"):
        self.config_path = config_path
        self.output_dir = output_dir
        self.config = self.load_config()
        
        # AWS specific settings
        self.s3_bucket = os.getenv("S3_BUCKET", "radon-training-data")
        self.s3_prefix = os.getenv("S3_PREFIX", "radon-models")
        
    def load_config(self):
        """Load RADON configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def setup_distributed_training(self):
        """Setup distributed training for multi-GPU"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')
            
            logger.info(f"Distributed training: rank {rank}/{world_size}")
            return True
        return False
    
    def load_training_data(self):
        """Load training datasets from S3 or local"""
        logger.info("Loading training datasets...")
        
        # Try to load from S3 first
        try:
            s3 = boto3.client('s3')
            datasets = []
            
            # Load Russian corpus
            ru_dataset = self.load_dataset_from_s3("russian-corpus")
            if ru_dataset:
                datasets.append(ru_dataset)
                logger.info(f"Loaded Russian corpus: {len(ru_dataset)} samples")
            
            # Load English corpus
            en_dataset = self.load_dataset_from_s3("english-corpus")
            if en_dataset:
                datasets.append(en_dataset)
                logger.info(f"Loaded English corpus: {len(en_dataset)} samples")
            
            # Load code corpus
            code_dataset = self.load_dataset_from_s3("code-corpus")
            if code_dataset:
                datasets.append(code_dataset)
                logger.info(f"Loaded code corpus: {len(code_dataset)} samples")
            
            if not datasets:
                # Fallback to local datasets
                logger.warning("No S3 datasets found, using local fallback")
                return self.load_local_datasets()
            
            return datasets
            
        except Exception as e:
            logger.warning(f"S3 loading failed: {e}, using local datasets")
            return self.load_local_datasets()
    
    def load_dataset_from_s3(self, dataset_name: str):
        """Load dataset from S3"""
        try:
            s3 = boto3.client('s3')
            key = f"{self.s3_prefix}/datasets/{dataset_name}.json"
            
            # Download dataset file
            local_path = f"/tmp/{dataset_name}.json"
            s3.download_file(self.s3_bucket, key, local_path)
            
            # Load as HuggingFace dataset
            from datasets import Dataset
            with open(local_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return Dataset.from_list(data)
            
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name} from S3: {e}")
            return None
    
    def load_local_datasets(self):
        """Load local fallback datasets"""
        datasets = []
        
        # Create sample datasets for testing
        sample_data = [
            {"text": "Привет! Как дела? Это тестовый текст на русском языке."},
            {"text": "Hello! How are you? This is a test text in English."},
            {"text": "def hello_world():\n    print('Hello, World!')"},
            {"text": "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()"},
        ]
        
        from datasets import Dataset
        dataset = Dataset.from_list(sample_data * 1000)  # Repeat for more data
        datasets.append(dataset)
        
        logger.info(f"Created local fallback dataset: {len(dataset)} samples")
        return datasets
    
    def prepare_model(self):
        """Prepare RADON model for training"""
        logger.info("Preparing RADON model...")
        
        # Import RADON model components
        import sys
        sys.path.append('/opt/ml/code')
        
        from models.mistral_model import MistralForCausalLM
        from models.config import ModelConfig
        
        # Create model config
        model_config = ModelConfig(**self.config)
        
        # Initialize model
        model = MistralForCausalLM(model_config)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info(f"Model moved to GPU: {torch.cuda.get_device_name()}")
        
        # Setup distributed training
        if self.setup_distributed_training():
            model = DDP(model)
            logger.info("Model wrapped with DDP")
        
        return model
    
    def prepare_tokenizer(self):
        """Prepare tokenizer"""
        logger.info("Preparing tokenizer...")
        
        # Try to load from HuggingFace first
        try:
            tokenizer = AutoTokenizer.from_pretrained("MagistrTheOne/RadonSAI")
            logger.info("Loaded tokenizer from HuggingFace")
            return tokenizer
        except:
            logger.warning("Failed to load tokenizer from HF, using fallback")
        
        # Fallback to local tokenizer
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        
        # Create simple BPE tokenizer
        tokenizer = Tokenizer(BPE())
        # Add basic vocabulary
        tokenizer.add_tokens(["<pad>", "<unk>", "<s>", "</s>"])
        
        return tokenizer
    
    def setup_training_args(self):
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=1000,
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000,
            save_total_limit=3,
            prediction_loss_only=True,
            fp16=True,  # Use mixed precision
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for now
        )
    
    def train(self):
        """Main training function"""
        logger.info("🚀 Starting RADON training on AWS...")
        
        # Load datasets
        datasets = self.load_training_data()
        if not datasets:
            raise ValueError("No training data available")
        
        # Combine datasets
        from datasets import concatenate_datasets
        combined_dataset = concatenate_datasets(datasets)
        logger.info(f"Combined dataset size: {len(combined_dataset)}")
        
        # Prepare model and tokenizer
        model = self.prepare_model()
        tokenizer = self.prepare_tokenizer()
        
        # Prepare data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=combined_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info("Saving trained model...")
        trainer.save_model()
        
        # Upload to S3
        self.upload_to_s3()
        
        logger.info("🎉 Training completed!")
    
    def upload_to_s3(self):
        """Upload trained model to S3"""
        try:
            s3 = boto3.client('s3')
            
            # Upload model files
            for file_path in Path(self.output_dir).rglob("*"):
                if file_path.is_file():
                    s3_key = f"{self.s3_prefix}/models/{file_path.name}"
                    s3.upload_file(str(file_path), self.s3_bucket, s3_key)
                    logger.info(f"Uploaded {file_path.name} to S3")
            
            logger.info("✅ Model uploaded to S3")
            
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")

def main():
    """Main execution function"""
    # Get configuration
    config_path = os.getenv("CONFIG_PATH", "/opt/ml/code/configs/model_config_mistral_balanced_tier.json")
    output_dir = os.getenv("OUTPUT_DIR", "/opt/ml/model")
    
    # Create trainer
    trainer = RADONTrainer(config_path, output_dir)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
