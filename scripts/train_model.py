"""
Training script for RADON Custom Transformer models
"""

import os
import json
import argparse
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from models.hybrid_model import HybridTransformerModel
from models.config import ModelConfig, TrainingConfig
from tokenizer.custom_tokenizer import CustomTokenizer
from utils.logging_utils import setup_logger, log_training
from utils.model_utils import save_model, get_model_size


class TextDataset(Dataset):
    """Dataset for text training"""
    
    def __init__(self, texts: list, tokenizer: CustomTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": inputs["input_ids"].squeeze()
        }


def load_training_data(data_path: str) -> list:
    """Load training data from file"""
    
    texts = []
    
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                texts = data
            elif isinstance(data, dict) and 'texts' in data:
                texts = data['texts']
            else:
                raise ValueError("Invalid JSON format")
    
    elif data_path.endswith('.txt'):
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    else:
        raise ValueError("Unsupported file format. Use .json or .txt")
    
    return texts


def train_model(
    model: HybridTransformerModel,
    tokenizer: CustomTokenizer,
    train_data: list,
    config: TrainingConfig,
    model_config: ModelConfig,
    output_dir: str,
    logger: Optional[Any] = None
) -> Dict[str, Any]:
    """Train the model"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create dataset and dataloader
    train_dataset = TextDataset(train_data, tokenizer, model_config.max_position_embeddings)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    global_step = 0
    total_loss = 0
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (step + 1):.4f}'
            })
            
            # Log training step
            if logger and global_step % config.logging_steps == 0:
                log_training(
                    logger,
                    epoch=epoch,
                    step=global_step,
                    loss=loss.item(),
                    learning_rate=scheduler.get_last_lr()[0],
                    model_name=model.model_type,
                    additional_metrics={
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step
                    }
                )
            
            # Save checkpoint
            if global_step % config.save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                save_model(
                    model,
                    checkpoint_dir,
                    model_name=f"{model.model_type}_checkpoint_{global_step}",
                    config=model_config.to_dict(),
                    tokenizer=tokenizer
                )
                if logger:
                    logger.info(f"Checkpoint saved at step {global_step}")
        
        # Log epoch metrics
        if logger:
            log_training(
                logger,
                epoch=epoch,
                step=global_step,
                loss=epoch_loss / len(train_dataloader),
                learning_rate=scheduler.get_last_lr()[0],
                model_name=model.model_type,
                additional_metrics={
                    "epoch_loss": epoch_loss / len(train_dataloader),
                    "epoch": epoch
                }
            )
    
    # Save final model
    final_model_dir = os.path.join(output_dir, "final_model")
    save_model(
        model,
        final_model_dir,
        model_name=f"{model.model_type}_final",
        config=model_config.to_dict(),
        tokenizer=tokenizer
    )
    
    # Training results
    results = {
        "total_steps": global_step,
        "total_epochs": config.num_epochs,
        "final_loss": total_loss / global_step,
        "model_size": get_model_size(model),
        "output_directory": output_dir
    }
    
    if logger:
        logger.info(f"Training completed. Results: {results}")
    
    return results


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Train RADON Custom Transformer")
    parser.add_argument("--model_config", required=True, help="Path to model configuration")
    parser.add_argument("--training_config", required=True, help="Path to training configuration")
    parser.add_argument("--data_path", required=True, help="Path to training data")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--tokenizer_path", help="Path to tokenizer")
    parser.add_argument("--resume_from", help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup logging
    logger = setup_logger("radon_training", log_file="logs/training.log")
    logger.info(f"Starting training with args: {args}")
    
    try:
        # Load configurations
        model_config = ModelConfig.from_json(args.model_config)
        training_config = TrainingConfig.from_json(args.training_config)
        
        logger.info(f"Model config: {model_config.to_dict()}")
        logger.info(f"Training config: {training_config.to_dict()}")
        
        # Load training data
        train_data = load_training_data(args.data_path)
        logger.info(f"Loaded {len(train_data)} training samples")
        
        # Load or create tokenizer
        if args.tokenizer_path and os.path.exists(args.tokenizer_path):
            tokenizer = CustomTokenizer.from_pretrained(args.tokenizer_path)
            logger.info(f"Loaded tokenizer from {args.tokenizer_path}")
        else:
            # Create default tokenizer
            tokenizer = CustomTokenizer()
            logger.info("Using default tokenizer")
        
        # Create or load model
        if args.resume_from and os.path.exists(args.resume_from):
            model = HybridTransformerModel.from_pretrained(args.resume_from)
            logger.info(f"Resumed training from {args.resume_from}")
        else:
            model = HybridTransformerModel(model_config)
            logger.info("Created new model")
        
        # Log model information
        model_info = get_model_size(model)
        logger.info(f"Model size: {model_info}")
        
        # Train model
        start_time = time.time()
        results = train_model(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            config=training_config,
            model_config=model_config,
            output_dir=args.output_dir,
            logger=logger
        )
        training_time = time.time() - start_time
        
        # Log final results
        results["training_time"] = training_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final results: {results}")
        
        # Save training results
        results_path = os.path.join(args.output_dir, "training_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
