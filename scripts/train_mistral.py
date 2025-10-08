#!/usr/bin/env python3
"""
Training script for Mistral model with advanced optimizations
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, AdamW
from typing import Dict, List, Optional, Any
import time
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.hybrid_model import HybridTransformerModel
from models.config import ModelConfig
from tokenizer.hybrid_tokenizer import HybridTokenizer
from tokenizer.custom_tokenizer import CustomTokenizer
from utils.logging_utils import setup_logger


class TextDataset(Dataset):
    """Dataset for text training"""
    
    def __init__(self, texts: List[str], tokenizer: Any, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class MistralTrainer:
    """Trainer for Mistral model with advanced optimizations"""
    
    def __init__(self, config: ModelConfig, tokenizer: Any, device: str = "cuda"):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize model
        self.model = HybridTransformerModel(config)
        self.model.to(device)
        
        # Setup logging
        self.logger = setup_logger("mistral_training", "logs/mistral_training.log")
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.best_loss = float('inf')
        
    def setup_optimizer(self, learning_rate: float = 5e-4, weight_decay: float = 0.01):
        """Setup optimizer and scheduler"""
        # Use AdamW optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        self.logger.info(f"Optimizer setup: AdamW with lr={learning_rate}, weight_decay={weight_decay}")
    
    def setup_scheduler(self, num_training_steps: int, warmup_steps: int = 100):
        """Setup learning rate scheduler"""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.logger.info(f"Scheduler setup: Linear with {warmup_steps} warmup steps")
    
    def setup_mixed_precision(self, use_fp16: bool = True):
        """Setup mixed precision training"""
        if use_fp16 and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_fp16 = True
            self.logger.info("Mixed precision training enabled (FP16)")
        else:
            self.scaler = None
            self.use_fp16 = False
            self.logger.info("Mixed precision training disabled")
    
    def setup_gradient_checkpointing(self, enable: bool = True):
        """Setup gradient checkpointing for memory efficiency"""
        if enable:
            self.model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        else:
            self.model.gradient_checkpointing_disable()
            self.logger.info("Gradient checkpointing disabled")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        self.logger.info(f"Starting epoch {epoch}")
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # Backward pass
            if self.use_fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
                
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, LR: {lr:.2e}"
                )
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"Validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float, output_dir: str):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'loss': loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss
        }
        
        torch.save(training_state, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint"""
        # Load model
        self.model = HybridTransformerModel.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        
        # Load training state
        training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            
            self.global_step = training_state['global_step']
            self.best_loss = training_state['best_loss']
            
            if self.optimizer and 'optimizer_state_dict' in training_state:
                self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in training_state:
                self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
            
            self.logger.info(f"Checkpoint loaded from {checkpoint_dir}")
            return training_state['epoch']
        
        return 0
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
              num_epochs: int, output_dir: str, save_steps: int = 500):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train epoch
            train_loss = self.train_epoch(train_dataloader, epoch)
            
            # Evaluate
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, output_dir)
                    self.logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if epoch % save_steps == 0:
                self.save_checkpoint(epoch, train_loss, output_dir)
        
        # Save final model
        final_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Final model saved to {final_dir}")


def load_training_data(data_path: str, max_samples: Optional[int] = None) -> List[str]:
    """Load training data from file"""
    texts = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
                
                if max_samples and len(texts) >= max_samples:
                    break
    
    return texts


def main():
    parser = argparse.ArgumentParser(description="Train Mistral model")
    parser.add_argument("--model_config", required=True, help="Path to model config file")
    parser.add_argument("--data_path", required=True, help="Path to training data file")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for trained model")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--use_fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--max_samples", type=int, help="Maximum number of training samples")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--save_steps", type=int, default=500, help="Steps between checkpoints")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_config):
        print(f"Error: Model config file not found: {args.model_config}")
        sys.exit(1)
    
    if not os.path.exists(args.data_path):
        print(f"Error: Training data file not found: {args.data_path}")
        sys.exit(1)
    
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer directory not found: {args.tokenizer_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.model_config, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    config = ModelConfig(**config_dict)
    
    # Load tokenizer
    try:
        tokenizer = HybridTokenizer.from_pretrained(args.tokenizer_path)
    except:
        print("Warning: Hybrid tokenizer not found, using custom tokenizer")
        tokenizer = CustomTokenizer.from_pretrained(args.tokenizer_path)
    
    # Load training data
    print("Loading training data...")
    texts = load_training_data(args.data_path, args.max_samples)
    print(f"Loaded {len(texts)} training samples")
    
    # Split data
    val_size = int(len(texts) * args.val_split)
    train_texts = texts[:-val_size] if val_size > 0 else texts
    val_texts = texts[-val_size:] if val_size > 0 else []
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, args.max_length)
    val_dataset = TextDataset(val_texts, tokenizer, args.max_length) if val_texts else None
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    ) if val_dataset else None
    
    # Initialize trainer
    trainer = MistralTrainer(config, tokenizer, args.device)
    
    # Setup optimizations
    trainer.setup_optimizer(args.learning_rate, args.weight_decay)
    trainer.setup_scheduler(len(train_dataloader) * args.num_epochs, args.warmup_steps)
    trainer.setup_mixed_precision(args.use_fp16)
    trainer.setup_gradient_checkpointing(args.gradient_checkpointing)
    
    # Train model
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        save_steps=args.save_steps
    )
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
