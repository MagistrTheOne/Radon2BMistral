#!/usr/bin/env python3
"""
Demo script for Mistral model capabilities
"""

import os
import sys
import json
import argparse
import torch
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.hybrid_model import HybridTransformerModel
from models.config import ModelConfig
from tokenizer.hybrid_tokenizer import HybridTokenizer
from tokenizer.custom_tokenizer import CustomTokenizer


class MistralDemo:
    """Demo class for showcasing Mistral capabilities"""
    
    def __init__(self, config_path: str, tokenizer_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        self.config = ModelConfig(**config_dict)
        
        # Load tokenizer
        try:
            self.tokenizer = HybridTokenizer.from_pretrained(tokenizer_path)
        except:
            print("Warning: Hybrid tokenizer not found, using custom tokenizer")
            self.tokenizer = CustomTokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        self.model = HybridTransformerModel(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Demo initialized with {self.config.model_type} model on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def generate_text(self, prompt: str, max_length: int = 200, temperature: float = 0.7, 
                     use_flash_attention: bool = False) -> str:
        """Generate text from prompt"""
        print(f"\nGenerating text for prompt: '{prompt}'")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_flash_attention=use_flash_attention
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove original prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def benchmark_generation_speed(self, prompts: List[str], num_runs: int = 3) -> Dict[str, float]:
        """Benchmark generation speed"""
        print(f"\nBenchmarking generation speed with {len(prompts)} prompts, {num_runs} runs...")
        
        import time
        
        times = []
        tokens_generated = []
        
        for run in range(num_runs):
            for prompt in prompts:
                start_time = time.time()
                
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate text
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        max_length=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Count generated tokens
                generated_tokens = outputs[0].size(1) - inputs["input_ids"].size(1)
                
                times.append(generation_time)
                tokens_generated.append(generated_tokens)
        
        # Calculate statistics
        import numpy as np
        avg_time = np.mean(times)
        avg_tokens = np.mean(tokens_generated)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        return {
            "avg_generation_time": avg_time,
            "avg_tokens_generated": avg_tokens,
            "tokens_per_second": tokens_per_second
        }
    
    def test_long_context(self, max_length: int = 4000) -> str:
        """Test long context generation"""
        print(f"\nTesting long context generation (max_length={max_length})...")
        
        # Create a long prompt
        long_prompt = "Машинное обучение - это область искусственного интеллекта, которая изучает алгоритмы и статистические модели, используемые компьютерными системами для выполнения задач без явных инструкций. " * 10
        
        return self.generate_text(long_prompt, max_length=max_length)
    
    def test_multilingual(self) -> Dict[str, str]:
        """Test multilingual generation"""
        print("\nTesting multilingual generation...")
        
        prompts = {
            "Russian": "Привет! Расскажи о машинном обучении:",
            "English": "Hello! Tell me about machine learning:",
            "Code": "def hello_world():",
            "Mixed": "Привет! Hello! def hello_world():"
        }
        
        results = {}
        for language, prompt in prompts.items():
            results[language] = self.generate_text(prompt, max_length=150)
        
        return results
    
    def test_model_switching(self):
        """Test model architecture switching"""
        print("\nTesting model architecture switching...")
        
        if self.config.model_type == "hybrid":
            # Test switching to different architectures
            architectures = ["mistral", "gpt2", "t5"]
            
            for arch in architectures:
                try:
                    if arch == "mistral":
                        self.model.switch_to_mistral()
                    elif arch == "gpt2":
                        self.model.switch_to_gpt2()
                    elif arch == "t5":
                        self.model.switch_to_t5()
                    
                    print(f"Switched to {arch} architecture")
                    
                    # Test generation
                    prompt = "Hello, world!"
                    result = self.generate_text(prompt, max_length=50)
                    print(f"Generation result: {result[:100]}...")
                    
                except Exception as e:
                    print(f"Error switching to {arch}: {e}")
        else:
            print("Model switching only available in hybrid mode")
    
    def run_full_demo(self):
        """Run full demonstration"""
        print("="*80)
        print("RADON MISTRAL DEMO")
        print("="*80)
        
        # Test prompts
        test_prompts = [
            "Привет! Как дела?",
            "Hello! How are you?",
            "Машинное обучение - это",
            "Machine learning is",
            "def hello_world():",
            "import torch"
        ]
        
        # 1. Basic generation
        print("\n1. BASIC TEXT GENERATION")
        print("-" * 40)
        
        for prompt in test_prompts[:3]:
            result = self.generate_text(prompt, max_length=100)
            print(f"Prompt: {prompt}")
            print(f"Generated: {result}")
            print()
        
        # 2. Multilingual test
        print("\n2. MULTILINGUAL GENERATION")
        print("-" * 40)
        
        multilingual_results = self.test_multilingual()
        for language, result in multilingual_results.items():
            print(f"{language}: {result}")
            print()
        
        # 3. Long context test
        print("\n3. LONG CONTEXT GENERATION")
        print("-" * 40)
        
        long_result = self.test_long_context(max_length=2000)
        print(f"Long context result: {long_result[:200]}...")
        
        # 4. Speed benchmark
        print("\n4. GENERATION SPEED BENCHMARK")
        print("-" * 40)
        
        speed_results = self.benchmark_generation_speed(test_prompts, num_runs=2)
        print(f"Average generation time: {speed_results['avg_generation_time']:.3f}s")
        print(f"Average tokens generated: {speed_results['avg_tokens_generated']:.1f}")
        print(f"Tokens per second: {speed_results['tokens_per_second']:.1f}")
        
        # 5. Model switching test
        print("\n5. MODEL ARCHITECTURE SWITCHING")
        print("-" * 40)
        
        self.test_model_switching()
        
        # 6. Model information
        print("\n6. MODEL INFORMATION")
        print("-" * 40)
        
        model_info = self.model.get_model_info()
        print(f"Model type: {model_info['model_type']}")
        print(f"Total parameters: {model_info['num_parameters']:,}")
        print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"Hidden size: {self.config.hidden_size}")
        print(f"Number of layers: {self.config.num_layers}")
        print(f"Attention heads: {self.config.num_attention_heads}")
        print(f"Max position embeddings: {self.config.max_position_embeddings}")
        print(f"Sliding window: {getattr(self.config, 'sliding_window', 'N/A')}")
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Mistral model demo")
    parser.add_argument("--config", required=True, help="Path to model config file")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer directory")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--prompt", help="Single prompt to test")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer directory not found: {args.tokenizer}")
        sys.exit(1)
    
    # Initialize demo
    demo = MistralDemo(args.config, args.tokenizer, args.device)
    
    # Run demo
    if args.prompt:
        # Single prompt test
        result = demo.generate_text(
            args.prompt, 
            max_length=args.max_length, 
            temperature=args.temperature,
            use_flash_attention=args.use_flash_attention
        )
        print(f"Generated text: {result}")
    else:
        # Full demo
        demo.run_full_demo()


if __name__ == "__main__":
    main()
