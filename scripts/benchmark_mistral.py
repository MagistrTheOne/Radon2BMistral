#!/usr/bin/env python3
"""
Benchmark script for comparing Mistral vs GPT-2 performance
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.hybrid_model import HybridTransformerModel
from models.config import ModelConfig
from tokenizer.hybrid_tokenizer import HybridTokenizer
from tokenizer.custom_tokenizer import CustomTokenizer


class BenchmarkRunner:
    """Benchmark runner for comparing model performance"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.results = {}
        
    def load_model(self, config_path: str, model_type: str = "mistral") -> Tuple[Any, Any]:
        """Load model and tokenizer"""
        print(f"Loading {model_type} model from {config_path}...")
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = ModelConfig(**config_dict)
        config.model_type = model_type
        
        # Create model
        model = HybridTransformerModel(config)
        model.to(self.device)
        model.eval()
        
        # Load tokenizer (use hybrid for Mistral, custom for GPT-2)
        if model_type == "mistral":
            try:
                tokenizer = HybridTokenizer.from_pretrained("./tokenizer_output")
            except:
                print("Warning: Hybrid tokenizer not found, using custom tokenizer")
                tokenizer = CustomTokenizer.from_pretrained("./tokenizer_output")
        else:
            tokenizer = CustomTokenizer.from_pretrained("./tokenizer_output")
        
        return model, tokenizer
    
    def benchmark_memory_usage(self, model: Any) -> Dict[str, float]:
        """Benchmark memory usage"""
        print("Benchmarking memory usage...")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Measure model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)  # GB
        
        # Measure peak memory during forward pass
        dummy_input = torch.randint(0, 1000, (1, 100), device=self.device)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            torch.cuda.empty_cache()
        else:
            peak_memory = 0.0
        
        return {
            "model_size_gb": model_size,
            "peak_memory_gb": peak_memory,
            "memory_efficiency": model_size / max(peak_memory, 0.001)
        }
    
    def benchmark_generation_speed(self, model: Any, tokenizer: Any, prompts: List[str], 
                                 max_length: int = 100, num_runs: int = 5) -> Dict[str, float]:
        """Benchmark text generation speed"""
        print(f"Benchmarking generation speed with {len(prompts)} prompts, {num_runs} runs...")
        
        times = []
        tokens_generated = []
        
        for run in range(num_runs):
            for prompt in prompts:
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Measure generation time
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        max_length=max_length,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Count generated tokens
                generated_tokens = outputs[0].size(1) - inputs["input_ids"].size(1)
                
                times.append(generation_time)
                tokens_generated.append(generated_tokens)
        
        # Calculate statistics
        avg_time = np.mean(times)
        avg_tokens = np.mean(tokens_generated)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        return {
            "avg_generation_time": avg_time,
            "avg_tokens_generated": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "std_generation_time": np.std(times),
            "std_tokens_generated": np.std(tokens_generated)
        }
    
    def benchmark_attention_efficiency(self, model: Any, tokenizer: Any, 
                                     sequence_lengths: List[int]) -> Dict[str, Any]:
        """Benchmark attention efficiency for different sequence lengths"""
        print("Benchmarking attention efficiency...")
        
        results = {}
        
        for seq_len in sequence_lengths:
            print(f"  Testing sequence length: {seq_len}")
            
            # Create dummy input
            dummy_input = torch.randint(0, 1000, (1, seq_len), device=self.device)
            
            # Measure forward pass time
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(dummy_input)
            
            end_time = time.time()
            forward_time = end_time - start_time
            
            # Measure memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                torch.cuda.empty_cache()
            else:
                memory_used = 0.0
            
            results[f"seq_len_{seq_len}"] = {
                "forward_time": forward_time,
                "memory_used_mb": memory_used,
                "tokens_per_second": seq_len / forward_time if forward_time > 0 else 0
            }
        
        return results
    
    def benchmark_quality(self, model: Any, tokenizer: Any, 
                         test_prompts: List[str]) -> Dict[str, float]:
        """Benchmark generation quality (perplexity-based)"""
        print("Benchmarking generation quality...")
        
        perplexities = []
        
        for prompt in test_prompts:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate continuation
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    max_length=inputs["input_ids"].size(1) + 50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Calculate perplexity on generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_tokens = tokenizer(generated_text, return_tensors="pt", add_special_tokens=True)
            generated_tokens = {k: v.to(self.device) for k, v in generated_tokens.items()}
            
            # Forward pass for perplexity calculation
            with torch.no_grad():
                logits = model(generated_tokens["input_ids"]).logits
                
                # Calculate perplexity
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = generated_tokens["input_ids"][..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )
                
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
        
        return {
            "avg_perplexity": np.mean(perplexities),
            "std_perplexity": np.std(perplexities),
            "quality_score": 1.0 / (1.0 + np.mean(perplexities))  # Higher is better
        }
    
    def run_full_benchmark(self, mistral_config: str, gpt2_config: str, 
                          num_runs: int = 5) -> Dict[str, Any]:
        """Run full benchmark comparing Mistral vs GPT-2"""
        print("Starting full benchmark...")
        
        # Test prompts
        test_prompts = [
            "Привет! Как дела?",
            "Hello! How are you?",
            "Машинное обучение - это",
            "Machine learning is",
            "def hello_world():",
            "import torch",
            "Русский текст для тестирования качества генерации.",
            "English text for testing generation quality."
        ]
        
        # Sequence lengths for attention benchmark
        sequence_lengths = [100, 500, 1000, 2000, 4000, 8000]
        
        results = {}
        
        # Benchmark Mistral
        print("\n=== Benchmarking Mistral ===")
        mistral_model, mistral_tokenizer = self.load_model(mistral_config, "mistral")
        
        results["mistral"] = {
            "memory": self.benchmark_memory_usage(mistral_model),
            "generation_speed": self.benchmark_generation_speed(
                mistral_model, mistral_tokenizer, test_prompts, num_runs=num_runs
            ),
            "attention_efficiency": self.benchmark_attention_efficiency(
                mistral_model, mistral_tokenizer, sequence_lengths
            ),
            "quality": self.benchmark_quality(mistral_model, mistral_tokenizer, test_prompts)
        }
        
        # Clean up Mistral model
        del mistral_model, mistral_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Benchmark GPT-2
        print("\n=== Benchmarking GPT-2 ===")
        gpt2_model, gpt2_tokenizer = self.load_model(gpt2_config, "gpt2")
        
        results["gpt2"] = {
            "memory": self.benchmark_memory_usage(gpt2_model),
            "generation_speed": self.benchmark_generation_speed(
                gpt2_model, gpt2_tokenizer, test_prompts, num_runs=num_runs
            ),
            "attention_efficiency": self.benchmark_attention_efficiency(
                gpt2_model, gpt2_tokenizer, sequence_lengths
            ),
            "quality": self.benchmark_quality(gpt2_model, gpt2_tokenizer, test_prompts)
        }
        
        # Calculate improvements
        results["improvements"] = self.calculate_improvements(results["mistral"], results["gpt2"])
        
        return results
    
    def calculate_improvements(self, mistral_results: Dict, gpt2_results: Dict) -> Dict[str, float]:
        """Calculate improvement metrics"""
        improvements = {}
        
        # Memory efficiency
        mistral_memory = mistral_results["memory"]["peak_memory_gb"]
        gpt2_memory = gpt2_results["memory"]["peak_memory_gb"]
        improvements["memory_reduction"] = (gpt2_memory - mistral_memory) / gpt2_memory * 100
        
        # Generation speed
        mistral_speed = mistral_results["generation_speed"]["tokens_per_second"]
        gpt2_speed = gpt2_results["generation_speed"]["tokens_per_second"]
        improvements["speed_improvement"] = (mistral_speed - gpt2_speed) / gpt2_speed * 100
        
        # Quality
        mistral_quality = mistral_results["quality"]["quality_score"]
        gpt2_quality = gpt2_results["quality"]["quality_score"]
        improvements["quality_improvement"] = (mistral_quality - gpt2_quality) / gpt2_quality * 100
        
        return improvements
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a formatted way"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        # Memory comparison
        print("\n📊 MEMORY USAGE")
        print("-" * 40)
        mistral_memory = results["mistral"]["memory"]
        gpt2_memory = results["gpt2"]["memory"]
        
        print(f"Mistral Model Size: {mistral_memory['model_size_gb']:.2f} GB")
        print(f"Mistral Peak Memory: {mistral_memory['peak_memory_gb']:.2f} GB")
        print(f"GPT-2 Model Size: {gpt2_memory['model_size_gb']:.2f} GB")
        print(f"GPT-2 Peak Memory: {gpt2_memory['peak_memory_gb']:.2f} GB")
        print(f"Memory Reduction: {results['improvements']['memory_reduction']:.1f}%")
        
        # Speed comparison
        print("\n⚡ GENERATION SPEED")
        print("-" * 40)
        mistral_speed = results["mistral"]["generation_speed"]
        gpt2_speed = results["gpt2"]["generation_speed"]
        
        print(f"Mistral Tokens/sec: {mistral_speed['tokens_per_second']:.1f}")
        print(f"GPT-2 Tokens/sec: {gpt2_speed['tokens_per_second']:.1f}")
        print(f"Speed Improvement: {results['improvements']['speed_improvement']:.1f}%")
        
        # Quality comparison
        print("\n🎯 GENERATION QUALITY")
        print("-" * 40)
        mistral_quality = results["mistral"]["quality"]
        gpt2_quality = results["gpt2"]["quality"]
        
        print(f"Mistral Perplexity: {mistral_quality['avg_perplexity']:.2f}")
        print(f"GPT-2 Perplexity: {gpt2_quality['avg_perplexity']:.2f}")
        print(f"Quality Improvement: {results['improvements']['quality_improvement']:.1f}%")
        
        # Attention efficiency
        print("\n🧠 ATTENTION EFFICIENCY")
        print("-" * 40)
        mistral_attention = results["mistral"]["attention_efficiency"]
        gpt2_attention = results["gpt2"]["attention_efficiency"]
        
        for seq_len in [100, 1000, 4000, 8000]:
            if f"seq_len_{seq_len}" in mistral_attention:
                mistral_tps = mistral_attention[f"seq_len_{seq_len}"]["tokens_per_second"]
                gpt2_tps = gpt2_attention[f"seq_len_{seq_len}"]["tokens_per_second"]
                improvement = (mistral_tps - gpt2_tps) / gpt2_tps * 100 if gpt2_tps > 0 else 0
                print(f"Seq Length {seq_len}: Mistral {mistral_tps:.1f} tps, GPT-2 {gpt2_tps:.1f} tps ({improvement:+.1f}%)")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mistral vs GPT-2")
    parser.add_argument("--mistral_config", required=True, help="Path to Mistral config file")
    parser.add_argument("--gpt2_config", required=True, help="Path to GPT-2 config file")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Check if config files exist
    if not os.path.exists(args.mistral_config):
        print(f"Error: Mistral config file not found: {args.mistral_config}")
        sys.exit(1)
    
    if not os.path.exists(args.gpt2_config):
        print(f"Error: GPT-2 config file not found: {args.gpt2_config}")
        sys.exit(1)
    
    # Run benchmark
    runner = BenchmarkRunner(device=args.device)
    results = runner.run_full_benchmark(args.mistral_config, args.gpt2_config, args.num_runs)
    
    # Print results
    runner.print_results(results)
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()