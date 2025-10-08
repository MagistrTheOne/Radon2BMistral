"""
Tier-3 Model Comparison Benchmark
Compare RADON against GPT-2, DistilGPT-2, DialoGPT
"""

import os
import json
import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import GPUtil
from pathlib import Path


class Tier3Comparison:
    """
    Compare RADON against tier-3 models:
    - GPT-2 (117M, 345M, 774M, 1.5B)
    - DistilGPT-2 (82M)
    - DialoGPT (117M, 345M, 774M)
    """
    
    def __init__(self):
        self.models = {
            "radon": "MagistrTheOne/RadonSAI-Pretrained",
            "gpt2": "gpt2",
            "gpt2-medium": "gpt2-medium", 
            "gpt2-large": "gpt2-large",
            "distilgpt2": "distilgpt2",
            "microsoft/DialoGPT-small": "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium": "microsoft/DialoGPT-medium"
        }
        
        self.loaded_models = {}
        self.results = {}
        
        # Test prompts
        self.test_prompts = {
            "russian_simple": "Машинное обучение - это",
            "russian_complex": "Создай нейронную сеть для классификации изображений на Python:",
            "english_simple": "Machine learning is",
            "english_complex": "Implement a transformer model for natural language processing:",
            "code_python": "def calculate_loss(y_true, y_pred):",
            "code_javascript": "function processData(data) {",
            "multilingual": "Explain machine learning in both Russian and English:",
            "technical": "Describe the architecture of a modern transformer model:"
        }
    
    def load_model(self, model_name: str, model_key: str) -> bool:
        """Load a specific model"""
        print(f"Loading {model_key}: {model_name}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.loaded_models[model_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "name": model_name
            }
            
            print(f"✅ {model_key} loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {model_key}: {e}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load all models for comparison"""
        print("🔄 Loading all models for comparison...")
        
        results = {}
        for model_key, model_name in self.models.items():
            results[model_key] = self.load_model(model_name, model_key)
        
        successful_models = [k for k, v in results.items() if v]
        print(f"✅ Successfully loaded {len(successful_models)} models: {successful_models}")
        
        return results
    
    def benchmark_generation_speed(self) -> Dict[str, Dict[str, float]]:
        """Benchmark text generation speed"""
        print("\n⚡ Benchmarking Generation Speed...")
        
        results = {}
        
        for model_key, model_data in self.loaded_models.items():
            print(f"  Testing {model_key}...")
            
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            # Test with different prompt lengths
            speed_results = {}
            
            for prompt_name, prompt in self.test_prompts.items():
                start_time = time.time()
                
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=inputs["input_ids"].shape[1] + 50,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    generation_time = time.time() - start_time
                    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Calculate metrics
                    input_tokens = inputs["input_ids"].shape[1]
                    output_tokens = outputs.shape[1] - input_tokens
                    tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
                    
                    speed_results[prompt_name] = {
                        "generation_time": generation_time,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tokens_per_second": tokens_per_second,
                        "output_length": len(output_text)
                    }
                    
                except Exception as e:
                    print(f"    Error with {prompt_name}: {e}")
                    speed_results[prompt_name] = {
                        "generation_time": float('inf'),
                        "tokens_per_second": 0,
                        "error": str(e)
                    }
            
            # Calculate average speed
            valid_speeds = [r["tokens_per_second"] for r in speed_results.values() 
                          if "tokens_per_second" in r and r["tokens_per_second"] > 0]
            
            speed_results["average_tokens_per_second"] = np.mean(valid_speeds) if valid_speeds else 0
            results[model_key] = speed_results
            
            print(f"    Average speed: {speed_results['average_tokens_per_second']:.2f} tokens/sec")
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Dict[str, float]]:
        """Benchmark memory usage"""
        print("\n💾 Benchmarking Memory Usage...")
        
        results = {}
        
        for model_key, model_data in self.loaded_models.items():
            print(f"  Testing {model_key}...")
            
            model = model_data["model"]
            
            # Get GPU memory if available
            gpu_memory = {}
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_memory = {
                        "gpu_memory_used": gpus[0].memoryUsed,
                        "gpu_memory_total": gpus[0].memoryTotal,
                        "gpu_memory_percent": gpus[0].memoryUtil * 100
                    }
            except:
                gpu_memory = {}
            
            # Get CPU memory
            cpu_memory = {
                "cpu_memory_used": psutil.virtual_memory().used / (1024**3),  # GB
                "cpu_memory_total": psutil.virtual_memory().total / (1024**3),  # GB
                "cpu_memory_percent": psutil.virtual_memory().percent
            }
            
            # Count model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024**2)
            
            results[model_key] = {
                **gpu_memory,
                **cpu_memory,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": model_size_mb,
                "model_size_gb": model_size_mb / 1024
            }
            
            print(f"    Parameters: {total_params:,}")
            print(f"    Model size: {model_size_mb:.1f} MB")
            if gpu_memory:
                print(f"    GPU memory: {gpu_memory.get('gpu_memory_used', 0)} MB")
    
    def benchmark_quality(self) -> Dict[str, Dict[str, float]]:
        """Benchmark generation quality using perplexity"""
        print("\n🎯 Benchmarking Generation Quality...")
        
        results = {}
        
        # Test sentences for perplexity calculation
        test_sentences = [
            "Машинное обучение - это подраздел искусственного интеллекта",
            "Machine learning is a subset of artificial intelligence",
            "def calculate_accuracy(y_true, y_pred): return (y_true == y_pred).mean()",
            "Neural networks are inspired by biological neural networks"
        ]
        
        for model_key, model_data in self.loaded_models.items():
            print(f"  Testing {model_key}...")
            
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            quality_results = {}
            
            for i, sentence in enumerate(test_sentences):
                try:
                    # Tokenize sentence
                    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    # Calculate perplexity
                    with torch.no_grad():
                        outputs = model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss
                        perplexity = torch.exp(loss).item()
                    
                    quality_results[f"sentence_{i+1}"] = {
                        "perplexity": perplexity,
                        "loss": loss.item(),
                        "sentence": sentence[:50] + "..." if len(sentence) > 50 else sentence
                    }
                    
                except Exception as e:
                    print(f"    Error with sentence {i+1}: {e}")
                    quality_results[f"sentence_{i+1}"] = {
                        "perplexity": float('inf'),
                        "error": str(e)
                    }
            
            # Calculate average perplexity
            valid_perplexities = [r["perplexity"] for r in quality_results.values() 
                                if "perplexity" in r and r["perplexity"] != float('inf')]
            
            quality_results["average_perplexity"] = np.mean(valid_perplexities) if valid_perplexities else float('inf')
            results[model_key] = quality_results
            
            print(f"    Average perplexity: {quality_results['average_perplexity']:.2f}")
        
        return results
    
    def benchmark_multilingual_capability(self) -> Dict[str, Dict[str, float]]:
        """Benchmark multilingual capabilities"""
        print("\n🌍 Benchmarking Multilingual Capabilities...")
        
        results = {}
        
        # Multilingual test cases
        multilingual_tests = {
            "russian_understanding": {
                "prompt": "Объясни что такое машинное обучение",
                "expected_keywords": ["машинное", "обучение", "алгоритм", "данные"]
            },
            "english_understanding": {
                "prompt": "Explain what machine learning is",
                "expected_keywords": ["machine", "learning", "algorithm", "data"]
            },
            "code_generation": {
                "prompt": "Напиши функцию на Python для сортировки",
                "expected_keywords": ["def", "sort", "python", "function"]
            },
            "translation": {
                "prompt": "Переведи на английский: Искусственный интеллект",
                "expected_keywords": ["artificial", "intelligence", "AI"]
            }
        }
        
        for model_key, model_data in self.loaded_models.items():
            print(f"  Testing {model_key}...")
            
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            capability_results = {}
            
            for test_name, test_data in multilingual_tests.items():
                try:
                    # Generate response
                    inputs = tokenizer(test_data["prompt"], return_tensors="pt", truncation=True, max_length=256)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=inputs["input_ids"].shape[1] + 100,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if response.startswith(test_data["prompt"]):
                        response = response[len(test_data["prompt"]):].strip()
                    
                    # Evaluate response quality
                    response_lower = response.lower()
                    expected_keywords = test_data["expected_keywords"]
                    
                    keyword_matches = sum(1 for keyword in expected_keywords 
                                        if keyword.lower() in response_lower)
                    keyword_score = keyword_matches / len(expected_keywords)
                    
                    capability_results[test_name] = {
                        "keyword_score": keyword_score,
                        "response_length": len(response),
                        "response": response[:100] + "..." if len(response) > 100 else response
                    }
                    
                except Exception as e:
                    print(f"    Error with {test_name}: {e}")
                    capability_results[test_name] = {
                        "keyword_score": 0.0,
                        "error": str(e)
                    }
            
            # Calculate average capability score
            valid_scores = [r["keyword_score"] for r in capability_results.values() 
                          if "keyword_score" in r]
            
            capability_results["average_score"] = np.mean(valid_scores) if valid_scores else 0.0
            results[model_key] = capability_results
            
            print(f"    Average capability: {capability_results['average_score']:.3f}")
        
        return results
    
    def run_full_comparison(self) -> Dict[str, Any]:
        """Run complete tier-3 comparison"""
        print("🚀 Starting RADON vs Tier-3 Models Comparison")
        print("=" * 60)
        
        # Load all models
        load_results = self.load_all_models()
        if not any(load_results.values()):
            return {"error": "Failed to load any models"}
        
        start_time = time.time()
        
        # Run all benchmarks
        self.results = {
            "model_loading": load_results,
            "generation_speed": self.benchmark_generation_speed(),
            "memory_usage": self.benchmark_memory_usage(),
            "quality": self.benchmark_quality(),
            "multilingual_capability": self.benchmark_multilingual_capability()
        }
        
        # Calculate overall rankings
        self._calculate_rankings()
        
        # Add metadata
        self.results["metadata"] = {
            "evaluation_time": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models_tested": len([k for k, v in load_results.items() if v])
        }
        
        print(f"\n⏱️  Total evaluation time: {self.results['metadata']['evaluation_time']:.2f}s")
        
        return self.results
    
    def _calculate_rankings(self):
        """Calculate overall model rankings"""
        print("\n🏆 Calculating Model Rankings...")
        
        # Extract metrics for ranking
        rankings = {}
        
        for model_key in self.loaded_models.keys():
            if model_key not in self.results["generation_speed"]:
                continue
                
            # Speed ranking (higher is better)
            speed_score = self.results["generation_speed"][model_key].get("average_tokens_per_second", 0)
            
            # Quality ranking (lower perplexity is better)
            quality_score = 1.0 / (self.results["quality"][model_key].get("average_perplexity", float('inf')) + 1e-6)
            
            # Capability ranking (higher is better)
            capability_score = self.results["multilingual_capability"][model_key].get("average_score", 0)
            
            # Memory efficiency (lower memory usage is better)
            memory_score = 1.0 / (self.results["memory_usage"][model_key].get("model_size_gb", 1) + 1e-6)
            
            # Overall score (weighted average)
            overall_score = (
                speed_score * 0.3 +
                quality_score * 0.3 +
                capability_score * 0.3 +
                memory_score * 0.1
            )
            
            rankings[model_key] = {
                "speed_score": speed_score,
                "quality_score": quality_score,
                "capability_score": capability_score,
                "memory_score": memory_score,
                "overall_score": overall_score
            }
        
        # Sort by overall score
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1]["overall_score"], reverse=True)
        
        self.results["rankings"] = {
            "by_overall_score": sorted_rankings,
            "detailed_scores": rankings
        }
        
        print("📊 Model Rankings (by overall score):")
        for i, (model_key, scores) in enumerate(sorted_rankings, 1):
            print(f"  {i}. {model_key}: {scores['overall_score']:.3f}")
    
    def save_results(self, output_path: str = "results/tier3_comparison.json"):
        """Save comparison results"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results saved to {output_path}")


def main():
    """Run tier-3 comparison benchmark"""
    comparison = Tier3Comparison()
    results = comparison.run_full_comparison()
    comparison.save_results()
    
    return results


if __name__ == "__main__":
    main()
