"""
Быстрый тест RADON на локальной машине
"""

import os
import time
import json
import torch
from pathlib import Path
from typing import Dict, Any, List

from models.config import ModelConfig
from models.hybrid_model import HybridTransformerModel
from tokenizer.hybrid_tokenizer import HybridTokenizer


class RADONTester:
    """Тестер для RADON модели"""
    
    def __init__(self, config_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.config = ModelConfig.from_json(config_path)
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Загрузить модель"""
        print(f"🔄 Loading model on {self.device}...")
        start_time = time.time()
        
        # Инициализируем модель
        self.model = HybridTransformerModel(self.config)
        self.model.to(self.device)
        self.model.eval()
        
        # Инициализируем токенизатор
        self.tokenizer = HybridTokenizer()
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        return load_time
    
    def test_memory_usage(self) -> Dict[str, float]:
        """Тест использования памяти"""
        print("🧠 Testing memory usage...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            
            return {
                "memory_allocated_gb": memory_before,
                "memory_reserved_gb": memory_reserved,
                "total_memory_gb": memory_before + memory_reserved
            }
        else:
            return {"memory_allocated_gb": 0, "memory_reserved_gb": 0, "total_memory_gb": 0}
    
    def test_generation_speed(self, prompts: List[str], max_length: int = 100) -> Dict[str, float]:
        """Тест скорости генерации"""
        print("⚡ Testing generation speed...")
        
        times = []
        tokens_generated = []
        
        for i, prompt in enumerate(prompts):
            print(f"  Test {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            start_time = time.time()
            
            # Токенизация
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_length=max_length,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            
            # Декодирование
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_tokens = len(generated_text.split()) - len(prompt.split())
            
            times.append(generation_time)
            tokens_generated.append(new_tokens)
            
            print(f"    Generated {new_tokens} tokens in {generation_time:.2f}s")
        
        # Статистика
        avg_time = sum(times) / len(times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        return {
            "avg_generation_time": avg_time,
            "avg_tokens_generated": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "total_tests": len(prompts)
        }
    
    def test_multilingual(self) -> Dict[str, Any]:
        """Тест многоязычности"""
        print("🌍 Testing multilingual capabilities...")
        
        test_prompts = [
            "Привет! Как дела?",
            "Hello! How are you?",
            "def hello_world():",
            "Машинное обучение - это",
            "Machine learning is",
            "import torch as",
            "Трансформеры используют",
            "Transformers use attention",
            "class NeuralNetwork:",
            "Нейронные сети состоят"
        ]
        
        results = {}
        
        for prompt in test_prompts:
            try:
                # Токенизация
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Генерация
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        max_length=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Декодирование
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                results[prompt] = {
                    "success": True,
                    "generated": generated_text,
                    "input_tokens": inputs["input_ids"].size(1),
                    "output_tokens": outputs.size(1)
                }
                
            except Exception as e:
                results[prompt] = {
                    "success": False,
                    "error": str(e)
                }
        
        success_count = sum(1 for r in results.values() if r.get("success", False))
        
        return {
            "total_tests": len(test_prompts),
            "successful_tests": success_count,
            "success_rate": success_count / len(test_prompts),
            "results": results
        }
    
    def test_context_length(self, max_context: int = 2048) -> Dict[str, Any]:
        """Тест длины контекста"""
        print(f"📏 Testing context length up to {max_context} tokens...")
        
        # Создаем длинный контекст
        long_text = "Машинное обучение " * (max_context // 2)
        
        try:
            # Токенизация
            inputs = self.tokenizer(
                long_text,
                return_tensors="pt",
                add_special_tokens=True,
                max_length=max_context,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            actual_length = inputs["input_ids"].size(1)
            
            # Генерация
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    max_length=actual_length + 50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            
            return {
                "success": True,
                "input_length": actual_length,
                "generation_time": generation_time,
                "max_context_supported": actual_length
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "max_context_supported": 0
            }
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Запустить полный набор тестов"""
        print("🧪 Running RADON test suite...")
        print("=" * 50)
        
        results = {}
        
        # 1. Загрузка модели
        results["model_loading"] = {
            "load_time": self.load_model()
        }
        
        # 2. Использование памяти
        results["memory_usage"] = self.test_memory_usage()
        
        # 3. Тест скорости генерации
        test_prompts = [
            "Привет! Как дела?",
            "Hello! How are you?",
            "def hello_world():",
            "Машинное обучение - это",
            "Machine learning is"
        ]
        results["generation_speed"] = self.test_generation_speed(test_prompts)
        
        # 4. Тест многоязычности
        results["multilingual"] = self.test_multilingual()
        
        # 5. Тест длины контекста
        results["context_length"] = self.test_context_length()
        
        # 6. Информация о модели
        model_info = self.model.get_model_info()
        results["model_info"] = model_info
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Сохранить результаты тестов"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📊 Results saved to {output_file}")


def main():
    """Основная функция"""
    
    print("🚀 RADON Local Test Suite")
    print("=" * 40)
    
    # Проверяем доступность CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Выбираем конфигурацию
    config_path = "configs/model_config_mistral_2b.json"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    # Создаем тестер
    tester = RADONTester(config_path)
    
    # Запускаем тесты
    results = tester.run_full_test_suite()
    
    # Сохраняем результаты
    tester.save_results(results, "test_results.json")
    
    # Выводим краткую сводку
    print("\n📊 Test Results Summary:")
    print(f"  Model: {results['model_info']['model_name']} ({results['model_info']['model_type']})")
    print(f"  Parameters: {results['model_info']['num_parameters']:,}")
    print(f"  Load time: {results['model_loading']['load_time']:.2f}s")
    print(f"  Memory: {results['memory_usage']['total_memory_gb']:.2f} GB")
    print(f"  Speed: {results['generation_speed']['tokens_per_second']:.1f} tokens/sec")
    print(f"  Multilingual: {results['multilingual']['success_rate']:.1%} success")
    print(f"  Context: {results['context_length']['max_context_supported']} tokens")
    
    print("\n✅ Test suite completed!")


if __name__ == "__main__":
    main()
