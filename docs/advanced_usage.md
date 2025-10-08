# 🔬 RADON Advanced Usage Guide

Advanced techniques and optimizations for RADON

## 🏗️ Architecture Deep Dive

### Mistral + Llama 3 Innovations

RADON combines Mistral architecture with Llama 3 innovations:

```python
# Key architectural components
from models.components import (
    RMSNorm,           # Root Mean Square Layer Normalization
    SwiGLU,            # Swish-Gated Linear Unit
    RoPE,              # Rotary Position Embeddings
    GroupedQueryAttention,  # GQA for memory efficiency
    SlidingWindowAttention # Efficient long-context processing
)
```

### Model Configuration
```python
from models.config import ModelConfig

# Custom configuration
config = ModelConfig(
    model_name="radon",
    model_type="mistral",
    hidden_size=2048,
    num_layers=24,
    num_attention_heads=32,
    num_kv_heads=8,  # GQA ratio 4:1
    intermediate_size=5632,
    max_position_embeddings=8192,
    sliding_window=4096,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    activation_function="silu"
)
```

## 🚀 Performance Optimization

### Flash Attention 2 Integration
```python
# Enable Flash Attention 2 for 2x speedup
model = AutoModelForCausalLM.from_pretrained(
    "MagistrTheOne/RadonSAI-Pretrained",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```

### Memory Optimization
```python
# Gradient checkpointing for training
model.gradient_checkpointing_enable()

# Memory-efficient attention
model.config.use_cache = False
model.config.output_attentions = False
model.config.output_hidden_states = False
```

### Quantization Options
```python
from transformers import BitsAndBytesConfig

# INT8 quantization
int8_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# INT4 quantization (experimental)
int4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "MagistrTheOne/RadonSAI-Pretrained",
    quantization_config=int8_config,
    device_map="auto"
)
```

## 🎯 Advanced Generation

### Custom Generation Parameters
```python
def advanced_generate(
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3
):
    """Advanced text generation with fine-tuned parameters"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response
```

### Streaming Generation
```python
def stream_generate(prompt: str, max_length: int = 200):
    """Stream text generation for real-time applications"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Sample next token
            next_token = torch.multinomial(
                torch.softmax(next_token_logits / 0.7, dim=-1), 
                num_samples=1
            )
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Add to generated tokens
            generated_tokens.append(next_token.item())
            
            # Update inputs for next iteration
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=1)
            
            # Yield partial result
            partial_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            yield partial_text
    
    # Return final result
    final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    yield final_text

# Usage
for partial_text in stream_generate("Машинное обучение - это"):
    print(partial_text, end="", flush=True)
```

## 🔧 Fine-tuning

### Prepare Training Data
```python
from datasets import Dataset

def prepare_training_data(texts: List[str], max_length: int = 512):
    """Prepare data for fine-tuning"""
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

# Example training data
training_texts = [
    "Машинное обучение - это подраздел искусственного интеллекта",
    "Machine learning is a subset of artificial intelligence",
    "def calculate_accuracy(y_true, y_pred): return (y_true == y_pred).mean()"
]

train_dataset = prepare_training_data(training_texts)
```

### Fine-tuning with LoRA
```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Training setup
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./radon-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=True,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="cosine"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()
```

## 🌍 Multilingual Optimization

### Language-Specific Generation
```python
def generate_by_language(prompt: str, language: str = "auto"):
    """Generate text optimized for specific language"""
    
    # Language-specific parameters
    language_configs = {
        "russian": {
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        },
        "english": {
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.05
        },
        "code": {
            "temperature": 0.3,
            "top_p": 0.8,
            "repetition_penalty": 1.2
        }
    }
    
    config = language_configs.get(language, language_configs["english"])
    
    return advanced_generate(prompt, **config)

# Usage
russian_text = generate_by_language("Объясни нейронные сети:", "russian")
english_text = generate_by_language("Explain neural networks:", "english")
code_text = generate_by_language("def train_model():", "code")
```

### Cross-lingual Translation
```python
def translate_text(text: str, target_language: str):
    """Translate text using RADON"""
    
    translation_prompts = {
        "ru_to_en": f"Переведи на английский: {text}",
        "en_to_ru": f"Переведи на русский: {text}",
        "ru_to_code": f"Напиши код на Python для: {text}",
        "en_to_code": f"Write Python code for: {text}"
    }
    
    prompt = translation_prompts.get(target_language, text)
    return generate_text(prompt, max_length=150)
```

## 📊 Benchmarking and Evaluation

### Performance Benchmarking
```python
import time
import psutil
import GPUtil

def benchmark_model(prompts: List[str], iterations: int = 5):
    """Comprehensive model benchmarking"""
    
    results = {
        "generation_times": [],
        "memory_usage": [],
        "tokens_per_second": [],
        "gpu_utilization": []
    }
    
    for iteration in range(iterations):
        for prompt in prompts:
            start_time = time.time()
            
            # Generate text
            response = generate_text(prompt, max_length=100)
            
            generation_time = time.time() - start_time
            tokens_generated = len(response.split())
            tokens_per_second = tokens_generated / generation_time
            
            # Memory usage
            cpu_memory = psutil.virtual_memory().percent
            gpu_memory = 0
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_memory = gpus[0].memoryUtil * 100
            except:
                pass
            
            results["generation_times"].append(generation_time)
            results["memory_usage"].append(cpu_memory)
            results["tokens_per_second"].append(tokens_per_second)
            results["gpu_utilization"].append(gpu_memory)
    
    # Calculate statistics
    stats = {
        "avg_generation_time": np.mean(results["generation_times"]),
        "avg_tokens_per_second": np.mean(results["tokens_per_second"]),
        "avg_memory_usage": np.mean(results["memory_usage"]),
        "avg_gpu_utilization": np.mean(results["gpu_utilization"])
    }
    
    return stats
```

### Quality Evaluation
```python
def evaluate_quality(test_cases: List[Dict[str, str]]):
    """Evaluate generation quality"""
    
    quality_scores = []
    
    for test_case in test_cases:
        prompt = test_case["prompt"]
        expected_keywords = test_case["expected_keywords"]
        
        response = generate_text(prompt)
        response_lower = response.lower()
        
        # Calculate keyword match score
        keyword_matches = sum(1 for keyword in expected_keywords 
                              if keyword.lower() in response_lower)
        keyword_score = keyword_matches / len(expected_keywords)
        
        quality_scores.append(keyword_score)
    
    return {
        "average_quality": np.mean(quality_scores),
        "quality_scores": quality_scores
    }
```

## 🔌 API Integration

### OpenAI-Compatible API
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="RADON API")

class ChatCompletionRequest(BaseModel):
    model: str = "radon"
    messages: List[Dict[str, str]]
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    
    # Extract prompt from messages
    prompt = request.messages[-1]["content"]
    
    # Generate response
    response = generate_text(
        prompt, 
        max_length=request.max_tokens,
        temperature=request.temperature
    )
    
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response.split()),
            "total_tokens": len(prompt.split()) + len(response.split())
        }
    }

# Run API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 🚀 Production Deployment

### Docker Optimization
```dockerfile
# Multi-stage build for production
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Install dependencies
RUN apt-get update && apt-get install -y python3.9 python3-pip
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Copy optimized model and dependencies
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY . /app
WORKDIR /app

# Optimize for production
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "api/app.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: radon-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: radon-api
  template:
    metadata:
      labels:
        app: radon-api
    spec:
      containers:
      - name: radon-api
        image: radon-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "12Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
```

## 🔍 Monitoring and Logging

### Performance Monitoring
```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('radon.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('radon')

def log_generation(prompt: str, response: str, generation_time: float):
    """Log generation metrics"""
    logger.info(f"Generation completed in {generation_time:.3f}s")
    logger.info(f"Prompt: {prompt[:50]}...")
    logger.info(f"Response: {response[:50]}...")
    logger.info(f"Tokens per second: {len(response.split()) / generation_time:.2f}")
```

### Health Checks
```python
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test model inference
        test_prompt = "Test"
        test_response = generate_text(test_prompt, max_length=10)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
            "device": str(next(model.parameters()).device) if model else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

## 🎯 Best Practices

### Memory Management
- Use FP16 for 50% memory reduction
- Enable gradient checkpointing for training
- Use smaller batch sizes for inference
- Monitor GPU memory usage

### Performance Optimization
- Use Flash Attention 2 for 2x speedup
- Enable KV cache for repeated generation
- Batch multiple requests together
- Optimize for your specific hardware

### Quality Control
- Use appropriate temperature settings
- Implement repetition penalty
- Monitor generation quality
- Use validation datasets

---

**For more advanced techniques, see the full documentation at [GitHub](https://github.com/MagistrTheOne/Radon2BMistral)**
