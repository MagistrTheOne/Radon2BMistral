# 📚 RADON API Reference

Complete API documentation for RADON

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("MagistrTheOne/RadonSAI-Pretrained")
tokenizer = AutoTokenizer.from_pretrained("MagistrTheOne/RadonSAI-Pretrained")
```

## 🤖 Model Classes

### MistralForCausalLM

Main model class for text generation.

```python
from models.mistral_model import MistralForCausalLM
from models.config import ModelConfig

# Initialize model
config = ModelConfig.from_pretrained("configs/model_config_mistral_2b.json")
model = MistralForCausalLM(config)
```

#### Methods

##### `forward(input_ids, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None)`

Forward pass through the model.

**Parameters:**
- `input_ids` (torch.LongTensor): Input token IDs
- `attention_mask` (torch.Tensor, optional): Attention mask
- `position_ids` (torch.LongTensor, optional): Position indices
- `past_key_values` (List[Tuple[torch.Tensor, torch.Tensor]], optional): Cached key/value pairs
- `inputs_embeds` (torch.FloatTensor, optional): Input embeddings
- `labels` (torch.LongTensor, optional): Labels for language modeling
- `use_cache` (bool, optional): Whether to use cache
- `output_attentions` (bool, optional): Whether to output attention weights
- `output_hidden_states` (bool, optional): Whether to output hidden states
- `return_dict` (bool, optional): Whether to return a dictionary

**Returns:**
- `CausalLMOutputWithPast`: Model outputs

##### `generate(input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9, do_sample=True, pad_token_id=None, eos_token_id=None)`

Generate text using the model.

**Parameters:**
- `input_ids` (torch.LongTensor): Input token IDs
- `max_length` (int): Maximum length of generated text
- `temperature` (float): Sampling temperature
- `top_k` (int): Top-k sampling parameter
- `top_p` (float): Top-p sampling parameter
- `do_sample` (bool): Whether to use sampling
- `pad_token_id` (int, optional): Padding token ID
- `eos_token_id` (int, optional): End-of-sequence token ID

**Returns:**
- `torch.LongTensor`: Generated token IDs

##### `get_system_prompt()`

Get RADON system prompt.

**Returns:**
- `str`: System prompt text

##### `get_model_identity()`

Get RADON model identity information.

**Returns:**
- `dict`: Model identity data

## 🔧 Configuration

### ModelConfig

Configuration class for RADON models.

```python
from models.config import ModelConfig

config = ModelConfig(
    model_name="radon",
    model_type="mistral",
    hidden_size=2048,
    num_layers=24,
    num_attention_heads=32,
    num_kv_heads=8,
    intermediate_size=5632,
    max_position_embeddings=8192,
    sliding_window=4096,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    activation_function="silu"
)
```

#### Parameters

- `model_name` (str): Model name, default "radon"
- `model_type` (str): Model type, default "mistral"
- `hidden_size` (int): Hidden dimension size
- `num_layers` (int): Number of transformer layers
- `num_attention_heads` (int): Number of attention heads
- `num_kv_heads` (int): Number of key-value heads for GQA
- `intermediate_size` (int): Feed-forward network size
- `max_position_embeddings` (int): Maximum sequence length
- `sliding_window` (int): Sliding window size for attention
- `rope_theta` (float): RoPE theta parameter
- `rms_norm_eps` (float): RMSNorm epsilon
- `activation_function` (str): Activation function name

## 🎯 Generation Parameters

### Text Generation

```python
def generate_text(
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0
) -> str:
```

**Parameters:**
- `prompt` (str): Input prompt
- `max_length` (int): Maximum generation length
- `temperature` (float): Sampling temperature (0.1-2.0)
- `top_k` (int): Top-k sampling (1-100)
- `top_p` (float): Top-p sampling (0.1-1.0)
- `do_sample` (bool): Whether to use sampling
- `repetition_penalty` (float): Repetition penalty (0.1-2.0)
- `length_penalty` (float): Length penalty
- `no_repeat_ngram_size` (int): N-gram repetition prevention

**Returns:**
- `str`: Generated text

### Advanced Generation

```python
def advanced_generate(
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
    use_cache: bool = True,
    sliding_window_size: int = None
) -> str:
```

**Additional Parameters:**
- `use_cache` (bool): Whether to use KV cache
- `sliding_window_size` (int): Sliding window size for Mistral

## 🌍 Multilingual Support

### Language Detection

```python
def detect_language(text: str) -> str:
    """Detect text language"""
    # Returns: "russian", "english", "code", or "unknown"
```

### Language-Specific Generation

```python
def generate_by_language(
    prompt: str,
    language: str = "auto",
    max_length: int = 100
) -> str:
```

**Languages:**
- `"russian"`: Optimized for Russian text
- `"english"`: Optimized for English text
- `"code"`: Optimized for code generation
- `"auto"`: Automatic language detection

### Translation

```python
def translate_text(
    text: str,
    target_language: str,
    source_language: str = "auto"
) -> str:
```

**Translation Options:**
- `"ru_to_en"`: Russian to English
- `"en_to_ru"`: English to Russian
- `"ru_to_code"`: Russian description to code
- `"en_to_code"`: English description to code

## 🔌 API Endpoints

### FastAPI Server

Start the API server:
```bash
python api/app.py
```

### Endpoints

#### `POST /api/v1/generate`

Generate text using RADON.

**Request Body:**
```json
{
    "prompt": "Машинное обучение - это",
    "max_length": 100,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "do_sample": true,
    "repetition_penalty": 1.0,
    "sliding_window_size": 4096,
    "use_flash_attention": false,
    "user_id": "optional_user_id",
    "request_id": "optional_request_id"
}
```

**Response:**
```json
{
    "generated_text": "подраздел искусственного интеллекта...",
    "prompt": "Машинное обучение - это",
    "generation_time": 0.234,
    "model_name": "radon",
    "model_type": "mistral",
    "request_id": "uuid-here",
    "model_identity": {
        "name": "RADON",
        "creator": "MagistrTheOne",
        "architecture": "Mistral-based with Llama 3 innovations"
    },
    "system_prompt": "I am RADON, a Mistral-based Russian-English transformer..."
}
```

#### `POST /api/v1/tokenize`

Tokenize text.

**Request Body:**
```json
{
    "text": "Машинное обучение - это",
    "user_id": "optional_user_id"
}
```

**Response:**
```json
{
    "tokens": [1234, 5678, 9012],
    "token_count": 3,
    "text": "Машинное обучение - это"
}
```

#### `POST /api/v1/detokenize`

Detokenize tokens.

**Request Body:**
```json
{
    "tokens": [1234, 5678, 9012],
    "user_id": "optional_user_id"
}
```

**Response:**
```json
{
    "text": "Машинное обучение - это",
    "token_count": 3
}
```

#### `POST /api/v1/switch_model`

Switch model architecture.

**Request Body:**
```json
{
    "model_type": "mistral",
    "config_path": "configs/model_config_mistral_2b.json"
}
```

**Response:**
```json
{
    "success": true,
    "model_type": "mistral",
    "model_name": "radon",
    "message": "Model switched successfully"
}
```

#### `GET /api/v1/model_info`

Get model information.

**Response:**
```json
{
    "model_name": "radon",
    "model_type": "mistral",
    "model_size": "2B",
    "parameters": 2000000000,
    "context_length": 8192,
    "languages": ["russian", "english", "code"],
    "optimizations": ["flash_attention_2", "fp16", "gradient_checkpointing"]
}
```

#### `GET /healthz`

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "model_loaded": true,
    "device": "cuda:0",
    "memory_usage": "8.5GB"
}
```

## 🔧 Optimization APIs

### Flash Attention 2

```python
# Enable Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    "MagistrTheOne/RadonSAI-Pretrained",
    attn_implementation="flash_attention_2"
)
```

### Quantization

```python
from transformers import BitsAndBytesConfig

# INT8 quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    "MagistrTheOne/RadonSAI-Pretrained",
    quantization_config=quantization_config
)
```

### Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Disable unnecessary outputs
model.config.use_cache = False
model.config.output_attentions = False
model.config.output_hidden_states = False
```

## 📊 Benchmarking APIs

### Performance Benchmark

```python
from scripts.benchmark_mistral import BenchmarkRunner

benchmark = BenchmarkRunner()
results = benchmark.run_benchmark()
```

### Russian NLP Evaluation

```python
from benchmarks.russian_nlp_suite import RussianNLPSuite

suite = RussianNLPSuite()
results = suite.run_full_evaluation()
```

### Tier-3 Comparison

```python
from benchmarks.compare_tier3 import Tier3Comparison

comparison = Tier3Comparison()
results = comparison.run_full_comparison()
```

## 🚀 Deployment APIs

### Docker

```bash
# Build image
docker build -t radon-api .

# Run with GPU
docker run --gpus all -p 8000:8000 radon-api

# Run CPU-only
docker run -p 8000:8000 radon-api
```

### Docker Compose

```yaml
version: '3.8'
services:
  radon-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEVICE=cuda
      - USE_FLASH_ATTENTION=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Kubernetes

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
            nvidia.com/gpu: 1
          limits:
            memory: "12Gi"
            nvidia.com/gpu: 1
```

## 🔍 Error Handling

### Common Errors

#### CUDA Out of Memory
```python
# Solution: Use CPU or reduce batch size
model = AutoModelForCausalLM.from_pretrained(
    "MagistrTheOne/RadonSAI-Pretrained",
    device_map="cpu"  # Use CPU
)
```

#### Import Errors
```bash
# Install missing dependencies
pip install flash-attn accelerate bitsandbytes
```

#### Model Loading Errors
```python
# Check model path and permissions
import os
print(os.path.exists("MagistrTheOne/RadonSAI-Pretrained"))
```

### Error Codes

- `400`: Bad Request - Invalid parameters
- `500`: Internal Server Error - Model error
- `503`: Service Unavailable - Model not loaded
- `504`: Gateway Timeout - Generation timeout

## 📚 Examples

### Basic Text Generation
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("MagistrTheOne/RadonSAI-Pretrained")
tokenizer = AutoTokenizer.from_pretrained("MagistrTheOne/RadonSAI-Pretrained")

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate text
response = generate("Машинное обучение - это")
print(response)
```

### Multilingual Generation
```python
# Russian
russian_response = generate("Объясни что такое нейронные сети:")

# English
english_response = generate("Explain what neural networks are:")

# Code
code_response = generate("def calculate_accuracy(y_true, y_pred):")
```

### API Client
```python
import requests

# Generate text via API
response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={
        "prompt": "Машинное обучение - это",
        "max_length": 100,
        "temperature": 0.7
    }
)

result = response.json()
print(result["generated_text"])
```

---

**For more examples and tutorials, visit [GitHub](https://github.com/MagistrTheOne/Radon2BMistral)**
