# 🚀 RADON Quick Start Guide

Get up and running with RADON in minutes!

## 📋 Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ VRAM (RTX 4070 recommended)
- 16GB+ RAM

## ⚡ Quick Installation

### 1. Clone Repository
```bash
git clone https://github.com/MagistrTheOne/Radon2BMistral.git
cd Radon2BMistral
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Flash Attention (Optional)
```bash
pip install flash-attn --no-build-isolation
```

## 🎯 Basic Usage

### Load RADON Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load RADON model
model = AutoModelForCausalLM.from_pretrained("MagistrTheOne/RadonSAI-Pretrained")
tokenizer = AutoTokenizer.from_pretrained("MagistrTheOne/RadonSAI-Pretrained")

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Generate Text
```python
def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.7):
    """Generate text with RADON"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

# Test generation
prompt = "Машинное обучение - это"
response = generate_text(prompt)
print(f"Prompt: {prompt}")
print(f"RADON Response: {response}")
```

## 🌍 Multilingual Examples

### Russian Text Generation
```python
russian_prompts = [
    "Машинное обучение - это",
    "Объясни что такое нейронные сети:",
    "Создай нейронную сеть на Python:"
]

for prompt in russian_prompts:
    response = generate_text(prompt)
    print(f"Russian: {prompt}")
    print(f"RADON: {response}\n")
```

### English Text Generation
```python
english_prompts = [
    "Machine learning is",
    "Explain what neural networks are:",
    "Create a neural network in Python:"
]

for prompt in english_prompts:
    response = generate_text(prompt)
    print(f"English: {prompt}")
    print(f"RADON: {response}\n")
```

### Code Generation
```python
code_prompts = [
    "def calculate_accuracy(y_true, y_pred):",
    "class NeuralNetwork:",
    "import torch.nn as nn"
]

for prompt in code_prompts:
    response = generate_text(prompt, max_length=200)
    print(f"Code: {prompt}")
    print(f"RADON Generated:\n```python\n{response}\n```\n")
```

## 🎮 Interactive Demo

### Gradio Interface
```python
import gradio as gr

def create_demo():
    def generate_with_metrics(prompt, max_length, temperature):
        start_time = time.time()
        response = generate_text(prompt, max_length, temperature)
        generation_time = time.time() - start_time
        
        return response, f"Generated in {generation_time:.3f}s"
    
    with gr.Blocks(title="RADON Demo") as demo:
        gr.Markdown("# 🤖 RADON Interactive Demo")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Enter your prompt", lines=3)
                max_length = gr.Slider(10, 500, 100, label="Max Length")
                temperature = gr.Slider(0.1, 2.0, 0.7, label="Temperature")
                generate_btn = gr.Button("🚀 Generate", variant="primary")
            
            with gr.Column():
                output = gr.Textbox(label="Generated Text", lines=10)
                metrics = gr.Textbox(label="Metrics")
        
        generate_btn.click(
            fn=generate_with_metrics,
            inputs=[prompt, max_length, temperature],
            outputs=[output, metrics]
        )
    
    return demo

# Launch demo
demo = create_demo()
demo.launch(share=True)
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build Docker image
docker build -t radon-api .

# Run with GPU support
docker run --gpus all -p 8000:8000 radon-api

# Run with CPU only
docker run -p 8000:8000 radon-api
```

### Docker Compose
```bash
# Start with GPU
docker-compose up -d

# Start CPU-only version
docker-compose --profile cpu up -d
```

## 🚀 API Usage

### Start API Server
```bash
python api/app.py
```

### Generate Text via API
```python
import requests

# Generate text
response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={
        "prompt": "Машинное обучение - это",
        "max_length": 100,
        "temperature": 0.7
    }
)

result = response.json()
print(f"Generated: {result['generated_text']}")
print(f"Model: {result['model_name']}")
print(f"Time: {result['generation_time']}s")
```

## 🔧 Optimization for RTX 4070

### Enable FP16
```python
model = AutoModelForCausalLM.from_pretrained(
    "MagistrTheOne/RadonSAI-Pretrained",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Enable Flash Attention 2
```python
# Install flash-attn first
# pip install flash-attn --no-build-isolation

model = AutoModelForCausalLM.from_pretrained(
    "MagistrTheOne/RadonSAI-Pretrained",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```

### INT8 Quantization
```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    "MagistrTheOne/RadonSAI-Pretrained",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## 📊 Performance Tips

### Memory Optimization
- Use FP16 for 50% memory reduction
- Enable gradient checkpointing for training
- Use smaller batch sizes for inference
- Consider INT8 quantization for deployment

### Speed Optimization
- Use Flash Attention 2 for 2x speedup
- Enable KV cache for repeated generation
- Use batch processing for multiple prompts
- Optimize for your specific hardware

## 🆘 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Reduce batch size or use CPU
model = AutoModelForCausalLM.from_pretrained(
    "MagistrTheOne/RadonSAI-Pretrained",
    torch_dtype=torch.float16,
    device_map="cpu"  # Use CPU if GPU memory insufficient
)
```

**Slow Generation**
```python
# Enable optimizations
model = AutoModelForCausalLM.from_pretrained(
    "MagistrTheOne/RadonSAI-Pretrained",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
```

**Import Errors**
```bash
# Install missing dependencies
pip install flash-attn accelerate bitsandbytes
```

## 📚 Next Steps

1. **Explore Examples**: Check out `examples/` directory
2. **Run Benchmarks**: Use `scripts/benchmark_mistral.py`
3. **Fine-tune Model**: Follow `docs/fine_tuning.md`
4. **Deploy to Cloud**: See `docs/deployment.md`
5. **Join Community**: [GitHub Discussions](https://github.com/MagistrTheOne/Radon2BMistral/discussions)

## 🔗 Resources

- **GitHub**: [MagistrTheOne/Radon2BMistral](https://github.com/MagistrTheOne/Radon2BMistral)
- **Hugging Face**: [MagistrTheOne/RadonSAI](https://huggingface.co/MagistrTheOne/RadonSAI)
- **Documentation**: [Full Documentation](https://github.com/MagistrTheOne/Radon2BMistral/tree/main/docs)
- **Issues**: [Report Issues](https://github.com/MagistrTheOne/Radon2BMistral/issues)

---

**Created with ❤️ by MagistrTheOne**
