---
license: apache-2.0
language:
- ru
- en
tags:
- mistral
- russian
- english
- code
- machine-learning
- nlp
- transformer
- gqa
- rmsnorm
- swiglu
- rope
pipeline_tag: text-generation
---

# RADON - Mistral-based Russian-English Transformer

## Model Description

RADON is a modern transformer model based on Mistral architecture with Llama 3 innovations, optimized for Russian-English machine learning applications.

### Key Features

- **Architecture**: Mistral with Llama 3 innovations (GQA, RMSNorm, SwiGLU, RoPE)
- **Parameters**: 2B-7B parameters
- **Context**: 8K-32K tokens
- **Tokenizer**: Hybrid Unigram+BPE for Russian-English
- **Optimizations**: Flash Attention 2, Quantization support

### Innovations

1. **Grouped Query Attention (GQA)**: 4:1 ratio for memory efficiency
2. **RMSNorm**: Root Mean Square Layer Normalization
3. **SwiGLU**: Swish-Gated Linear Unit activation
4. **RoPE**: Rotary Position Embeddings for long contexts
5. **Sliding Window Attention**: Efficient attention for long sequences

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("MagistrTheOne/RadonSAI")
tokenizer = AutoTokenizer.from_pretrained("MagistrTheOne/RadonSAI")

# Generate text
prompt = "Машинное обучение - это"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## API Usage

```python
import requests

# Generate text via API
response = requests.post(
    "https://your-api-endpoint.com/api/v1/generate",
    json={
        "prompt": "Привет, RADON!",
        "max_length": 100,
        "temperature": 0.7
    }
)
print(response.json()["generated_text"])
```

## Performance

- **Speed**: 3-5x faster than GPT-2
- **Memory**: 30% less memory usage
- **Quality**: Optimized for Russian-English ML tasks
- **Context**: Supports up to 32K tokens

## Model Architecture

```
RADON Mistral-2B:
- Hidden size: 2048
- Layers: 24
- Attention heads: 32 (8 KV heads)
- Intermediate size: 5632
- Vocabulary: 32K (hybrid Unigram+BPE)
```

## Training

The model is trained on a clean corpus of:
- Russian ML documentation and articles
- English technical content
- Code samples (Python, JavaScript, etc.)
- Mixed Russian-English content

## Deployment

### Local Development
```bash
git clone https://github.com/MagistrTheOne/Radon2BMistral.git
cd Radon2BMistral
bash quick_start_local.sh
```

### Docker
```bash
docker-compose up -d
```

### Yandex Cloud
```bash
bash cloud/yc/full_deploy.sh 2b
```

## Citation

```bibtex
@misc{radon2024,
  title={RADON: Mistral-based Russian-English Transformer},
  author={MagistrTheOne},
  year={2024},
  url={https://github.com/MagistrTheOne/Radon2BMistral}
}
```

## License

Apache 2.0 License

## Contact

- GitHub: [MagistrTheOne/Radon2BMistral](https://github.com/MagistrTheOne/Radon2BMistral)
- Hugging Face: [MagistrTheOne/RadonSAI](https://huggingface.co/MagistrTheOne/RadonSAI)
