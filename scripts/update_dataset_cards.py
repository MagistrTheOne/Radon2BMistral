"""
Update dataset cards with proper YAML metadata
"""

import os
import yaml
from huggingface_hub import HfApi
from pathlib import Path


def create_dataset_card_with_yaml(dataset_name: str, description: str, tags: list) -> str:
    """Создать карточку датасета с YAML метаданными"""
    
    yaml_metadata = {
        "license": "apache-2.0",
        "language": ["ru", "en"],
        "tags": tags,
        "pipeline_tag": "text-generation",
        "size_categories": "1K<n<10K",
        "task_categories": ["text-generation", "text-classification"],
        "source_datasets": ["original"],
        "preprocessing": "text",
        "model": "MagistrTheOne/RadonSAI"
    }
    
    card_content = f"""---
{yaml.dump(yaml_metadata, default_flow_style=False, allow_unicode=True)}---

# {dataset_name}

## Description
{description}

## Usage

### Load Dataset
```python
from datasets import load_dataset

dataset = load_dataset("MagistrTheOne/{dataset_name}")
print(dataset)
```

### Use with RADON Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load RADON model
model = AutoModelForCausalLM.from_pretrained("MagistrTheOne/RadonSAI")
tokenizer = AutoTokenizer.from_pretrained("MagistrTheOne/RadonSAI")

# Load dataset
dataset = load_dataset("MagistrTheOne/{dataset_name}")

# Example usage
for example in dataset['train']:
    prompt = example['prompt']
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, temperature=0.7)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {{prompt}}")
    print(f"Generated: {{result}}")
    print("---")
```

## Dataset Structure

The dataset contains the following fields:
- `prompt`: Input text prompt
- `category`: Dataset category (multilingual, long_context, code_generation, examples)
- `language`: Language of the prompt (russian, english, mixed)

## Examples

```python
# Get first example
example = dataset['train'][0]
print(example)

# Filter by category
filtered = dataset['train'].filter(lambda x: x['category'] == 'multilingual')
print(f"Multilingual examples: {{len(filtered)}}")
```

## Citation

```bibtex
@misc{{radon2024{dataset_name.replace('-', '')},
  title={{RADON {dataset_name} Dataset}},
  author={{MagistrTheOne}},
  year={{2024}},
  url={{https://huggingface.co/datasets/MagistrTheOne/{dataset_name}}}
}}
```

## License

Apache 2.0 License

## Related

- **Model**: [MagistrTheOne/RadonSAI](https://huggingface.co/MagistrTheOne/RadonSAI)
- **GitHub**: [MagistrTheOne/Radon2BMistral](https://github.com/MagistrTheOne/Radon2BMistral)
"""
    
    return card_content


def update_dataset_card(
    dataset_name: str,
    repo_id: str,
    hf_token: str = None
):
    """Обновить карточку датасета с YAML метаданными"""
    
    print(f"📝 Updating card for {dataset_name}...")
    
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
    
    api = HfApi(token=hf_token)
    
    # Определяем описание и теги для каждого датасета
    dataset_info = {
        "radon-test-multilingual": {
            "description": "Multilingual test dataset for RADON model evaluation with Russian and English prompts",
            "tags": ["multilingual", "russian", "english", "test", "evaluation", "prompts"]
        },
        "radon-test-long_context": {
            "description": "Long context test dataset for RADON model evaluation with extended text samples",
            "tags": ["long-context", "test", "evaluation", "extended-text", "context"]
        },
        "radon-test-code_generation": {
            "description": "Code generation test dataset for RADON model evaluation with programming prompts",
            "tags": ["code-generation", "programming", "python", "test", "evaluation", "code"]
        },
        "radon-examples": {
            "description": "Usage examples and expected responses for RADON model with Russian and English samples",
            "tags": ["examples", "usage", "prompts", "responses", "russian", "english"]
        }
    }
    
    if dataset_name not in dataset_info:
        print(f"⚠️  Unknown dataset: {dataset_name}")
        return False
    
    info = dataset_info[dataset_name]
    
    try:
        # Создаем обновленную карточку
        card_content = create_dataset_card_with_yaml(
            dataset_name=dataset_name,
            description=info["description"],
            tags=info["tags"]
        )
        
        # Загружаем обновленную карточку
        api.upload_file(
            path_or_fileobj=card_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"📝 Update {dataset_name} card with YAML metadata"
        )
        
        print(f"✅ {dataset_name} card updated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update {dataset_name}: {e}")
        return False


def update_all_dataset_cards(
    base_repo_id: str = "MagistrTheOne",
    hf_token: str = None
):
    """Обновить все карточки датасетов"""
    
    print("📝 Updating all RADON dataset cards with YAML metadata...")
    
    datasets = [
        "radon-test-multilingual",
        "radon-test-long_context", 
        "radon-test-code_generation",
        "radon-examples"
    ]
    
    success_count = 0
    
    for dataset_name in datasets:
        repo_id = f"{base_repo_id}/{dataset_name}"
        success = update_dataset_card(
            dataset_name=dataset_name,
            repo_id=repo_id,
            hf_token=hf_token
        )
        if success:
            success_count += 1
    
    print(f"\n📊 Update Summary:")
    print(f"   ✅ Successful: {success_count}/{len(datasets)}")
    print(f"   ❌ Failed: {len(datasets) - success_count}/{len(datasets)}")
    
    return success_count == len(datasets)


def main():
    """Основная функция"""
    
    print("📝 RADON Dataset Cards Update")
    print("=" * 40)
    
    # Проверяем токен
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment variables")
        print("   Set it with: $env:HF_TOKEN='your_token_here'")
        return
    
    # Обновляем все карточки
    success = update_all_dataset_cards(
        base_repo_id="MagistrTheOne",
        hf_token=hf_token
    )
    
    if success:
        print("\n✅ All dataset cards updated successfully!")
        print("🔗 Updated datasets:")
        datasets = [
            "radon-test-multilingual",
            "radon-test-long_context", 
            "radon-test-code_generation",
            "radon-examples"
        ]
        for dataset in datasets:
            print(f"   📊 https://huggingface.co/datasets/MagistrTheOne/{dataset}")
    else:
        print("\n⚠️  Some dataset cards failed to update")


if __name__ == "__main__":
    main()
