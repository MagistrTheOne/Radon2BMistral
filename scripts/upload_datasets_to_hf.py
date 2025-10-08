"""
Upload RADON datasets to Hugging Face Hub
"""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from datasets import Dataset as HFDataset
import pandas as pd


def prepare_datasets_for_hf(
    data_dir: str = "./data",
    output_dir: str = "./hf_datasets"
):
    """Подготовить датасеты для загрузки на HF"""
    
    print("📊 Preparing RADON datasets for Hugging Face Hub...")
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {}
    
    # 1. Подготавливаем корпус для обучения
    print("[1/4] Preparing training corpus...")
    if os.path.exists(os.path.join(data_dir, "raw_corpus")):
        corpus_files = []
        for file in os.listdir(os.path.join(data_dir, "raw_corpus")):
            if file.endswith(('.txt', '.json')):
                corpus_files.append(os.path.join(data_dir, "raw_corpus", file))
        
        if corpus_files:
            # Создаем датасет корпуса
            corpus_data = []
            for file_path in corpus_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    corpus_data.append({
                        "text": content,
                        "source_file": os.path.basename(file_path),
                        "language": "mixed" if "combined" in file_path else "russian" if "russian" in file_path else "english"
                    })
            
            datasets["radon-corpus"] = {
                "data": corpus_data,
                "description": "RADON training corpus with Russian, English, and code samples",
                "tags": ["russian", "english", "code", "training", "corpus"]
            }
    
    # 2. Подготавливаем тестовые датасеты
    print("[2/4] Preparing test datasets...")
    test_datasets = {
        "multilingual": {
            "prompts": [
                "Машинное обучение - это",
                "Machine learning is",
                "def train_model():",
                "Создай нейронную сеть для",
                "Implement a function that"
            ],
            "expected_topics": ["ML", "programming", "AI", "neural networks"]
        },
        "long_context": {
            "prompts": [
                "В этой статье мы рассмотрим основы машинного обучения и его применение в различных областях. Машинное обучение - это подраздел искусственного интеллекта, который позволяет компьютерам обучаться и принимать решения без явного программирования. Существует три основных типа машинного обучения: обучение с учителем, обучение без учителя и обучение с подкреплением. Каждый из этих типов имеет свои особенности и области применения.",
                "The following is a comprehensive guide to machine learning algorithms and their implementations. We will cover supervised learning, unsupervised learning, and reinforcement learning approaches. Each method has its strengths and weaknesses, and the choice depends on the specific problem domain and available data."
            ],
            "context_length": "long"
        },
        "code_generation": {
            "prompts": [
                "def calculate_loss(y_true, y_pred):",
                "class NeuralNetwork:",
                "import torch.nn as nn",
                "def train_epoch(model, dataloader, optimizer):",
                "def evaluate_model(model, test_data):"
            ],
            "language": "python"
        }
    }
    
    for name, data in test_datasets.items():
        datasets[f"radon-test-{name}"] = {
            "data": [{"prompt": prompt, "category": name} for prompt in data["prompts"]],
            "description": f"RADON test dataset for {name} evaluation",
            "tags": ["test", "evaluation", name]
        }
    
    # 3. Создаем датасет с примерами использования
    print("[3/4] Preparing usage examples...")
    usage_examples = [
        {
            "prompt": "Объясни, что такое машинное обучение",
            "expected_response": "Машинное обучение - это подраздел искусственного интеллекта...",
            "category": "explanation",
            "language": "russian"
        },
        {
            "prompt": "Write a Python function to calculate accuracy",
            "expected_response": "def calculate_accuracy(y_true, y_pred):\n    return (y_true == y_pred).mean()",
            "category": "code_generation",
            "language": "english"
        },
        {
            "prompt": "Создай нейронную сеть для классификации изображений",
            "expected_response": "import torch.nn as nn\n\nclass ImageClassifier(nn.Module):\n    def __init__(self, num_classes):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 32, 3)\n        # ...",
            "category": "code_generation",
            "language": "russian"
        }
    ]
    
    datasets["radon-examples"] = {
        "data": usage_examples,
        "description": "RADON usage examples and expected responses",
        "tags": ["examples", "usage", "prompts", "responses"]
    }
    
    # 4. Создаем метадатасет с информацией о всех датасетах
    print("[4/4] Creating metadata...")
    metadata = {
        "radon_datasets": {
            "description": "RADON Mistral-based transformer datasets collection",
            "version": "1.0.0",
            "created_by": "MagistrTheOne",
            "model": "MagistrTheOne/RadonSAI",
            "datasets": list(datasets.keys()),
            "total_examples": sum(len(ds["data"]) for ds in datasets.values()),
            "languages": ["russian", "english", "mixed"],
            "categories": ["training", "testing", "examples", "corpus"]
        }
    }
    
    # Сохраняем все датасеты
    for name, dataset_info in datasets.items():
        dataset_path = os.path.join(output_dir, name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # Сохраняем данные
        with open(os.path.join(dataset_path, "data.json"), 'w', encoding='utf-8') as f:
            json.dump(dataset_info["data"], f, ensure_ascii=False, indent=2)
        
        # Создаем README для датасета
        readme_content = f"""# {name}

## Description
{dataset_info['description']}

## Tags
{', '.join(dataset_info['tags'])}

## Usage
```python
from datasets import load_dataset

dataset = load_dataset("MagistrTheOne/{name}")
```

## Examples
```python
# Load and use the dataset
data = dataset['train']
for example in data:
    print(example)
```
"""
        
        with open(os.path.join(dataset_path, "README.md"), 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    # Сохраняем метаданные
    with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Datasets prepared in {output_dir}")
    return output_dir, datasets


def upload_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    repo_id: str,
    hf_token: str = None
):
    """Загрузить отдельный датасет на HF"""
    
    print(f"📤 Uploading {dataset_name} to Hugging Face...")
    
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
    
    api = HfApi(token=hf_token)
    
    try:
        # Создаем репозиторий для датасета
        full_repo_id = f"{repo_id}/{dataset_name}"
        create_repo(
            repo_id=full_repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
            token=hf_token
        )
        
        # Загружаем файлы
        api.upload_folder(
            folder_path=dataset_path,
            repo_id=full_repo_id,
            repo_type="dataset",
            commit_message=f"📊 Upload {dataset_name} dataset"
        )
        
        print(f"✅ {dataset_name} uploaded: https://huggingface.co/datasets/{full_repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload {dataset_name}: {e}")
        return False


def upload_all_datasets(
    datasets_dir: str,
    base_repo_id: str = "MagistrTheOne",
    hf_token: str = None
):
    """Загрузить все датасеты на HF"""
    
    print("🚀 Uploading all RADON datasets to Hugging Face Hub...")
    
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
    
    success_count = 0
    total_count = 0
    
    # Загружаем каждый датасет
    for item in os.listdir(datasets_dir):
        item_path = os.path.join(datasets_dir, item)
        if os.path.isdir(item_path) and item != "metadata.json":
            total_count += 1
            success = upload_dataset_to_hf(
                dataset_path=item_path,
                dataset_name=item,
                repo_id=base_repo_id,
                hf_token=hf_token
            )
            if success:
                success_count += 1
    
    print(f"\n📊 Upload Summary:")
    print(f"   ✅ Successful: {success_count}/{total_count}")
    print(f"   ❌ Failed: {total_count - success_count}/{total_count}")
    
    return success_count == total_count


def main():
    """Основная функция"""
    
    print("📊 RADON Datasets Upload to Hugging Face")
    print("=" * 50)
    
    # Проверяем токен
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment variables")
        print("   Set it with: $env:HF_TOKEN='your_token_here'")
        return
    
    # Подготавливаем датасеты
    output_dir, datasets = prepare_datasets_for_hf()
    
    # Загружаем все датасеты
    success = upload_all_datasets(
        datasets_dir=output_dir,
        base_repo_id="MagistrTheOne",
        hf_token=hf_token
    )
    
    if success:
        print("\n✅ All datasets successfully uploaded!")
        print("🔗 Available datasets:")
        for name in datasets.keys():
            print(f"   📊 https://huggingface.co/datasets/MagistrTheOne/{name}")
    else:
        print("\n⚠️  Some datasets failed to upload")


if __name__ == "__main__":
    main()
