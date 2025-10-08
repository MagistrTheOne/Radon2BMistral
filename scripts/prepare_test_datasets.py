"""
Подготовка тестовых датасетов для оценки RADON
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def create_russian_ml_corpus(output_file: str, num_samples: int = 1000):
    """Создать русскоязычный ML корпус"""
    
    russian_ml_texts = [
        "Машинное обучение - это подраздел искусственного интеллекта, который фокусируется на алгоритмах.",
        "Нейронные сети состоят из слоев нейронов, соединенных весами.",
        "Глубокое обучение использует многослойные нейронные сети для решения сложных задач.",
        "Обучение с учителем требует размеченных данных для обучения модели.",
        "Обучение без учителя находит скрытые паттерны в данных без меток.",
        "Сверточные нейронные сети эффективны для обработки изображений.",
        "Рекуррентные нейронные сети подходят для последовательных данных.",
        "Трансформеры используют механизм внимания для обработки последовательностей.",
        "Градиентный спуск - это алгоритм оптимизации для обучения нейронных сетей.",
        "Регуляризация помогает предотвратить переобучение модели.",
        "Кросс-валидация позволяет оценить качество модели на новых данных.",
        "Метрики качества включают точность, полноту и F1-меру.",
        "Предобработка данных - важный этап в машинном обучении.",
        "Feature engineering - создание новых признаков из исходных данных.",
        "Ensemble методы комбинируют несколько моделей для улучшения качества.",
        "Гиперпараметры модели настраиваются для оптимизации производительности.",
        "Overfitting происходит, когда модель запоминает обучающие данные.",
        "Underfitting означает, что модель слишком простая для данных.",
        "Bias-variance tradeoff - компромисс между смещением и дисперсией.",
        "Данные должны быть репрезентативными для целевой задачи."
    ]
    
    # Генерируем вариации
    generated_texts = []
    for _ in range(num_samples):
        base_text = random.choice(russian_ml_texts)
        
        # Добавляем вариации
        variations = [
            f"В контексте {base_text.lower()}",
            f"Важно отметить, что {base_text.lower()}",
            f"Современные исследования показывают, что {base_text.lower()}",
            f"В области машинного обучения {base_text.lower()}",
            f"Практический опыт показывает, что {base_text.lower()}"
        ]
        
        generated_texts.append(random.choice(variations))
    
    # Сохраняем
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in generated_texts:
            f.write(text + '\n\n')
    
    print(f"✅ Created Russian ML corpus: {len(generated_texts)} samples")
    return len(generated_texts)


def create_english_tech_corpus(output_file: str, num_samples: int = 1000):
    """Создать англоязычный tech корпус"""
    
    english_tech_texts = [
        "Machine learning algorithms can be supervised, unsupervised, or reinforcement learning.",
        "Deep learning models require large amounts of data and computational resources.",
        "Neural networks are inspired by biological neural networks in the brain.",
        "Convolutional neural networks are particularly effective for image recognition tasks.",
        "Recurrent neural networks can process sequential data with memory.",
        "Transformers use self-attention mechanisms to process sequences efficiently.",
        "Natural language processing combines linguistics with machine learning.",
        "Computer vision enables machines to interpret and understand visual information.",
        "Data preprocessing is crucial for successful machine learning projects.",
        "Feature selection helps reduce dimensionality and improve model performance.",
        "Cross-validation provides a robust estimate of model generalization.",
        "Hyperparameter tuning optimizes model performance on validation data.",
        "Regularization techniques prevent overfitting in machine learning models.",
        "Ensemble methods combine multiple models to achieve better performance.",
        "Transfer learning leverages pre-trained models for new tasks.",
        "Model interpretability is important for understanding AI decisions.",
        "Ethical considerations are crucial in AI system development.",
        "Bias in training data can lead to unfair AI system outcomes.",
        "Explainable AI aims to make machine learning models more transparent.",
        "Continuous learning allows models to adapt to new data over time."
    ]
    
    # Генерируем вариации
    generated_texts = []
    for _ in range(num_samples):
        base_text = random.choice(english_tech_texts)
        
        # Добавляем вариации
        variations = [
            f"In modern AI, {base_text.lower()}",
            f"Recent research demonstrates that {base_text.lower()}",
            f"Industry best practices suggest that {base_text.lower()}",
            f"Advanced techniques show that {base_text.lower()}",
            f"Practical applications reveal that {base_text.lower()}"
        ]
        
        generated_texts.append(random.choice(variations))
    
    # Сохраняем
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in generated_texts:
            f.write(text + '\n\n')
    
    print(f"✅ Created English tech corpus: {len(generated_texts)} samples")
    return len(generated_texts)


def create_code_corpus(output_file: str, num_samples: int = 500):
    """Создать корпус кода"""
    
    code_samples = [
        "def train_model(X, y, epochs=100):\n    model = Sequential()\n    model.add(Dense(64, activation='relu'))\n    model.add(Dense(1, activation='sigmoid'))\n    model.compile(optimizer='adam', loss='binary_crossentropy')\n    model.fit(X, y, epochs=epochs)\n    return model",
        
        "import torch\nimport torch.nn as nn\n\nclass Transformer(nn.Module):\n    def __init__(self, d_model, nhead):\n        super().__init__()\n        self.attention = nn.MultiheadAttention(d_model, nhead)\n        self.norm = nn.LayerNorm(d_model)\n    \n    def forward(self, x):\n        attn_output, _ = self.attention(x, x, x)\n        return self.norm(x + attn_output)",
        
        "def preprocess_text(text):\n    text = text.lower()\n    text = re.sub(r'[^a-zA-Z0-9\\s]', '', text)\n    tokens = text.split()\n    return [word for word in tokens if len(word) > 2]",
        
        "class DataLoader:\n    def __init__(self, dataset, batch_size=32):\n        self.dataset = dataset\n        self.batch_size = batch_size\n    \n    def __iter__(self):\n        for i in range(0, len(self.dataset), self.batch_size):\n            yield self.dataset[i:i+self.batch_size]",
        
        "def calculate_accuracy(predictions, targets):\n    correct = (predictions == targets).sum().item()\n    total = targets.size(0)\n    return correct / total",
        
        "def tokenize_text(text, tokenizer):\n    tokens = tokenizer.encode(text)\n    return torch.tensor(tokens).unsqueeze(0)",
        
        "def save_model(model, path):\n    torch.save(model.state_dict(), path)\n    print(f'Model saved to {path}')",
        
        "def load_model(model_class, path, **kwargs):\n    model = model_class(**kwargs)\n    model.load_state_dict(torch.load(path))\n    return model"
    ]
    
    # Генерируем вариации
    generated_texts = []
    for _ in range(num_samples):
        base_code = random.choice(code_samples)
        
        # Добавляем комментарии и вариации
        variations = [
            f"# Machine learning implementation\n{base_code}",
            f"# Deep learning model\n{base_code}",
            f"# Data processing function\n{base_code}",
            f"# Neural network architecture\n{base_code}",
            f"# Training utility\n{base_code}"
        ]
        
        generated_texts.append(random.choice(variations))
    
    # Сохраняем
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in generated_texts:
            f.write(text + '\n\n')
    
    print(f"✅ Created code corpus: {len(generated_texts)} samples")
    return len(generated_texts)


def create_combined_corpus(
    russian_file: str,
    english_file: str, 
    code_file: str,
    output_file: str,
    russian_ratio: float = 0.4,
    english_ratio: float = 0.4,
    code_ratio: float = 0.2
):
    """Создать объединенный корпус"""
    
    print("[+] Creating combined test corpus...")
    
    # Читаем корпусы
    with open(russian_file, 'r', encoding='utf-8') as f:
        russian_texts = [line.strip() for line in f if line.strip()]
    
    with open(english_file, 'r', encoding='utf-8') as f:
        english_texts = [line.strip() for line in f if line.strip()]
    
    with open(code_file, 'r', encoding='utf-8') as f:
        code_texts = [line.strip() for line in f if line.strip()]
    
    # Вычисляем количество
    total_samples = len(russian_texts) + len(english_texts) + len(code_texts)
    russian_count = int(total_samples * russian_ratio)
    english_count = int(total_samples * english_ratio)
    code_count = int(total_samples * code_ratio)
    
    # Создаем объединенный корпус
    combined_texts = []
    combined_texts.extend(russian_texts[:russian_count])
    combined_texts.extend(english_texts[:english_count])
    combined_texts.extend(code_texts[:code_count])
    
    # Перемешиваем
    random.shuffle(combined_texts)
    
    # Сохраняем
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in combined_texts:
            f.write(text + '\n\n')
    
    stats = {
        "total_samples": len(combined_texts),
        "russian_samples": russian_count,
        "english_samples": english_count,
        "code_samples": code_count,
        "total_chars": sum(len(text) for text in combined_texts)
    }
    
    print(f"✅ Combined corpus: {stats['total_samples']} samples, {stats['total_chars']:,} chars")
    return stats


def main():
    """Основная функция"""
    
    print("🧪 Preparing test datasets for RADON evaluation...")
    
    # Создаем директории
    os.makedirs("data/test_datasets", exist_ok=True)
    
    # Создаем корпусы
    russian_count = create_russian_ml_corpus("data/test_datasets/russian_ml.txt", 1000)
    english_count = create_english_tech_corpus("data/test_datasets/english_tech.txt", 1000)
    code_count = create_code_corpus("data/test_datasets/code_samples.txt", 500)
    
    # Создаем объединенный корпус
    stats = create_combined_corpus(
        "data/test_datasets/russian_ml.txt",
        "data/test_datasets/english_tech.txt",
        "data/test_datasets/code_samples.txt",
        "data/test_datasets/combined_test.txt"
    )
    
    # Сохраняем статистику
    with open("data/test_datasets/corpus_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n🎉 Test datasets ready!")
    print(f"📊 Total samples: {stats['total_samples']}")
    print(f"📊 Total characters: {stats['total_chars']:,}")
    print(f"📁 Files created in data/test_datasets/")


if __name__ == "__main__":
    main()
