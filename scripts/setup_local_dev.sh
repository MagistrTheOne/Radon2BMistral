#!/usr/bin/env bash
set -euo pipefail

# Быстрая настройка для локальной разработки
echo "🚀 Setting up RADON for local development..."

# 1. Создать виртуальное окружение
echo "[1/5] Creating virtual environment..."
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
# venv\Scripts\activate  # Windows

# 2. Установить зависимости
echo "[2/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Создать директории
echo "[3/5] Creating directories..."
mkdir -p data/{raw_corpus,processed,test_datasets}
mkdir -p logs outputs checkpoints models tokenizer

# 4. Подготовить тестовые датасеты
echo "[4/5] Preparing test datasets..."
python scripts/prepare_test_datasets.py

# 5. Инициализировать модель
echo "[5/5] Initializing model..."
python scripts/initialize_mistral.py \
    --config_path configs/model_config_mistral_2b.json \
    --output_dir models/checkpoint \
    --model_size 2b

echo "✅ Local setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate venv: source venv/bin/activate"
echo "2. Run tests: python scripts/run_test_suite.py"
echo "3. Start API: python -m uvicorn api.app:app --reload"
echo "4. Test: curl http://localhost:8000/health"
