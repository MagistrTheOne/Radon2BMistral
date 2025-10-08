#!/usr/bin/env bash
set -euo pipefail

# Быстрый старт RADON на локальной машине
echo "🚀 RADON Local Quick Start"
echo "=========================="

# 1. Проверка окружения
echo "[1/6] Checking environment..."
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install pip"
    exit 1
fi

# 2. Создание виртуального окружения
echo "[2/6] Creating virtual environment..."
if [[ ! -d "venv" ]]; then
    python -m venv venv
fi

# Активация venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# 3. Установка зависимостей
echo "[3/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Подготовка тестовых данных
echo "[4/6] Preparing test datasets..."
python scripts/prepare_test_datasets.py

# 5. Инициализация модели
echo "[5/6] Initializing model..."
python scripts/initialize_mistral.py \
    --config_path configs/model_config_mistral_2b.json \
    --output_dir models/checkpoint \
    --model_size 2b

# 6. Запуск тестов
echo "[6/6] Running test suite..."
python scripts/run_test_suite.py

echo ""
echo "🎉 Quick start completed!"
echo ""
echo "📋 Next steps:"
echo "1. View results: cat test_results.json"
echo "2. Start API: python -m uvicorn api.app:app --reload"
echo "3. Test API: curl http://localhost:8000/health"
echo "4. Generate text: curl -X POST http://localhost:8000/api/v1/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"Привет, RADON!\", \"max_length\": 50}'"
echo ""
echo "🔧 Useful commands:"
echo "  - Monitor GPU: watch -n 1 nvidia-smi"
echo "  - View logs: tail -f logs/api.log"
echo "  - Stop API: Ctrl+C"
