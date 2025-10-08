# RADON Custom Transformer Framework

Модульный каркас для создания кастомных трансформеров с поддержкой GPT-2 и T5 архитектур, кастомным токенизатором на базе SentencePiece, и интеграцией с Hugging Face и VK для простого деплоя через Docker.

## 🚀 Особенности

- **Гибридная архитектура**: Поддержка GPT-2 и T5 с возможностью переключения
- **Кастомный токенизатор**: SentencePiece BPE для русского и английского языков
- **Hugging Face интеграция**: Полная совместимость с HF Hub
- **VK Bot интеграция**: Webhook для автоматических ответов
- **Docker контейнеризация**: Готовый к деплою контейнер
- **Модульная архитектура**: Легко расширяемый и настраиваемый
- **Логирование и мониторинг**: Полное отслеживание запросов и метрик

## 📁 Структура проекта

```
RADON/
├── models/                    # Архитектуры моделей
│   ├── config.py             # Конфигурация моделей
│   ├── transformer_gpt2.py   # GPT-2 архитектура
│   ├── transformer_t5.py     # T5 архитектура
│   └── hybrid_model.py       # Гибридная модель
├── tokenizer/                 # Токенизатор
│   ├── train_tokenizer.py    # Обучение SentencePiece
│   └── custom_tokenizer.py   # HF-совместимый токенизатор
├── api/                       # FastAPI сервер
│   ├── app.py                # Основное приложение
│   ├── routes.py             # API эндпоинты
│   └── vk_webhook.py         # VK интеграция
├── utils/                     # Утилиты
│   ├── logging_utils.py       # Логирование
│   └── model_utils.py         # Работа с моделями
├── configs/                   # Конфигурации
│   ├── model_config_small.json
│   ├── model_config_medium.json
│   └── training_config.json
├── scripts/                   # Скрипты
│   ├── train_model.py        # Обучение модели
│   └── deploy_to_hf.py       # Деплой на HF Hub
├── data/                     # Данные
│   └── sample_corpus.txt     # Примерный корпус
├── requirements.txt           # Зависимости
├── Dockerfile               # Docker контейнер
├── docker-compose.yml       # Docker Compose
└── README.md                # Документация
```

## 🛠 Установка и настройка

### 1. Клонирование и виртуальное окружение

```bash
# Клонирование репозитория
git clone <repository-url>
cd RADON

# Создание виртуального окружения
python -m venv venv

# Активация (Windows)
venv\Scripts\activate

# Активация (Linux/macOS)
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Настройка окружения

```bash
# Создание необходимых директорий
mkdir -p logs outputs checkpoints models/checkpoint tokenizer/checkpoint

# Настройка переменных окружения (опционально)
export MODEL_CONFIG_PATH=configs/model_config_small.json
export MODEL_PATH=./models/checkpoint
export TOKENIZER_PATH=./tokenizer/checkpoint
export DEVICE=cpu
export VK_ACCESS_TOKEN=your_vk_token
export VK_CONFIRMATION_CODE=your_confirmation_code
```

## 🎯 Быстрый старт

### 1. Обучение токенизатора

```bash
# Обучение на корпусе
python tokenizer/train_tokenizer.py \
    --input data/sample_corpus.txt \
    --output ./tokenizer_output \
    --vocab_size 32000 \
    --prefix custom_tokenizer
```

### 2. Обучение модели

```bash
# Обучение GPT-2 модели
python scripts/train_model.py \
    --model_config configs/model_config_small.json \
    --training_config configs/training_config.json \
    --data_path data/sample_corpus.txt \
    --output_dir ./outputs \
    --tokenizer_path ./tokenizer_output
```

### 3. Запуск API сервера

```bash
# Запуск сервера
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Тестирование API

```bash
# Проверка здоровья
curl http://localhost:8000/health

# Генерация текста
curl -X POST "http://localhost:8000/api/v1/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Привет! Как дела?",
       "max_length": 100,
       "temperature": 0.7
     }'
```

## 🐳 Docker деплой

### 1. Сборка и запуск

```bash
# Сборка образа
docker build -t radon-transformer .

# Запуск контейнера
docker run -d -p 8000:8000 \
  -e VK_ACCESS_TOKEN=your_token \
  -e VK_CONFIRMATION_CODE=your_code \
  radon-transformer
```

### 2. Docker Compose

```bash
# Запуск с Docker Compose
docker-compose up -d

# Запуск с мониторингом
docker-compose --profile monitoring up -d

# Запуск в продакшене
docker-compose --profile production up -d
```

## 🔧 Конфигурация

### Модель (configs/model_config_small.json)

```json
{
  "model_type": "gpt2",
  "vocab_size": 32000,
  "hidden_size": 256,
  "num_layers": 6,
  "num_attention_heads": 8,
  "max_position_embeddings": 512,
  "dropout": 0.1
}
```

### Обучение (configs/training_config.json)

```json
{
  "learning_rate": 5e-4,
  "batch_size": 8,
  "num_epochs": 3,
  "warmup_steps": 100,
  "weight_decay": 0.01,
  "save_steps": 500
}
```

## 📡 API Эндпоинты

### Генерация текста

```bash
POST /api/v1/generate
{
  "prompt": "Привет! Как дела?",
  "max_length": 100,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9
}
```

### Токенизация

```bash
POST /api/v1/tokenize
{
  "text": "Привет, мир!",
  "add_special_tokens": true,
  "return_tokens": true
}
```

### Переключение модели

```bash
POST /api/v1/model/switch
{
  "model_type": "gpt2"  # или "t5"
}
```

### VK Webhook

```bash
POST /vk/webhook
{
  "type": "message_new",
  "object": {
    "message": {
      "text": "Привет!",
      "from_id": 12345
    }
  }
}
```

## 🚀 Деплой на Hugging Face Hub

### 1. Подготовка

```bash
# Получение HF токена
# https://huggingface.co/settings/tokens

# Установка переменной окружения
export HF_TOKEN=your_hf_token
```

### 2. Деплой

```bash
# Деплой модели
python scripts/deploy_to_hf.py \
    --model_path ./outputs/final_model \
    --tokenizer_path ./tokenizer_output \
    --repo_name your-username/radon-model \
    --hf_token $HF_TOKEN \
    --private \
    --verify
```

## 🔍 Мониторинг и логирование

### Логи

```bash
# Просмотр логов API
tail -f logs/api.log

# Просмотр логов обучения
tail -f logs/training.log

# Просмотр логов деплоя
tail -f logs/deployment.log
```

### Метрики

```bash
# Prometheus (если включен)
curl http://localhost:9090

# Grafana (если включен)
# http://localhost:3000 (admin/admin)
```

## 🧪 Тестирование

### Unit тесты

```bash
# Запуск тестов
pytest tests/

# С покрытием
pytest --cov=. tests/
```

### API тесты

```bash
# Тестирование эндпоинтов
curl -X GET http://localhost:8000/health
curl -X GET http://localhost:8000/model/info
curl -X POST http://localhost:8000/api/v1/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Test", "max_length": 50}'
```

## 📊 Производительность

### Бенчмарки

```bash
# Запуск бенчмарка
curl -X GET http://localhost:8000/api/v1/model/benchmark
```

### Оптимизация

- Используйте GPU для обучения и инференса
- Настройте batch_size в зависимости от памяти
- Включите mixed precision для ускорения
- Используйте кэширование для часто запрашиваемых данных

## 🔧 Расширение функциональности

### Добавление новой архитектуры

1. Создайте новый файл в `models/`
2. Наследуйтесь от `PreTrainedModel`
3. Добавьте в `hybrid_model.py`
4. Обновите конфигурацию

### Добавление нового токенизатора

1. Создайте класс в `tokenizer/`
2. Наследуйтесь от `PreTrainedTokenizer`
3. Реализуйте необходимые методы
4. Обновите импорты

### Добавление новых API эндпоинтов

1. Добавьте роуты в `api/routes.py`
2. Создайте Pydantic модели для запросов/ответов
3. Добавьте логирование и обработку ошибок
4. Обновите документацию

## 🐛 Устранение неполадок

### Частые проблемы

1. **Модель не загружается**
   - Проверьте пути к файлам
   - Убедитесь в наличии checkpoint'ов
   - Проверьте совместимость версий

2. **Ошибки токенизатора**
   - Проверьте наличие model файла
   - Убедитесь в правильности кодировки
   - Проверьте специальные токены

3. **Проблемы с Docker**
   - Проверьте доступность портов
   - Убедитесь в правильности переменных окружения
   - Проверьте логи контейнера

### Логи и отладка

```bash
# Логи Docker контейнера
docker logs radon-transformer-api

# Логи с детализацией
docker logs -f radon-transformer-api

# Вход в контейнер
docker exec -it radon-transformer-api bash
```

## 📝 Лицензия

Этот проект предоставляется для исследовательских и образовательных целей.

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Создайте Pull Request

## 📞 Поддержка

Для вопросов и поддержки:
- Создайте Issue в репозитории
- Проверьте документацию
- Изучите логи для диагностики

---

**RADON Custom Transformer Framework** - мощный инструмент для создания и деплоя кастомных трансформеров с полной интеграцией в экосистему ML.
