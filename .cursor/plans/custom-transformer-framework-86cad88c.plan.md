<!-- 86cad88c-5dc3-4879-884a-c9c6b5bf5fd0 34990563-02bd-474d-97fa-a459f93d97be -->
# План разработки RADON Custom Transformer Framework

## 1. Структура проекта и виртуальное окружение

Создам следующую структуру:

```
RADON/
├── venv/                          # Виртуальное окружение
├── models/
│   ├── __init__.py
│   ├── config.py                  # Конфигурация модели
│   ├── transformer_gpt2.py        # GPT-2 архитектура
│   ├── transformer_t5.py          # T5 архитектура
│   └── hybrid_model.py            # Гибридная модель с выбором режима
├── tokenizer/
│   ├── __init__.py
│   ├── train_tokenizer.py         # Обучение SentencePiece токенизатора
│   └── custom_tokenizer.py        # Обертка для интеграции с HF
├── api/
│   ├── __init__.py
│   ├── app.py                     # FastAPI сервер
│   ├── routes.py                  # Эндпоинты для генерации
│   └── vk_webhook.py              # VK webhook интеграция
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py           # Логирование
│   └── model_utils.py             # Сохранение/конвертация моделей
├── configs/
│   ├── model_config_small.json    # Малая модель
│   ├── model_config_medium.json   # Средняя модель
│   └── training_config.json       # Параметры обучения
├── data/
│   └── sample_corpus.txt          # Примерный корпус для токенизатора
├── scripts/
│   ├── train_model.py             # Скрипт обучения модели
│   └── deploy_to_hf.py            # Скрипт деплоя на HF Hub
├── requirements.txt               # Зависимости с точными версиями
├── Dockerfile                     # Docker контейнер
├── .dockerignore
├── .gitignore
└── README.md
```

## 2. Базовая конфигурация

**configs/model_config_small.json** - конфигурируемые параметры:

- vocab_size, hidden_size, num_layers, num_attention_heads
- model_type: "gpt2" | "t5" | "hybrid"
- dropout, max_position_embeddings

## 3. Архитектура моделей

**models/transformer_gpt2.py:**

- Кастомная GPT-2 архитектура (decoder-only)
- Наследуется от `PreTrainedModel`
- Поддержка динамических параметров из config

**models/transformer_t5.py:**

- Кастомная T5 архитектура (encoder-decoder)
- Наследуется от `PreTrainedModel`

**models/hybrid_model.py:**

- Обертка для переключения между GPT-2 и T5
- Единая точка входа с выбором архитектуры

## 4. Кастомный токенизатор

**tokenizer/train_tokenizer.py:**

- Обучение SentencePiece BPE токенизатора на русском/английском
- Заглушка для обучения на пользовательских данных
- Сохранение в формат совместимый с HF

**tokenizer/custom_tokenizer.py:**

- Обертка в стиле `PreTrainedTokenizer`
- Методы encode/decode, save_pretrained, from_pretrained

## 5. FastAPI сервер

**api/app.py:**

- Инициализация FastAPI
- Загрузка модели и токенизатора при старте
- Middleware для логирования

**api/routes.py:**

- POST `/generate` - генерация текста
- POST `/tokenize` - тестирование токенизатора
- GET `/health` - проверка статуса
- GET `/model_info` - информация о загруженной модели

**api/vk_webhook.py:**

- POST `/vk/webhook` - прием сообщений от VK
- Обработка callback событий
- Отправка ответов через VK API

## 6. Утилиты

**utils/model_utils.py:**

- Функция сохранения в PyTorch (.pt)
- Функция конвертации в Safetensors
- Загрузка модели из разных форматов

**utils/logging_utils.py:**

- Настройка логгера
- Функции для логирования запросов, ошибок, метрик

## 7. Скрипты деплоя

**scripts/deploy_to_hf.py:**

- Авторизация через HF token
- push_to_hub для модели и токенизатора
- Создание model card

## 8. Docker контейнеризация

**Dockerfile:**

```dockerfile
FROM python:3.11.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 9. Зависимости

**requirements.txt** с точными версиями:

- torch==2.1.0
- transformers==4.35.0
- sentencepiece==0.1.99
- safetensors==0.4.0
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- pydantic==2.5.0
- requests==2.31.0
- huggingface-hub==0.19.4

## 10. Документация

**README.md:**

- Быстрый старт
- Установка зависимостей в venv
- Обучение токенизатора
- Обучение модели
- Запуск API сервера
- Docker команды
- Деплой на Hugging Face

### To-dos

- [ ] Создать структуру проекта и виртуальное окружение
- [ ] Создать конфигурационные файлы для моделей (small, medium) и параметров обучения
- [ ] Реализовать базовые архитектуры GPT-2 и T5 с поддержкой PreTrainedModel
- [ ] Создать гибридную модель с возможностью выбора архитектуры
- [ ] Реализовать обучение SentencePiece токенизатора и обертку для HF
- [ ] Создать утилиты для логирования и работы с моделями (сохранение/конвертация)
- [ ] Создать FastAPI сервер с эндпоинтами для генерации и VK webhook
- [ ] Создать скрипты для обучения модели и деплоя на Hugging Face Hub
- [ ] Создать Dockerfile и docker-compose для контейнеризации
- [ ] Создать requirements.txt с точными версиями всех зависимостей
- [ ] Создать README.md с инструкциями по установке, запуску и деплою