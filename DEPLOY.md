# RADON Cloud Deployment Guide

Руководство по деплою RADON в Yandex Cloud (Serverless Containers) и Hugging Face (Hub + Space).

## 📋 Содержание

1. [Предпосылки](#предпосылки)
2. [Yandex Cloud Deploy](#yandex-cloud-deploy)
3. [Hugging Face Deploy](#hugging-face-deploy)
4. [CI/CD Setup](#cicd-setup)
5. [Мониторинг и откат](#мониторинг-и-откат)

---

## Предпосылки

### Необходимые инструменты

```bash
# Docker
docker --version

# jq (для парсинга JSON)
sudo apt-get install jq  # Ubuntu/Debian
brew install jq          # macOS

# Yandex Cloud CLI
curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash
source ~/.bashrc
yc --version

# Hugging Face CLI
pip install -U huggingface_hub git-lfs
git lfs install
```

### Структура проекта

Убедитесь, что в проекте есть:
- `Dockerfile` с портом из переменной `$PORT`
- `api/app.py` с эндпоинтом `GET /health`
- `requirements.txt` с зависимостями
- Конфиги в `configs/`

---

## Yandex Cloud Deploy

### 1. Инициализация YC CLI

```bash
# Интерактивная настройка
yc init

# Выберите:
# - Cloud
# - Folder
# - Compute zone (ru-central1-a)
```

### 2. Создание сервисного аккаунта и ролей

```bash
# Запустить скрипт инициализации
bash cloud/yc/init_yc.sh

# Что делает скрипт:
# - Создает service account "radon-sa"
# - Выдает необходимые роли (editor, puller, invoker, lockbox viewer)
# - Создает Container Registry "radon-registry"
# - Создает Serverless Container "radon-api"
```

### 3. Настройка переменных окружения

```bash
# Скопируйте пример
cp .env.cloud.example .env.cloud

# Отредактируйте значения
nano .env.cloud

# Экспортируйте переменные
export $(cat .env.cloud | xargs)

# Получите Registry ID
yc container registry list
# Установите переменную YC_REGISTRY
export YC_REGISTRY=cr.yandex/ru-central1/<registry_id>
```

### 4. Сборка и push Docker образа

```bash
# Сборка и push в YCR
export YC_REGISTRY=cr.yandex/ru-central1/<your-registry-id>
bash cloud/yc/build_and_push.sh

# Образ будет помечен тегом из git commit hash
```

### 5. Деплой Serverless Container

```bash
# Деплой с дефолтными параметрами
export YC_REGISTRY=cr.yandex/ru-central1/<your-registry-id>
export CONTAINER_NAME=radon-api
export TAG=<tag-from-step-4>

bash cloud/yc/deploy_serverless.sh

# Скрипт выведет публичный URL:
# https://<hash>.containers.yandexcloud.net
```

### 6. Настройка секретов (Lockbox)

```bash
# Экспортируйте секреты
export HF_TOKEN=hf_xxxxxxxxxxxxx
export VK_BOT_TOKEN=vk1.a.xxxxxxxxx

# Создайте Lockbox secret
bash cloud/yc/set_lockbox.sh

# Получите Secret ID для использования в деплое
```

### 7. Проверка деплоя

```bash
# Установите URL из вывода deploy_serverless.sh
API=https://<hash>.containers.yandexcloud.net

# Health check
curl -fsSL $API/health

# Тест генерации
curl -s -X POST "$API/api/v1/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Привет, RADON!",
    "max_length": 64,
    "temperature": 0.7
  }' | jq
```

### 8. VK Webhook (опционально)

В настройках группы VK:
1. Перейдите в **Управление → Работа с API → Callback API**
2. Укажите URL: `https://<hash>.containers.yandexcloud.net/vk/webhook`
3. Установите переменную окружения `VK_CONFIRMATION_CODE` в конфиге
4. Подтвердите сервер

---

## Hugging Face Deploy

### 1. Авторизация

```bash
# Логин в HF CLI
huggingface-cli login

# Введите токен из https://huggingface.co/settings/tokens
```

### 2. Push модели и токенизатора

```bash
# Скопируйте пример конфига
cp .env.hf.example .env.hf
nano .env.hf

# Экспортируйте переменные
export $(cat .env.hf | xargs)

# Запустите push
python cloud/hf/push_to_hub.py
```

### 3. Конвертация в Safetensors (опционально)

```bash
# Конвертировать веса модели
export RADON_CONFIG=configs/model_config_small.json
export OUT=artifacts/model.safetensors

python scripts/convert_safetensors.py

# Загрузить на HF Hub
huggingface-cli upload MagistrTheOne/RADON artifacts/model.safetensors model.safetensors
```

### 4. Создание HF Space

**Вариант A: Через веб-интерфейс**

1. Перейдите на https://huggingface.co/new-space
2. Выберите **Docker** runtime
3. Подключите GitHub репозиторий или загрузите файлы
4. Скопируйте `cloud/hf/space/Dockerfile` в корень Space репозитория
5. В Settings добавьте Secrets: `HF_TOKEN`, `RADON_CONFIG`

**Вариант B: Через Git**

```bash
# Клонируйте Space репозиторий
git clone https://huggingface.co/spaces/<username>/radon
cd radon

# Скопируйте Dockerfile
cp ../RADON/cloud/hf/space/Dockerfile .
cp ../RADON/cloud/hf/space/README.md .

# Скопируйте код проекта
cp -r ../RADON/{api,models,tokenizer,utils,configs,requirements.txt} .

# Commit и push
git add .
git commit -m "Initial Space deployment"
git push
```

---

## CI/CD Setup

### GitHub Actions для автодеплоя

Файл уже создан: `.github/workflows/deploy-yc.yml`

**Настройка секретов в GitHub:**

1. Перейдите в **Settings → Secrets and variables → Actions**
2. Добавьте следующие secrets:

```
YC_OAUTH            # OAuth токен YC (получить: yc iam create-token)
YC_FOLDER_ID        # ID папки YC
YC_CLOUD_ID         # ID облака YC
YC_REGISTRY         # cr.yandex/ru-central1/<registry-id>
YC_CONTAINER_NAME   # radon-api
```

**Получение YC токенов:**

```bash
# OAuth token (действует 1 год)
yc config list

# Folder ID
yc config get folder-id

# Cloud ID
yc config get cloud-id

# Registry URL
yc container registry list --format json | jq -r '.[0].id'
# => cr.yandex/ru-central1/<registry-id>
```

**Триггер деплоя:**

После настройки, каждый push в `main` ветку автоматически:
1. Соберет Docker образ
2. Запушит в YCR
3. Задеплоит новую ревизию Serverless Container
4. Выведет публичный URL в логах

---

## Мониторинг и откат

### Просмотр логов

```bash
# Tail последних 10 минут логов
bash cloud/yc/logs_tail.sh

# Настроить период
SINCE=1h bash cloud/yc/logs_tail.sh
```

### Список ревизий

```bash
# Показать все ревизии контейнера
yc serverless container revision list --container-name radon-api

# Вывод:
# +----------------------+-----+----------------+-----+
# |          ID          | ... |     IMAGE      | ... |
# +----------------------+-----+----------------+-----+
# | d4exxxxxxxxxxxxxxxxx | ... | radon-api:abc1 | ... |
# | d4eyyyyyyyyyyyyyyyyy | ... | radon-api:def2 | ... |
# +----------------------+-----+----------------+-----+
```

### Откат на предыдущую ревизию

```bash
# Получить ID ревизии из списка выше
REV_ID=d4eyyyyyyyyyyyyyyyyy

# Выполнить откат
REV_ID=$REV_ID bash cloud/yc/rollback.sh

# Проверить, что откат прошел успешно
curl -fsSL https://<hash>.containers.yandexcloud.net/health
```

### Метрики и мониторинг

**Базовый мониторинг (встроенный в YC):**

1. Откройте https://console.cloud.yandex.ru/
2. Перейдите в **Serverless Containers → radon-api**
3. Вкладка **Monitoring** покажет:
   - Requests per second
   - Response time
   - Error rate
   - Memory/CPU usage

**Расширенный мониторинг (опционально):**

Для Prometheus/Grafana см. файлы `prometheus.yml` и `docker-compose.yml` в проекте.

---

## Бюджет и оптимизация

### Стоимость YC Serverless Containers

**Текущая конфигурация:**
- 1 vCPU
- 512 MB RAM
- 30s timeout
- Concurrency: 4

**Примерный расчет (при ~3000₽ кредита):**
- Idle: **₽0** (оплата только за выполнение)
- 1000 запросов/день × 2s среднее время = **~₽5-10/день**
- Месяц: **~₽150-300**

**Оптимизация:**
- Используйте малую конфигурацию (512MB) для тестов
- Увеличивайте concurrency (до 16) для снижения холодных стартов
- Держите timeout минимальным (30s для API, 60s для heavy tasks)

### Масштабирование

**Увеличение ресурсов:**

```bash
# В deploy_serverless.sh измените:
CORES=2
MEMORY=1GB
CONCURRENCY=8
TIMEOUT=60s
```

**Для GPU (если нужен inference на больших моделях):**
- Serverless Containers не поддерживают GPU
- Альтернатива: **Compute Instance** с GPU + Container Optimized Image

---

## Troubleshooting

### Проблема: Container не отвечает

```bash
# Проверьте логи
bash cloud/yc/logs_tail.sh

# Проверьте статус ревизии
yc serverless container revision get --revision-id <id>
```

### Проблема: Out of Memory

```bash
# Увеличьте память в deploy_serverless.sh
MEMORY=1GB bash cloud/yc/deploy_serverless.sh
```

### Проблема: Timeout при длинных генерациях

```bash
# Увеличьте timeout
TIMEOUT=60s bash cloud/yc/deploy_serverless.sh
```

### Проблема: Docker build fails

```bash
# Проверьте requirements.txt на совместимость
# Локальная сборка для отладки
docker build -t radon-test .
docker run -p 8000:8000 radon-test
```

---

## Быстрая шпаргалка

```bash
# === Yandex Cloud ===
# Инит
bash cloud/yc/init_yc.sh

# Build & Push
export YC_REGISTRY=cr.yandex/ru-central1/<id>
bash cloud/yc/build_and_push.sh

# Deploy
export CONTAINER_NAME=radon-api TAG=<tag>
bash cloud/yc/deploy_serverless.sh

# Logs
bash cloud/yc/logs_tail.sh

# Rollback
REV_ID=<id> bash cloud/yc/rollback.sh

# === Hugging Face ===
# Login
huggingface-cli login

# Push
export HF_TOKEN=hf_xxx HF_REPO=user/radon
python cloud/hf/push_to_hub.py

# Convert to safetensors
python scripts/convert_safetensors.py
```

---

## Дополнительные ресурсы

- [YC Serverless Containers Docs](https://cloud.yandex.ru/docs/serverless-containers/)
- [YC CLI Reference](https://cloud.yandex.ru/docs/cli/cli-ref/)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [Hugging Face Hub](https://huggingface.co/docs/hub/)

---

**Готово!** RADON теперь доступен в облаке без локального запуска.

