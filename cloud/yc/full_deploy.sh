#!/usr/bin/env bash
set -euo pipefail

# Полный план деплоя RADON на Yandex Cloud
# Использование: bash cloud/yc/full_deploy.sh [2b|7b]

MODEL_SIZE=${1:-2b}
YC_REGISTRY=${YC_REGISTRY:-cr.yandex/ru-central1/<registry-id>}
CONTAINER_NAME=${CONTAINER_NAME:-radon-api}

echo "🚀 RADON Mistral-${MODEL_SIZE} Deployment to Yandex Cloud"
echo "=================================================="

# 1. Проверка зависимостей
echo "[1/6] Checking dependencies..."
command -v yc >/dev/null 2>&1 || { echo "❌ YC CLI not found. Install: curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker not found"; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "❌ jq not found. Install: sudo apt-get install jq"; exit 1; }

# 2. Инициализация YC (если нужно)
echo "[2/6] Initializing Yandex Cloud..."
if ! yc config list >/dev/null 2>&1; then
    echo "⚠️  YC CLI not configured. Run: yc init"
    exit 1
fi

# 3. Подготовка модели (опционально)
echo "[3/6] Preparing model artifacts..."
if [[ -f "artifacts/radon-mistral-${MODEL_SIZE}.tar.gz" ]]; then
    echo "✅ Pre-built model found"
else
    echo "⚠️  No pre-built model. Creating from scratch..."
    bash cloud/yc/prepare_model.sh $MODEL_SIZE
fi

# 4. Сборка и push Docker образа
echo "[4/6] Building and pushing Docker image..."
export YC_REGISTRY=$YC_REGISTRY
export TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
bash cloud/yc/build_and_push.sh

# 5. Деплой serverless контейнера
echo "[5/6] Deploying serverless container..."
export MODEL_SIZE=$MODEL_SIZE
export CONTAINER_NAME=$CONTAINER_NAME
export TAG=$TAG
bash cloud/yc/deploy_serverless.sh

# 6. Проверка деплоя
echo "[6/6] Verifying deployment..."
URL=$(yc serverless container get --name "$CONTAINER_NAME" --format json | jq -r '.status.url')

echo "⏳ Waiting for container to start..."
sleep 30

echo "🔍 Testing health endpoint..."
if curl -f -s "$URL/health" >/dev/null; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    echo "📋 Checking logs..."
    bash cloud/yc/logs_tail.sh
    exit 1
fi

echo ""
echo "🎉 Deployment successful!"
echo "📡 API URL: $URL"
echo "🔧 Health: $URL/health"
echo "📊 Model info: $URL/api/v1/model/info"
echo "🧪 Test generation: curl -X POST '$URL/api/v1/generate' -H 'Content-Type: application/json' -d '{\"prompt\": \"Привет, RADON!\", \"max_length\": 50}'"
echo ""
echo "📋 Useful commands:"
echo "   - View logs: bash cloud/yc/logs_tail.sh"
echo "   - Rollback: REV_ID=<id> bash cloud/yc/rollback.sh"
echo "   - Scale: yc serverless container revision deploy --container-name $CONTAINER_NAME --memory 2GB --cores 2"
