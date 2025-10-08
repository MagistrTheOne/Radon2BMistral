#!/usr/bin/env bash
set -euo pipefail

: "${YC_REGISTRY:?}"
CONTAINER_NAME=${CONTAINER_NAME:-radon-api}
TAG=${TAG:-latest}

# Model configuration
MODEL_SIZE=${MODEL_SIZE:-2b}  # 2b or 7b
RADON_ARCH=${RADON_ARCH:-mistral}
RADON_CONFIG=${RADON_CONFIG:-configs/model_config_mistral_${MODEL_SIZE}.json}

# Resource allocation based on model size
if [[ "$MODEL_SIZE" == "7b" ]]; then
    CORES=${CORES:-2}
    MEMORY=${MEMORY:-2GB}
    CONCURRENCY=${CONCURRENCY:-2}
    TIMEOUT=${TIMEOUT:-60s}
else
    CORES=${CORES:-1}
    MEMORY=${MEMORY:-1GB}
    CONCURRENCY=${CONCURRENCY:-4}
    TIMEOUT=${TIMEOUT:-30s}
fi

LOG_LEVEL=${LOG_LEVEL:-INFO}
USE_FLASH_ATTENTION=${USE_FLASH_ATTENTION:-true}
DEVICE=${DEVICE:-cuda}

echo "[+] Deploying RADON Mistral-${MODEL_SIZE} to Yandex Cloud..."
echo "    - Container: $CONTAINER_NAME"
echo "    - Image: $YC_REGISTRY/radon-api:$TAG"
echo "    - Resources: $CORES cores, $MEMORY memory"
echo "    - Concurrency: $CONCURRENCY"
echo "    - Timeout: $TIMEOUT"

yc serverless container create --name "$CONTAINER_NAME" || true

yc serverless container revision deploy \
  --container-name "$CONTAINER_NAME" \
  --image "$YC_REGISTRY/radon-api:$TAG" \
  --cores $CORES --memory $MEMORY --concurrency $CONCURRENCY \
  --execution-timeout $TIMEOUT \
  --env RADON_ARCH=$RADON_ARCH \
  --env RADON_CONFIG=$RADON_CONFIG \
  --env MODEL_SIZE=$MODEL_SIZE \
  --env USE_FLASH_ATTENTION=$USE_FLASH_ATTENTION \
  --env DEVICE=$DEVICE \
  --env LOG_LEVEL=$LOG_LEVEL

yc serverless container allow-unauthenticated-invoke --name "$CONTAINER_NAME" || true

URL=$(yc serverless container get --name "$CONTAINER_NAME" --format json | jq -r '.status.url')
echo "[+] Deployment successful!"
echo "[+] URL: $URL"
echo "[+] Health check: curl -f $URL/health"

