#!/usr/bin/env bash
set -euo pipefail

# ENV required: YC_REGISTRY=cr.yandex/<region>/<registry_id>
: "${YC_REGISTRY:?YC_REGISTRY is required}"

TAG=${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}

echo "[+] Build $YC_REGISTRY/radon-api:$TAG"
docker build -t "$YC_REGISTRY/radon-api:$TAG" .

echo "[+] Push"
docker push "$YC_REGISTRY/radon-api:$TAG"

echo "TAG=$TAG"

