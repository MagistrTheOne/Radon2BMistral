#!/usr/bin/env bash
set -euo pipefail

# Требуется: jq, docker, yc CLI
# Предусловие: yc init (интерактив) уже выполнен, выбран folder.

SA_NAME=${SA_NAME:-radon-sa}
CONTAINER_NAME=${CONTAINER_NAME:-radon-api}

echo "[+] Create service account: $SA_NAME"
yc iam service-account create --name "$SA_NAME" || true
SA_ID=$(yc iam service-account get --name "$SA_NAME" --format json | jq -r .id)

echo "[+] Grant roles"
yc resource-manager folder add-access-binding --role editor --subject serviceAccount:$SA_ID || true
# доступ к образам
yc resource-manager folder add-access-binding --role container-registry.images.puller --subject serviceAccount:$SA_ID || true
# вызов serverless
yc resource-manager folder add-access-binding --role serverless.containers.invoker --subject serviceAccount:$SA_ID || true
# чтение Lockbox (если используешь)
yc resource-manager folder add-access-binding --role lockbox.payloadViewer --subject serviceAccount:$SA_ID || true

echo "[+] Ensure Container Registry exists"
yc container registry create --name radon-registry || true
yc container registry configure-docker

echo "[+] Create serverless container: $CONTAINER_NAME"
yc serverless container create --name "$CONTAINER_NAME" || true

echo "Done."

