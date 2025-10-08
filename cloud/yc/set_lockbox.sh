#!/usr/bin/env bash
set -euo pipefail

# Пример: экспортируем секреты и создаём Lockbox secret
# Требует заранее созданный SA с ролью lockbox.payloadViewer

: "${HF_TOKEN:?export HF_TOKEN first (hf_...)}"
: "${VK_BOT_TOKEN:=}"  # опционально

NAME=${NAME:-radon-secrets}
PAYLOAD=("HF_TOKEN=$HF_TOKEN")
if [[ -n "${VK_BOT_TOKEN}" ]]; then PAYLOAD+=("VK_BOT_TOKEN=$VK_BOT_TOKEN"); fi

ARGS=()
for p in "${PAYLOAD[@]}"; do ARGS+=(--payload "$p"); done

yc lockbox secret create --name "$NAME" "${ARGS[@]}" || true

yc lockbox secret get --name "$NAME" --format json | jq -r '.id'

