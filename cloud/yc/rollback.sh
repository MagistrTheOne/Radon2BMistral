#!/usr/bin/env bash
set -euo pipefail
CONTAINER_NAME=${CONTAINER_NAME:-radon-api}
REV_ID=${REV_ID:-}
if [[ -z "$REV_ID" ]]; then
  echo "Usage: REV_ID=<id> $0"; exit 1
fi
yc serverless container revision activate --container-name "$CONTAINER_NAME" --revision-id "$REV_ID"

