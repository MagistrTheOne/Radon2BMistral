#!/usr/bin/env bash
set -euo pipefail
CONTAINER_NAME=${CONTAINER_NAME:-radon-api}
SINCE=${SINCE:-10m}
yc serverless container logs tail --name "$CONTAINER_NAME" --since "$SINCE"

