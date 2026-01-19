#!/usr/bin/env bash
set -euo pipefail

TAG=${1:-bert-sentiment-api:1.0}
PORT=${2:-5000}

# Persist SQLite DB locally
mkdir -p ./runtime

docker run --rm -p ${PORT}:5000 \
  -e DB_PATH=/app/runtime/predictions.db \
  -v $(pwd)/runtime:/app/runtime \
  "$TAG"
