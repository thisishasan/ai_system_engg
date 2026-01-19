#!/usr/bin/env bash
set -euo pipefail

TAG=${1:-bert-sentiment-api:1.0}

if [ ! -d "models/bert_sentiment" ]; then
  echo "ERROR: models/bert_sentiment not found."
  echo "Train first: bash scripts/train.sh"
  echo "Then promote: bash scripts/promote.sh models/bert_sentiment"
  exit 1
fi

docker build -t "$TAG" .
echo "Built image: $TAG"
