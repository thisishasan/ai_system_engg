#!/usr/bin/env bash
set -euo pipefail

python train/train_bert.py

echo "\nTraining completed. Model artifacts are in models/bert_sentiment"
