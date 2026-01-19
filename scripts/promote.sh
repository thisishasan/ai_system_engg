#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-models/bert_sentiment}
VERSION=${2:-v1}

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi

python - <<PY
import json
import os

reg_path = "registry/registry.json"
model_path = os.environ.get("MODEL_PATH", "$MODEL_PATH")
version = os.environ.get("VERSION", "$VERSION")

reg = {"production": {}}
if os.path.exists(reg_path):
    with open(reg_path, "r", encoding="utf-8") as f:
        reg = json.load(f)

metrics = {}
if os.path.exists("results/eval.json"):
    with open("results/eval.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

reg["production"] = {
    "version": version,
    "path": model_path,
    "metrics": {
        "accuracy": metrics.get("eval_accuracy"),
        "f1_macro": metrics.get("eval_f1_macro"),
    },
}

with open(reg_path, "w", encoding="utf-8") as f:
    json.dump(reg, f, indent=2)

print(f"Promoted to production: {model_path} ({version})")
PY
