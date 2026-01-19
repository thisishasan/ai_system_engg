import os
import json
from typing import Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.preprocess import build_input

DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", "models/bert_sentiment")
REGISTRY_PATH = os.getenv("REGISTRY_PATH", "registry/registry.json")


def _load_production_path() -> str:
    if not os.path.exists(REGISTRY_PATH):
        return DEFAULT_MODEL_DIR
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            reg = json.load(f)
        prod = reg.get("production") or {}
        return prod.get("path") or DEFAULT_MODEL_DIR
    except Exception:
        return DEFAULT_MODEL_DIR


class SentimentService:
    def __init__(self):
        model_dir = _load_production_path()
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. Train first to create models/bert_sentiment."
            )

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, entity: str, text: str, max_length: int = 128) -> Dict[str, Any]:
        inp = build_input(entity, text)
        enc = self.tokenizer(
            inp,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        logits = self.model(**enc).logits
        probs = F.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())

        # id2label sometimes uses string keys
        id2label = getattr(self.model.config, "id2label", None) or {}
        if isinstance(id2label, dict):
            label = id2label.get(str(pred_id), id2label.get(pred_id, str(pred_id)))
        else:
            label = id2label[pred_id]

        prob_map = {}
        for i in range(probs.shape[0]):
            if isinstance(id2label, dict):
                k = id2label.get(str(i), id2label.get(i, str(i)))
            else:
                k = id2label[i]
            prob_map[str(k)] = round(float(probs[i].item()), 4)

        return {
            "entity": entity,
            "text": text,
            "label": str(label),
            "confidence": round(float(probs[pred_id].item()), 4),
            "probs": prob_map,
            "model_dir": self.model_dir,
        }
