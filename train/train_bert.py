import os
import json
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Force CPU usage (safe on all machines; prevents CUDA checks)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from app.preprocess import build_input, clean_text

# -------------------------------------------------
# Configuration (CPU-friendly)
# -------------------------------------------------
DATA_PATH = "data/train.csv"  # Kaggle headerless CSV (Case 2)
MODEL_NAME = "bert-base-uncased"  # Keep BERT; swap to distilbert for faster CPU if needed

OUTPUT_DIR = "models/bert_sentiment"
RESULTS_DIR = "results"
MONITORING_DIR = "monitoring"

LABELS = ["Negative", "Neutral", "Positive", "Irrelevant"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

SEED = 42
MAX_LEN = 96        # was 128 (reduced for CPU)
TEST_SIZE = 0.2
EPOCHS = 1          # was 2 (reduced for CPU demo)
BATCH_SIZE = 8      # was 16 (reduced for CPU)
LR = 2e-5

# Optional: speed up CPU training by sub-sampling (uncomment if very slow)
# MAX_TRAIN_SAMPLES = 20000


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def load_case2_headerless_csv(path: str) -> pd.DataFrame:
    """
    Assumes Kaggle Case 2 format:
      - No header row
      - Exactly 4 columns:
          tweet_id, entity, sentiment, text
    """
    return pd.read_csv(
        path,
        header=None,
        names=["tweet_id", "entity", "sentiment", "text"],
    )


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    set_seed(SEED)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    # Load Kaggle Case 2 headerless dataset
    df = load_case2_headerless_csv(DATA_PATH)

    # Validate required columns
    required_cols = ["text", "entity", "sentiment"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}' in {DATA_PATH}. "
                f"Available columns: {df.columns.tolist()}"
            )

    df = df.dropna(subset=required_cols).copy()

    # Normalize sentiment labels
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.capitalize()
    df = df[df["sentiment"].isin(LABELS)].copy()
    df["label"] = df["sentiment"].map(label2id)

    # Optional subsampling for very slow CPUs
    # if len(df) > MAX_TRAIN_SAMPLES:
    #     df = df.sample(n=MAX_TRAIN_SAMPLES, random_state=SEED).reset_index(drop=True)

    # Build entity-aware input text
    df["input_text"] = df.apply(
        lambda r: build_input(
            clean_text(str(r["entity"])),
            clean_text(str(r["text"]))
        ),
        axis=1
    )

    # Train/validation split
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=SEED,
    )

    train_ds = Dataset.from_pandas(train_df[["input_text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["input_text", "label"]], preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["input_text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        label2id=label2id,
        id2label=id2label,
    )

    print("Using CUDA:", torch.cuda.is_available())  # should be False on your laptop

    training_args = TrainingArguments(
        output_dir="tmp_trainer",
        eval_strategy="epoch",          # modern replacement for evaluation_strategy
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
        seed=SEED,
        fp16=False,                     # CPU only
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    # Save artifacts
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MONITORING_DIR, exist_ok=True)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(RESULTS_DIR, "eval.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    training_stats = {
        "num_samples": int(len(df)),
        "label_distribution": df["sentiment"].value_counts().to_dict(),
        "avg_text_length": float(df["text"].astype(str).str.len().mean()),
        "cpu_only": True,
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }

    with open(os.path.join(MONITORING_DIR, "training_stats.json"), "w") as f:
        json.dump(training_stats, f, indent=2)

    print("✅ Training complete")
    print("📦 Model saved to:", OUTPUT_DIR)
    print("📊 Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()

