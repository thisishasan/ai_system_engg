# Twitter Entity Sentiment Analysis System (Entity-Aware BERT)

---

## 1) Project Overview

This repository contains a complete, end-to-end **AI system** for **entity-level sentiment analysis** on Twitter-style text.  
Given a **tweet** and a **target entity**, the system predicts sentiment **towards the entity** (not just the whole tweet).

This project is engineered according to **AI Systems Engineering (AISE)** principles and includes the main stages of an MLOps lifecycle:
- Data management and schema validation
- Preprocessing and entity-aware input construction
- Model training (CPU-only supported)
- Evaluation and artifact saving
- Containerized REST API deployment (Flask + Docker)
- Basic monitoring hooks (metrics/logs/drift indicators)
- Reproducibility and versioning guidance

**No GPU is required**. Training and inference run on CPU.

---

## 2) Problem Statement

We solve a supervised multi-class classification problem:

```
(text, entity) → sentiment_label
```

### Sentiment Labels
- **Positive**
- **Negative**
- **Neutral**
- **Irrelevant**

---

## 3) Dataset

### 3.1 Source
Kaggle dataset: **Twitter Entity Sentiment Analysis**  
https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

### 3.2 Required Local Path
Place the training file here (not committed to Git):
```
data/train.csv
```

### 3.3 Assumed CSV Schema (Kaggle “Case 2” Headerless Format)
This project assumes the Kaggle file is **headerless** and contains **4 columns in this order**:

1. `tweet_id`
2. `entity`
3. `sentiment`
4. `text`

So the CSV looks like (no header row):
```
2401,Borderlands,Positive,im getting on borderlands and i will...
...
```

If your file has a header, you can still adapt it, but the current training script assumes the headerless format.

### 3.4 Data Management (MLOps-friendly)
- Dataset is treated as an **external artifact**
- **Not versioned** in GitHub
- Users must download it manually and place it in `data/`

---

## 4) Repository Structure

```
twitter-sentiment-bert-mlops/
├── app/                     # Flask inference service
├── train/                   # BERT fine-tuning pipeline
├── scripts/                 # Utility scripts (optional wrappers)
├── monitoring/              # Statistics / drift indicators (lightweight)
├── registry/                # Lightweight metadata / registry pointer
├── postman/                 # Postman collection for API testing
├── data/                    # External dataset (NOT committed)
│   └── train.csv
├── models/                  # Trained model artifacts (NOT committed)
│   └── bert_sentiment/
├── results/                 # Evaluation outputs (NOT committed)
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 5) Requirements

### 5.1 Runtime (to run the API with Docker)
- **Docker** (latest stable)

### 5.2 Training (only needed if you want to fine-tune the model yourself)
- **Python 3.10+**
- `pip`
- Virtual environments (`venv`)

Supported OS:
- Linux
- macOS
- Windows (Docker Desktop + WSL2 recommended)

---

## 6) Docker Installation (All OS)

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```
Then **log out and log back in** (or run `newgrp docker`).

Verify:
```bash
docker ps
```

### macOS / Windows
Install Docker Desktop:
- https://www.docker.com/products/docker-desktop/

Verify:
```bash
docker --version
```

---

## 7) Python Setup for Training (Important for Ubuntu 24.04)

### 7.1 Ubuntu/Debian: enable venv support
If you get `ensurepip is not available`, install:
```bash
sudo apt update
sudo apt install -y python3-venv
```

### 7.2 Create and activate virtual environment

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 7.3 Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 7.4 Fix for HuggingFace Trainer: `accelerate`
If you see:
`ImportError: Trainer requires accelerate>=0.21.0`
then run:
```bash
pip install -U accelerate
```

---

## 8) Model Training (CPU-only)

### 8.1 Run training
From the project root:
```bash
PYTHONPATH=. python train/train_bert.py
```

### 8.2 Expected outputs
After training finishes, you should see:
- Model saved in:
  ```
  models/bert_sentiment/
  ```
- Evaluation metrics saved in:
  ```
  results/eval.json
  ```
- Basic training statistics:
  ```
  monitoring/training_stats.json
  ```

### 8.3 Notes about training messages
- Warning about classifier weights being newly initialized is **normal** (the classification head is trained for your 4 labels).
- If training feels slow: this is expected on CPU; the script uses CPU-friendly defaults (smaller batch size, shorter sequence length, 1 epoch).

---

## 9) Build and Run the REST API with Docker

### 9.1 Build the Docker image
**Important:** Train first so `models/bert_sentiment/` exists, then build:

```bash
docker build -t bert-sentiment-api:1.0 .
```

### 9.2 Run container
```bash
docker run --rm -p 5000:5000 bert-sentiment-api:1.0
```

API base URL:
```
http://localhost:5000
```

---

## 10) API Endpoints

Common endpoints (may vary slightly based on your app implementation):

- `GET /health` — service health check
- `POST /predict` — predict sentiment for one (text, entity)
- `POST /batch_predict` — batch predictions
- `GET /metrics` — basic counters/metrics
- `GET /logs` — request logs (if enabled)
- `GET /monitor/drift` — lightweight drift indicators (if enabled)

### 10.1 Example: health check
```bash
curl http://localhost:5000/health
```

### 10.2 Example: prediction
```bash
curl -X POST http://localhost:5000/predict   -H "Content-Type: application/json"   -d '{"text":"I love the new update, it feels faster","entity":"Apple"}'
```

---

## 11) Postman Testing

A Postman collection is provided under:
```
postman/
```

Steps:
1. Open Postman
2. Import the collection JSON from `postman/`
3. Send requests to `http://localhost:5000`

---

## 12) Model Versioning

A practical versioning approach:
- Each trained model corresponds to a Docker image tag:
  - `bert-sentiment-api:1.0`
  - `bert-sentiment-api:2.0`
- To roll back, run an older tag.

Example:
```bash
docker build -t bert-sentiment-api:2.0 .
docker run --rm -p 5000:5000 bert-sentiment-api:2.0
```

---

## 13) MLOps Lifecycle Implemented

This project demonstrates a simplified MLOps lifecycle:

1. **Plan** (define task, constraints, success criteria)
2. **Data Management** (external dataset artifact, schema validation)
3. **Train** (fine-tune BERT on entity-aware inputs)
4. **Evaluate** (accuracy and macro F1)
5. **Register** (store artifacts + metadata in repo folders)
6. **Package** (Docker build)
7. **Deploy** (Docker run / REST API)
8. **Monitor** (basic metrics/logging/drift indicators)

---

## 14) Git and GitHub: Push the Code

### 14.1 Initialize repository
```bash
git init
git add .
git commit -m "Initial commit"
```

### 14.2 Create GitHub repo and push
```bash
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### 14.3 Important: what is ignored
This repo uses `.gitignore` to exclude:
- `data/` (dataset)
- `models/` (large artifacts)
- `.venv/` (virtual environment)
- logs/results caches

This aligns with good MLOps practice (code versioned, artifacts stored separately).

---

## 15) Troubleshooting

### 15.1 `pip install` fails with `externally-managed-environment` (Ubuntu 24.04)
Use a virtual environment (Section 7). Do not install into system Python.

### 15.2 `python: command not found`
Use `python3`. Optional:
```bash
sudo apt install -y python-is-python3
```

### 15.3 `accelerate` error with Trainer
```bash
pip install -U accelerate
```

### 15.4 Docker permission denied (`docker.sock: permission denied`)
```bash
sudo usermod -aG docker $USER
newgrp docker
```
Then try again:
```bash
docker ps
```

### 15.5 Training fails due to dataset format
Ensure `data/train.csv` is headerless Kaggle format with 4 columns (tweet_id, entity, sentiment, text).

---

