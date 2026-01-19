FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pattern A ("baked-in model"): fail the image build early if the model artifacts
# are missing. Train first to create: models/bert_sentiment/
RUN test -d models/bert_sentiment || (echo "ERROR: models/bert_sentiment is missing. Run: bash scripts/train.sh && bash scripts/promote.sh models/bert_sentiment" && exit 1)

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app.main:app", "--timeout", "120"]
