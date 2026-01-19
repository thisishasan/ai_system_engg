import os
from flask import Flask, request, jsonify

from app.service import SentimentService
from app.db import init_db, log_prediction, get_logs, get_metrics, fetch_recent_inputs
from monitoring.drift import load_training_stats, drift_report

DB_PATH = os.getenv("DB_PATH", "predictions.db")
TRAINING_STATS_PATH = os.getenv("TRAINING_STATS_PATH", "monitoring/training_stats.json")


def create_app() -> Flask:
    app = Flask(__name__)

    # Init DB
    init_db(DB_PATH)

    # Load model once at startup
    service = SentimentService()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_dir": service.model_dir}), 200

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        entity = payload.get("entity", "")
        if not text or not entity:
            return jsonify({"error": "Both 'text' and 'entity' are required."}), 400

        res = service.predict(entity=entity, text=text)
        log_prediction(entity, text, res["label"], res["confidence"], DB_PATH)
        return jsonify(res), 200

    @app.post("/batch_predict")
    def batch_predict():
        payload = request.get_json(silent=True) or {}
        items = payload.get("items", [])
        if not isinstance(items, list) or len(items) == 0:
            return jsonify({"error": "'items' must be a non-empty list."}), 400

        results = []
        for it in items:
            text = (it or {}).get("text", "")
            entity = (it or {}).get("entity", "")
            if not text or not entity:
                results.append({"error": "Missing text/entity", "input": it})
                continue
            res = service.predict(entity=entity, text=text)
            log_prediction(entity, text, res["label"], res["confidence"], DB_PATH)
            results.append(res)

        return jsonify({"results": results}), 200

    @app.get("/logs")
    def logs():
        limit = request.args.get("limit", default=50, type=int)
        return jsonify({"logs": get_logs(limit, DB_PATH)}), 200

    @app.get("/metrics")
    def metrics():
        return jsonify(get_metrics(DB_PATH)), 200

    @app.get("/monitor/drift")
    def monitor_drift():
        limit = request.args.get("limit", default=500, type=int)
        if not os.path.exists(TRAINING_STATS_PATH):
            return jsonify({
                "error": f"Training stats not found at {TRAINING_STATS_PATH}. Run training to generate it."
            }), 400
        training_stats = load_training_stats(TRAINING_STATS_PATH)
        live = fetch_recent_inputs(limit=limit, db_path=DB_PATH)
        rep = drift_report(training_stats, live)
        rep["n_live_samples"] = len(live)
        return jsonify(rep), 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
