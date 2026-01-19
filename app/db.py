import os
import sqlite3
from datetime import datetime

DEFAULT_DB_PATH = os.getenv("DB_PATH", "predictions.db")


def _connect(db_path: str = DEFAULT_DB_PATH):
    return sqlite3.connect(db_path)


def init_db(db_path: str = DEFAULT_DB_PATH):
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            entity TEXT,
            text TEXT,
            label TEXT,
            confidence REAL
        )
        """
    )
    conn.commit()
    conn.close()


def log_prediction(entity: str, text: str, label: str, confidence: float, db_path: str = DEFAULT_DB_PATH):
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (ts, entity, text, label, confidence) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), entity, text, label, float(confidence)),
    )
    conn.commit()
    conn.close()


def get_logs(limit: int = 50, db_path: str = DEFAULT_DB_PATH):
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT ts, entity, label, confidence, text FROM predictions ORDER BY id DESC LIMIT ?",
        (int(limit),),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {"ts": r[0], "entity": r[1], "label": r[2], "confidence": r[3], "text": r[4]}
        for r in rows
    ]


def get_metrics(db_path: str = DEFAULT_DB_PATH):
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM predictions")
    total = int(cur.fetchone()[0])
    cur.execute("SELECT label, COUNT(*) FROM predictions GROUP BY label")
    by_label = {label: int(cnt) for label, cnt in cur.fetchall()}
    cur.execute("SELECT AVG(confidence) FROM predictions")
    avg_conf = cur.fetchone()[0]
    conn.close()
    return {
        "total_predictions": total,
        "by_label": by_label,
        "avg_confidence": (float(avg_conf) if avg_conf is not None else None),
    }


def fetch_recent_inputs(limit: int = 500, db_path: str = DEFAULT_DB_PATH):
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT entity, text FROM predictions ORDER BY id DESC LIMIT ?",
        (int(limit),),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"entity": r[0], "text": r[1]} for r in rows]
