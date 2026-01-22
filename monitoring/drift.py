import json
import math
from typing import Dict, List

from app.preprocess import clean_text


def _psi(expected: List[float], actual: List[float], bins: int = 10) -> float:
    # Population Stability Index for numeric distributions
    if not expected or not actual:
        return 0.0
    mn = min(min(expected), min(actual))
    mx = max(max(expected), max(actual))
    if mn == mx:
        return 0.0
    step = (mx - mn) / bins
    def to_hist(arr: List[float]) -> List[float]:
        h = [0] * bins
        for v in arr:
            idx = int((v - mn) / step)
            if idx == bins:
                idx = bins - 1
            h[idx] += 1
        total = sum(h)
        return [x / total for x in h]

    e = to_hist(expected)
    a = to_hist(actual)
    psi = 0.0
    eps = 1e-6
    for ei, ai in zip(e, a):
        ei = max(ei, eps)
        ai = max(ai, eps)
        psi += (ai - ei) * math.log(ai / ei)
    return float(psi)


def compute_live_stats(records: List[Dict]) -> Dict:
    lengths = []
    url_like = []
    for r in records:
        t = r.get("text", "")
        c = clean_text(t)
        lengths.append(len(c.split()))
        url_like.append(1 if ("http" in (t or "").lower() or "www" in (t or "").lower()) else 0)
    return {
        "tweet_len_words": lengths,
        "has_url": url_like,
    }


def load_training_stats(path: str = "monitoring/training_stats.json") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def drift_report(training_stats: Dict, live_records: List[Dict]) -> Dict:
    live = compute_live_stats(live_records)
    report = {}
    report["psi_tweet_len_words"] = _psi(training_stats.get("tweet_len_words", []), live.get("tweet_len_words", []))
    report["psi_has_url"] = _psi(training_stats.get("has_url", []), live.get("has_url", []), bins=2)

    # Simple overall drift score
    report["drift_score"] = round((report["psi_tweet_len_words"] + report["psi_has_url"]) / 2.0, 6)

    # Rule-of-thumb thresholds for PSI
    # <0.1 no drift, 0.1-0.2 moderate, >0.2 significant
    report["status"] = (
        "ok" if report["drift_score"] < 0.1 else
        "moderate" if report["drift_score"] < 0.2 else
        "significant"
    )
    return report
