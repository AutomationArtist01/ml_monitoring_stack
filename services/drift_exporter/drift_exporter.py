"""
Drift Exporter – custom Prometheus exporter computing data drift on a rolling window.

    gateway ──POST /ingest {features, prediction, score}──►  rolling window (deque, WINDOW_SIZE rows)
                                                                     │  every COMPUTE_INTERVAL seconds
                                                                     ▼
                     reference profile (JSON built from training data by build_reference.py)
                                                                     │
                                                                     ▼
       /metrics:  drift_psi{feature}      Population Stability Index (numeric + categorical)
                  drift_ks_statistic{feature}, drift_ks_pvalue{feature}   (numeric only, two-sample KS)
                  drift_prediction_psi   PSI of the model *score* distribution (prediction drift)
                  drift_features_drifted  number of features with PSI > PSI_ALERT
                  drift_window_rows, drift_last_compute_timestamp, drift_compute_seconds

PSI rule of thumb:  < 0.10 stable · 0.10–0.25 moderate shift · > 0.25 significant shift
KS:                 p-value < 0.05 → distributions differ (statistically)

Env:
    REFERENCE_PATH      reference/telco_reference.json
    WINDOW_SIZE         500        rows kept in the rolling window
    MIN_ROWS            50         don't compute until this many rows have arrived
    COMPUTE_INTERVAL    15         seconds between recomputations
    PSI_ALERT           0.25       threshold used for drift_features_drifted
"""
import json
import os
import threading
import time
from collections import deque
from typing import Any

import numpy as np
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from scipy.stats import ks_2samp

REFERENCE_PATH = os.environ.get("REFERENCE_PATH", "reference/telco_reference.json")
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "500"))
MIN_ROWS = int(os.environ.get("MIN_ROWS", "50"))
COMPUTE_INTERVAL = float(os.environ.get("COMPUTE_INTERVAL", "15"))
PSI_ALERT = float(os.environ.get("PSI_ALERT", "0.25"))
EPS = 1e-4

with open(REFERENCE_PATH) as f:
    REF = json.load(f)
MODEL = REF.get("model", "model")
NUMERIC = REF["numeric"]          # {feature: {"edges": [...], "probs": [...], "values": [...sample...]}}
CATEGORICAL = REF["categorical"]  # {feature: {"probs": {category: p}}}
SCORE_REF = REF.get("score")      # {"edges": [...], "probs": [...]}

app = FastAPI(title="Drift Exporter", version="1.0.0")

L = ["model", "feature"]
PSI = Gauge("drift_psi", "Population Stability Index vs. reference", L + ["type"])
KS_STAT = Gauge("drift_ks_statistic", "Two-sample KS statistic vs. reference", L)
KS_P = Gauge("drift_ks_pvalue", "Two-sample KS p-value vs. reference", L)
PRED_PSI = Gauge("drift_prediction_psi", "PSI of the model score distribution vs. reference", ["model"])
DRIFTED = Gauge("drift_features_drifted", f"Number of features with PSI > {PSI_ALERT}", ["model"])
ROWS = Gauge("drift_window_rows", "Rows currently in the rolling window", ["model"])
LAST = Gauge("drift_last_compute_timestamp", "Unix time of last drift computation", ["model"])
DUR = Gauge("drift_compute_seconds", "Seconds taken by the last drift computation", ["model"])
INGESTED = Counter("drift_rows_ingested_total", "Rows received on /ingest", ["model"])
WINDOW_MEAN = Gauge("drift_window_mean", "Mean of numeric feature in the current window", L)
REF_MEAN = Gauge("drift_reference_mean", "Mean of numeric feature in the reference", L)
PSI_THRESHOLD = Gauge("drift_psi_threshold", "Configured PSI alert threshold", ["model"])
PSI_THRESHOLD.labels(MODEL).set(PSI_ALERT)

window: deque = deque(maxlen=WINDOW_SIZE)
lock = threading.Lock()


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPS, None)
    actual = np.clip(actual, EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def numeric_hist(values: np.ndarray, edges: list[float]) -> np.ndarray:
    counts, _ = np.histogram(values, bins=np.array(edges))
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def compute():
    with lock:
        rows = list(window)
    n = len(rows)
    ROWS.labels(MODEL).set(n)
    if n < MIN_ROWS:
        return
    t0 = time.perf_counter()
    drifted = 0

    for feat, ref in NUMERIC.items():
        vals = np.array([r["features"].get(feat) for r in rows if isinstance(r["features"].get(feat), (int, float))],
                        dtype=float)
        if len(vals) < MIN_ROWS:
            continue
        p = psi(np.array(ref["probs"]), numeric_hist(vals, ref["edges"]))
        PSI.labels(MODEL, feat, "numeric").set(p)
        drifted += p > PSI_ALERT
        stat, pval = ks_2samp(np.array(ref["values"]), vals)
        KS_STAT.labels(MODEL, feat).set(float(stat))
        KS_P.labels(MODEL, feat).set(float(pval))
        WINDOW_MEAN.labels(MODEL, feat).set(float(vals.mean()))
        REF_MEAN.labels(MODEL, feat).set(float(np.mean(ref["values"])))

    for feat, ref in CATEGORICAL.items():
        cats = list(ref["probs"].keys())
        counts = {c: 0 for c in cats}
        other = 0
        for r in rows:
            v = str(r["features"].get(feat))
            if v in counts:
                counts[v] += 1
            else:
                other += 1
        total = sum(counts.values()) + other
        if total < MIN_ROWS:
            continue
        expected = np.array([ref["probs"][c] for c in cats] + [EPS])
        actual = np.array([counts[c] / total for c in cats] + [other / total])
        p = psi(expected, actual)
        PSI.labels(MODEL, feat, "categorical").set(p)
        drifted += p > PSI_ALERT

    if SCORE_REF:
        scores = np.array([r["score"] for r in rows if isinstance(r.get("score"), (int, float))], dtype=float)
        if len(scores) >= MIN_ROWS:
            PRED_PSI.labels(MODEL).set(psi(np.array(SCORE_REF["probs"]), numeric_hist(scores, SCORE_REF["edges"])))

    DRIFTED.labels(MODEL).set(drifted)
    LAST.labels(MODEL).set(time.time())
    DUR.labels(MODEL).set(time.perf_counter() - t0)


def loop():
    while True:
        try:
            compute()
        except Exception as e:  # noqa: BLE001
            print("[drift] compute error:", e)
        time.sleep(COMPUTE_INTERVAL)


threading.Thread(target=loop, daemon=True).start()


@app.post("/ingest")
def ingest(payload: dict[str, Any]):
    feats = payload.get("features")
    if not isinstance(feats, dict):
        return Response(status_code=400, content=b'{"detail":"features must be an object"}')
    with lock:
        window.append({"features": feats, "score": payload.get("score"), "prediction": payload.get("prediction")})
    INGESTED.labels(MODEL).inc()
    return {"ok": True, "window_rows": len(window)}


@app.post("/reset")
def reset():
    with lock:
        window.clear()
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "window_rows": len(window), "window_size": WINDOW_SIZE}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
