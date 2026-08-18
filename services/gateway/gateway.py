# gateway.py — model-agnostic Prometheus exporter/proxy: put it in front of ANY HTTP model (env vars only)
import asyncio
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# --- Configuration (env) ---
UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://model-api:8000").rstrip("/")
PREDICT_PATH = os.environ.get("PREDICT_PATH", "/predict")
HEALTH_PATH = os.environ.get("HEALTH_PATH", "/health")
MODEL_NAME = os.environ.get("MODEL_NAME", "model")
PROB_FIELD = os.environ.get("PROB_FIELD", "churn_probability")
LABEL_FIELD = os.environ.get("LABEL_FIELD", "prediction")
DRIFT_URL = os.environ.get("DRIFT_URL", "http://drift-exporter:9105").rstrip("/")
UPSTREAM_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "10"))

# --- App + Prometheus metrics ---
app = FastAPI(title="ML Monitoring Gateway", version="1.0.0")

L = ["model"]
REQS = Counter("mlgw_requests_total", "Prediction requests through the gateway", L + ["status"])
ERRS = Counter("mlgw_errors_total", "Failed prediction requests", L + ["reason"])
LAT = Histogram(
    "mlgw_request_latency_seconds", "Upstream prediction latency (seconds)", L,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
PROB = Histogram(
    "mlgw_prediction_score", "Distribution of the model score/probability", L,
    buckets=tuple(round(i / 10, 1) for i in range(1, 10)) + (1.0,),
)
CLASS = Counter("mlgw_predictions_total", "Predictions by predicted class", L + ["predicted_class"])
INFLIGHT = Gauge("mlgw_inflight_requests", "In-flight prediction requests", L)
UP = Gauge("mlgw_upstream_up", "1 if the upstream model /health responds 200", L)
DRIFT_SENT = Counter("mlgw_drift_events_total", "Feature rows forwarded to the drift exporter", L + ["status"])
LAST_SCORE = Gauge("mlgw_last_score", "Score of the most recent prediction", L)


# --- Helpers ---
def _get(d: Any, dotted: str):
    for k in dotted.split("."):
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


client: httpx.AsyncClient | None = None


# --- Startup: upstream health probe ---
@app.on_event("startup")
async def _startup():
    global client
    client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    asyncio.create_task(_health_loop())


async def _health_loop():
    while True:
        try:
            r = await client.get(UPSTREAM_URL + HEALTH_PATH)
            UP.labels(MODEL_NAME).set(1 if r.status_code == 200 else 0)
        except Exception:  # noqa: BLE001
            UP.labels(MODEL_NAME).set(0)
        await asyncio.sleep(10)


# --- Feed the drift exporter (async) ---
async def _send_drift(features: dict, prediction: Any, score: Any):
    if not DRIFT_URL:
        return
    try:
        r = await client.post(f"{DRIFT_URL}/ingest",
                              json={"model": MODEL_NAME, "features": features,
                                    "prediction": prediction, "score": score})
        DRIFT_SENT.labels(MODEL_NAME, "ok" if r.status_code < 300 else "error").inc()
    except Exception:  # noqa: BLE001
        DRIFT_SENT.labels(MODEL_NAME, "error").inc()


# --- Endpoints ---
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM_URL, "model": MODEL_NAME}


# --- Proxy /predict and record the four signals ---
@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    INFLIGHT.labels(MODEL_NAME).inc()
    start = time.perf_counter()
    status = "500"
    try:
        r = await client.post(UPSTREAM_URL + PREDICT_PATH, content=body,
                              headers={"content-type": request.headers.get("content-type", "application/json")})
        status = str(r.status_code)
        if r.status_code >= 500:
            ERRS.labels(MODEL_NAME, "upstream_5xx").inc()
        elif r.status_code >= 400:
            ERRS.labels(MODEL_NAME, "client_4xx").inc()
        else:
            try:
                data = r.json()
                score = _get(data, PROB_FIELD) if PROB_FIELD else None
                label = _get(data, LABEL_FIELD) if LABEL_FIELD else None
                if isinstance(score, (int, float)):
                    PROB.labels(MODEL_NAME).observe(float(score))
                    LAST_SCORE.labels(MODEL_NAME).set(float(score))
                if label is not None:
                    CLASS.labels(MODEL_NAME, str(label)).inc()
                try:
                    features = await request.json()
                except Exception:  # noqa: BLE001
                    features = None
                if isinstance(features, dict):
                    asyncio.create_task(_send_drift(features, label, score))
            except ValueError:
                pass
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))
    except httpx.TimeoutException:
        ERRS.labels(MODEL_NAME, "timeout").inc()
        status = "504"
        return Response(content=b'{"detail":"upstream timeout"}', status_code=504, media_type="application/json")
    except httpx.HTTPError as e:
        ERRS.labels(MODEL_NAME, "connection").inc()
        status = "502"
        return Response(content=f'{{"detail":"upstream unreachable: {e.__class__.__name__}"}}'.encode(),
                        status_code=502, media_type="application/json")
    finally:
        LAT.labels(MODEL_NAME).observe(time.perf_counter() - start)
        REQS.labels(MODEL_NAME, status).inc()
        INFLIGHT.labels(MODEL_NAME).dec()
