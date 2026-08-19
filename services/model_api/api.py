# api.py — the other team's churn model API + transformer-order fix + Prometheus ML metrics + demo /chaos
# --- Configuration (env + model_meta.json) ---
# model/model_meta.json is written by the training pipeline's export step (`make promote`); when present the API
# serves that MLflow @champion model (a full sklearn Pipeline) and reports its registry name/version/run_id.
# Without it, the legacy artifact (the other team's skops model + external preprocessing) is served.
import json as _json
import os
import random
import threading
import time

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from prometheus_client import Counter, Gauge, Histogram, Info
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(HERE, "model"))
_meta_path = os.path.join(MODEL_PATH, "model_meta.json")
META = _json.load(open(_meta_path)) if os.path.exists(_meta_path) else {}
MODEL_NAME = os.environ.get("MODEL_NAME") or META.get("registered_name", "CustomerChurnGradientBoosting")
MODEL_VERSION = os.environ.get("MODEL_VERSION") or str(META.get("version", "1"))
MODEL_KIND = META.get("kind", "legacy-skops")          # "sklearn-pipeline" (champion export) | "legacy-skops"
MODEL_RUN_ID = META.get("run_id", "")
MODEL_ALIAS = META.get("alias", "")
CHAOS_ENABLED = os.environ.get("CHAOS_ENABLED", "1") == "1"

# --- App ---
app = FastAPI(
    title="Customer Churn Prediction API (monitored)",
    description="Telco churn model from the other team, wrapped for the Team-4 monitoring stack",
    version="1.1.0",
)


# Prometheus – HTTP-level metrics (http_requests_total, http_request_duration_seconds…)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# Prometheus – ML-level metrics (the four signals from the assignment)

LABELS = ["model", "version"]
prediction_requests = Counter(
    "prediction_requests_total", "Total prediction requests", LABELS
)
prediction_errors = Counter(
    "prediction_errors_total", "Total prediction errors", LABELS + ["reason"]
)
prediction_latency = Histogram(
    "prediction_request_latency_seconds",
    "End-to-end latency of /predict in seconds",
    LABELS,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
prediction_probability = Histogram(
    "prediction_probability",
    "Distribution of predicted churn probability (class 1)",
    LABELS,
    buckets=tuple(round(i / 10, 1) for i in range(1, 10)) + (1.0,),
)
predictions_by_class = Counter(
    "predictions_total", "Predictions by predicted class", LABELS + ["predicted_class"]
)
model_info = Info("model", "Model metadata")
model_info.info({"name": MODEL_NAME, "version": MODEL_VERSION, "framework": "sklearn-gbm", "kind": MODEL_KIND,
                 "run_id": MODEL_RUN_ID, "alias": MODEL_ALIAS})
model_loaded = Gauge("model_loaded", "1 if the model is loaded and ready")
chaos_latency_ms = Gauge("chaos_injected_latency_ms", "Artificial latency currently injected (demo)")
chaos_error_rate = Gauge("chaos_injected_error_rate", "Artificial error probability currently injected (demo)")


# Input schema (identical to the other team's API)

# --- Input schema ---
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# --- Feature definitions ---
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# --- Preprocessing (order matches training) ---
# BUG FIX: numeric FIRST, then categorical – same order as src/steps/preprocess.py (training).
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ]
)

DATASET_PATH = os.environ.get("DATASET_PATH", os.path.join(HERE, "data", "telco_churn.csv"))

# --- Load model (+ fit the external preprocessing only for the legacy artifact) ---
model = mlflow.sklearn.load_model(MODEL_PATH)
if MODEL_KIND == "sklearn-pipeline":
    feature_count = len(NUMERIC) + len(CATEGORICAL)          # preprocessing lives inside the pipeline
else:
    dataset = pd.read_csv(DATASET_PATH)
    dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce").fillna(0)
    preprocessor.fit(dataset.drop(columns=["customerID", "Churn"], errors="ignore"))
    feature_count = len(preprocessor.get_feature_names_out())
model_loaded.set(1)
print(f"[model_api] loaded {MODEL_NAME} v{MODEL_VERSION} ({MODEL_KIND}, run {MODEL_RUN_ID or '-'}) – {feature_count} features")


# Chaos state (demo only)

# --- Chaos state (demo only) ---
_chaos = {"latency_ms": 0, "error_rate": 0.0}
_lock = threading.Lock()


# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs", "health": "/health", "metrics": "/metrics"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME, "version": MODEL_VERSION, "features": feature_count,
            "kind": MODEL_KIND, "run_id": MODEL_RUN_ID, "alias": MODEL_ALIAS,
            "source": "mlflow-registry" if MODEL_KIND == "sklearn-pipeline" else "other-team-artifact"}


@app.post("/chaos")
def chaos(latency_ms: int = Query(0, ge=0, le=10000), error_rate: float = Query(0.0, ge=0.0, le=1.0)):
    """Demo helper: inject latency / errors so alerts can be demonstrated."""
    if not CHAOS_ENABLED:
        raise HTTPException(status_code=403, detail="chaos disabled")
    with _lock:
        _chaos["latency_ms"] = latency_ms
        _chaos["error_rate"] = error_rate
    chaos_latency_ms.set(latency_ms)
    chaos_error_rate.set(error_rate)
    return _chaos


# --- Prediction endpoint ---
@app.post("/predict")
def predict(customer: CustomerData):
    start = time.perf_counter()
    lbl = (MODEL_NAME, MODEL_VERSION)
    prediction_requests.labels(*lbl).inc()
    try:
        with _lock:
            lat, err = _chaos["latency_ms"], _chaos["error_rate"]
        if lat:
            time.sleep(lat / 1000)
        if err and random.random() < err:
            raise RuntimeError("chaos: injected failure")

        row = pd.DataFrame([customer.model_dump()])
        x = row[NUMERIC + CATEGORICAL] if MODEL_KIND == "sklearn-pipeline" else preprocessor.transform(row)
        proba = float(model.predict_proba(x)[0][1])
        pred = int(proba >= 0.5)

        prediction_probability.labels(*lbl).observe(proba)
        predictions_by_class.labels(*lbl, str(pred)).inc()
        return {"prediction": pred, "churn": "Yes" if pred else "No", "churn_probability": round(proba, 4),
                "model": MODEL_NAME, "version": MODEL_VERSION}
    except RuntimeError as e:
        prediction_errors.labels(*lbl, "injected").inc()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # noqa: BLE001
        prediction_errors.labels(*lbl, "exception").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        prediction_latency.labels(*lbl).observe(time.perf_counter() - start)
