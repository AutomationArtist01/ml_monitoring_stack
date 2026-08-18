# loadgen.py — replays real Telco rows through the gateway; demo scenarios switchable via HTTP
import os
import random
import threading
import time

import httpx
import pandas as pd
from fastapi import FastAPI, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

# --- Configuration (env) ---
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://gateway:8080").rstrip("/")
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://model-api:8000").rstrip("/")
DATA_PATH = os.environ.get("DATA_PATH", "data/telco_churn.csv")
DEFAULT_RPS = float(os.environ.get("RPS", "5"))

# --- Load data ---
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
ROWS = df.drop(columns=["customerID", "Churn"]).to_dict(orient="records")

# --- Scenario state + metrics ---
state = {"scenario": "normal", "rps": DEFAULT_RPS, "drift": False, "bad_payload": False}
SCENARIOS = ["normal", "drift", "latency", "errors", "burst", "bad_payload"]

REQ = Counter("loadgen_requests_total", "Requests sent by the load generator", ["status"])
SCEN = Gauge("loadgen_scenario", "Active scenario (1 = active)", ["scenario"])
RPS = Gauge("loadgen_target_rps", "Configured requests per second")
for s in SCENARIOS:
    SCEN.labels(s).set(1 if s == "normal" else 0)
RPS.set(DEFAULT_RPS)

# --- App ---
app = FastAPI(title="Load Generator", version="1.0.0")


# --- Helpers ---
def set_scenario(name: str):
    state["scenario"] = name
    for s in SCENARIOS:
        SCEN.labels(s).set(1 if s == name else 0)


def sample_row() -> dict:
    row = dict(random.choice(ROWS))
    if state["drift"]:
        # simulate a changed customer population: young, high-charge, month-to-month customers
        row["tenure"] = int(max(0, random.gauss(4, 3)))
        row["MonthlyCharges"] = round(min(120.0, max(20.0, row["MonthlyCharges"] * random.uniform(1.25, 1.6))), 2)
        row["TotalCharges"] = round(row["tenure"] * row["MonthlyCharges"], 2)
        if random.random() < 0.85:
            row["Contract"] = "Month-to-month"
        if random.random() < 0.7:
            row["InternetService"] = "Fiber optic"
        if random.random() < 0.6:
            row["PaymentMethod"] = "Electronic check"
    return row


# --- Traffic worker ---
def worker():
    with httpx.Client(timeout=15) as client:
        while True:
            rps = max(0.1, state["rps"])
            t0 = time.perf_counter()
            try:
                if state["bad_payload"] and random.random() < 0.3:
                    r = client.post(f"{GATEWAY_URL}/predict", content=b'{"gender": "Female", "tenure": "oops"',
                                    headers={"content-type": "application/json"})
                else:
                    r = client.post(f"{GATEWAY_URL}/predict", json=sample_row())
                REQ.labels(str(r.status_code)).inc()
            except Exception:  # noqa: BLE001
                REQ.labels("conn_error").inc()
            time.sleep(max(0.0, 1.0 / rps - (time.perf_counter() - t0)))


def chaos(latency_ms: int = 0, error_rate: float = 0.0):
    try:
        httpx.post(f"{MODEL_API_URL}/chaos", params={"latency_ms": latency_ms, "error_rate": error_rate}, timeout=5)
    except Exception as e:  # noqa: BLE001
        print("[loadgen] chaos call failed:", e)


# --- Endpoints: status + scenarios ---
@app.on_event("startup")
def _start():
    time.sleep(3)
    threading.Thread(target=worker, daemon=True).start()


@app.get("/status")
def status():
    return state


@app.post("/scenario/normal")
def s_normal():
    state.update(drift=False, bad_payload=False, rps=DEFAULT_RPS)
    RPS.set(DEFAULT_RPS)
    chaos(0, 0.0)
    set_scenario("normal")
    return state


@app.post("/scenario/drift")
def s_drift():
    state.update(drift=True, bad_payload=False)
    chaos(0, 0.0)
    set_scenario("drift")
    return state


@app.post("/scenario/latency")
def s_latency(ms: int = Query(800, ge=0, le=10000)):
    state.update(drift=False, bad_payload=False)
    chaos(ms, 0.0)
    set_scenario("latency")
    return {**state, "latency_ms": ms}


@app.post("/scenario/errors")
def s_errors(rate: float = Query(0.3, ge=0, le=1)):
    state.update(drift=False, bad_payload=False)
    chaos(0, rate)
    set_scenario("errors")
    return {**state, "error_rate": rate}


@app.post("/scenario/burst")
def s_burst(rps: float = Query(40, ge=1, le=200)):
    state.update(drift=False, bad_payload=False, rps=rps)
    RPS.set(rps)
    chaos(0, 0.0)
    set_scenario("burst")
    return state


@app.post("/scenario/bad_payload")
def s_bad():
    state.update(drift=False, bad_payload=True)
    chaos(0, 0.0)
    set_scenario("bad_payload")
    return state


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
