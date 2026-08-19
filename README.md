# Real-Time Model Monitoring & Alerting Stack

> **MLOps Project 4 · Team 4** — a reusable, model-agnostic observability stack that tells you *when a deployed ML model
> goes wrong* — not just when the server goes down.

**Prometheus · Grafana · Alertmanager · custom PSI/KS drift exporter · psutil hardware exporter · ZenML · MLflow · Optuna · Docker Compose · GitHub Actions · Render**

[![CI](https://github.com/AutomationArtist01/ml_monitoring_stack/actions/workflows/ci.yml/badge.svg)](https://github.com/AutomationArtist01/ml_monitoring_stack/actions/workflows/ci.yml)

---

## Table of contents
1. [Why this exists](#1-why-this-exists)
2. [What it does](#2-what-it-does)
3. [Quick start](#3-quick-start)
4. [Live demo scenarios](#4-live-demo-scenarios)
5. [Architecture](#5-architecture)
6. [Repository layout](#6-repository-layout)
7. [Training pipeline (ZenML + Optuna + MLflow)](#7-training-pipeline-zenml--optuna--mlflow)
8. [Dashboards & alerts](#8-dashboards--alerts)
9. [Wire it to *your* model](#9-wire-it-to-your-model)
10. [Deployment & CI/CD](#10-deployment--cicd)
11. [Tests](#11-tests)
12. [Success metrics](#12-success-metrics)
13. [Documentation](#13-documentation)

---

## 1 · Why this exists

Machine-learning services fail **silently**. The endpoint keeps returning `200 OK` while the model's inputs drift away
from the training data, or a bug scrambles the features before they reach the model. CPU/memory/uptime monitoring
cannot see that.

This project wraps a deployed model — the **Telco Customer-Churn** classifier built by the *customer-churn-mlops* team —
with a stack that watches both:

| Operational health | ML health |
|---|---|
| request rate · latency p50/p95/p99 · error rate · saturation (CPU/RAM/disk) | prediction distribution · positive-rate · **data drift** (PSI + KS-test per feature) · prediction drift |

…and **alerts humans** on Slack / e-mail through Alertmanager. Wrapping *another team's* model was deliberate: it proves
the stack does not depend on any single model.

**Dataset:** IBM Telco Customer Churn (7 043 customers, 19 features, 26.5 % churners) — the data the model was trained
on; used for retraining/tuning, as the drift *reference* distribution, and to replay realistic traffic.

## 2 · What it does

| Component | What you get |
|---|---|
| **Gateway** (`services/gateway`) | Put it in front of *any* HTTP model → request rate, latency histogram, error rate, prediction distribution, upstream health. **No model code changes** — env vars only. |
| **Model API** (`services/model_api`) | The other team's FastAPI churn service + the same metrics *inside* the code (`prometheus_client`) — both instrumentation styles shown. |
| **Drift exporter** (`services/drift_exporter`) | PSI + Kolmogorov–Smirnov per feature and PSI on the model score, on a 500-row rolling window vs a training reference. |
| **System exporter** (`services/system_exporter`) | psutil: host CPU / RAM / swap / disk / network / load + per-container CPU & memory. |
| **Prometheus** | scrapes everything every 5 s; 10 recording rules (SLIs); **18 alert rules** (operational · ML-quality · resources). |
| **Alertmanager** | routes by severity/category to Slack channels + e-mail; grouping, inhibition, silences. |
| **Grafana** | 5 dashboards, 84 panels, generated as code — Ops · Predictions · Drift · Stack health (incl. hardware). |
| **ZenML + Optuna + MLflow** (`services/trainer`) | validate (gate) → split → 25-trial Optuna tuning → evaluate → quality-gated Model Registry promotion → drift reference. |
| **Load generator** | replays real customers; one-command demo scenarios (drift / latency / errors / burst). |
| **Docker Compose · Render · GitHub Actions** | one command locally; Render blueprint for the API; CI = lint + tests + config validation + image builds + full-stack smoke test. |

## 3 · Quick start

Requires **Docker Desktop** only (Python 3.12 for `make test`).

```bash
git clone https://github.com/AutomationArtist01/ml_monitoring_stack.git && cd ml_monitoring_stack
make up          # first time: build + start 12 containers (a few minutes)
```
Later:
```bash
make start       # start without rebuilding (~20 s)
make down        # stop (data is kept)      make nuke = stop + wipe data
```
Optional:
```bash
make train       # ZenML pipeline: validate → Optuna (25 trials) → evaluate → register → drift reference (~1 min)
make smoke       # end-to-end check against the running stack (25 assertions)
make test        # unit tests, no Docker needed
```

Then open:

| UI | URL | Notes |
|---|---|---|
| **Grafana** | http://localhost:3000 | admin / admin → *Dashboards → ML Monitoring* |
| Prometheus | http://localhost:9090 | targets, PromQL, alerts |
| Alertmanager | http://localhost:9093 | grouped alerts, receivers, silences |
| MLflow | http://localhost:5001 | runs, comparison, model registry |
| **ZenML dashboard** | http://localhost:8237 | pipeline DAG (`churn_training_pipeline`) and every run's steps/artifacts — first visit asks for a name |
| Model API | http://localhost:8000/docs | Swagger UI — `/predict`, `/health`, `/metrics` · **cloud:** https://team4-churn-model-api.onrender.com/docs |
| Load generator | http://localhost:8090/docs | switch demo scenarios |
| Fake e-mail (Mailhog) | http://localhost:8025 | e-mail alerts land here |
| Fake Slack | http://localhost:8091 | Slack payloads land here |
| Raw metrics | :8080/metrics (gateway) · :9105/metrics (drift) · :9106/metrics (hardware) | |

Direct dashboard links: [Model API Ops](http://localhost:3000/d/ml-ops) · [Predictions](http://localhost:3000/d/ml-predictions) · [Data Drift](http://localhost:3000/d/ml-drift) · [Stack Health](http://localhost:3000/d/ml-stack) · [Hardware & Resources](http://localhost:3000/d/ml-hardware)

> Real Slack / SMTP: copy `.env.example` → `.env`, set `SLACK_WEBHOOK_URL` and `SMTP_*`, restart. Nothing else changes.

## 4 · Live demo scenarios

The stack starts in a healthy state (flat, green graphs — that is correct). To see it *work*:

```bash
make scenario-drift      # shift the customer population → PSI/KS rise, drift alerts fire (~2 min)
make scenario-latency    # inject 800 ms → p95 alert
make scenario-errors     # inject 30 % failures → error-rate alert (critical → Slack #ml-oncall + e-mail)
make scenario-burst      # 40 req/s → traffic spike
make scenario-normal     # back to normal → RESOLVED notifications
python3 tests/alert_status.py   # what fired, how it was routed, what Slack/e-mail received
```

## 5 · Architecture

```
                       demo scenarios (normal / drift / latency / errors / burst)
                       ┌───────────────────┐
                       │  load-generator   │ :8090
                       └────────┬──────────┘
                                │ POST /predict
                                ▼
┌───────────────┐      ┌───────────────────┐  POST /predict   ┌───────────────────────┐
│  any client   │─────►│  GATEWAY  :8080   │─────────────────►│  model-api  :8000     │
└───────────────┘      │  /metrics mlgw_*  │◄─────────────────│  churn model + metrics│
                       └───┬───────────┬───┘                  └───────────────────────┘
                           │           │ async /ingest
                           │  ┌────────▼──────────┐   ┌───────────────────┐
                           │  │  drift-exporter   │   │  system-exporter  │ psutil: CPU · RAM ·
                           │  │  :9105 PSI + KS   │   │  :9106 hardware   │ disk · net · containers
                           │  └─────────┬─────────┘   └─────────┬─────────┘
   scrape every 5 s        │            │                       │
┌──────────────────────────┴────────────┴───────────────────────┴───────────────────┐
│  PROMETHEUS :9090     TSDB · recording rules (ml:*) · 18 alert rules              │
└──────────┬──────────────────────────────────────────────────┬────────────────────┘
           ▼ alerts                                            ▼ PromQL
┌───────────────────────┐   Slack / SMTP   ┌──────────┐    ┌───────────────────────────┐
│  ALERTMANAGER :9093   │─────────────────►│ Slack    │    │  GRAFANA :3000            │
│  group·route·inhibit  │                  │ Mailhog  │    │  4 dashboards (as code)   │
└───────────────────────┘                  └──────────┘    └───────────────────────────┘
┌───────────────────────┐    ┌──────────────────────────────────────────────────────┐
│  MLFLOW :5001         │◄───│  trainer = ZenML pipeline (Optuna, registry, drift ref)│
└───────────────────────┘    └──────────────────────────────────────────────────────┘
```

## 6 · Repository layout

```
├── docker-compose.yml               the stack (11 services)
├── docker-compose.second-model.yml  overlay: monitor a 2nd model (the other team's ORIGINAL image, unmodified)
├── Makefile · .env.example · render.yaml · .github/workflows/ci.yml
├── services/
│   ├── gateway/          model-agnostic exporter/proxy (mlgw_* metrics)
│   ├── model_api/        churn model API + in-process metrics + demo /chaos
│   ├── drift_exporter/   PSI + KS rolling window · build_reference.py · reference profile
│   ├── system_exporter/  psutil hardware + per-container stats
│   ├── load_generator/   traffic replay + scenarios
│   ├── trainer/          ZenML pipeline (pipeline.py) · run_pipeline.py
│   ├── mlflow/           MLflow server image
│   └── webhook_catcher/  fake Slack
├── monitoring/
│   ├── prometheus/       prometheus.yml · rules/recording_rules.yml · rules/alert_rules.yml
│   ├── grafana/          provisioning/ · build_dashboards.py → dashboards/*.json
│   └── alertmanager/     alertmanager.yml · templates/slack.tmpl · entrypoint.sh
├── tests/                test_units.py · smoke_test.py · alert_status.py · sample_request.json
├── data/telco_churn.csv
└── external/customer-churn-mlops/   snapshot of the other team's repo (+ PROVENANCE.md)
```
Every service has its own `Dockerfile`, `requirements.in` and a fully pinned `requirements.txt` (`uv pip compile`).

## 7 · Training pipeline (ZenML + Optuna + MLflow)

```
ingest_data → validate_data ─╳ halts on missing columns / nulls > 5 % / minority class < 10 %
            → split_data (stratified 80/20)
            → tune_and_train   Optuna TPE + MedianPruner · 25 trials · 3-fold CV AUC · every trial = nested MLflow run
            → evaluate_model   hold-out AUC / F1 / precision / recall (logged with the model + signature)
            → register_model   gate AUC ≥ 0.80 → Model Registry `telco-churn-classifier` @champion, stage=production
            → build_drift_reference   writes the PSI/KS reference for the drift exporter
            → export_runs_csv
```
Last run: 25 trials (8 pruned) · best CV AUC 0.850 · hold-out AUC 0.847 / F1 0.587 · registered v3.
`make train` (or `make train-quick` for 8 trials). Runs appear in the **ZenML dashboard** (http://localhost:8237) and in MLflow.

## 8 · Dashboards & alerts

| Dashboard | Shows |
|---|---|
| **1 · Model API – Operational** | request rate, p50/p95/p99 vs SLO, error ratio & reasons, availability, latency heatmap, demo state |
| **2 · Predictions** | score histogram/heatmap, positive-rate, prediction PSI, in-process vs gateway latency |
| **3 · Data Drift** | PSI per feature (0.10 / 0.25 thresholds), PSI/KS over time, live-vs-reference means, drift table |
| **4 · Stack Health** | targets up, firing alerts, Prometheus/Alertmanager internals, hardware row |
| **5 · Hardware & Resources** | psutil: CPU/RAM/disk/swap gauges, per-core CPU, memory vs total, network in/out, disk, load vs cores, per-container CPU & memory (stacked + ranked) |

Alerts (18): `ModelServiceDown` `UpstreamModelUnreachable` `HighLatencyP95` `HighLatencyP99Critical` `HighErrorRate`
`NoTraffic` `TrafficSpike` · `FeatureDriftModerate` `FeatureDriftHigh` `FeatureDriftKS` `MultipleFeaturesDrifted`
`PredictionDrift` `PositiveRateAnomaly` `DriftExporterStale` · `HostHighCPU` `HostHighMemory` `HostDiskAlmostFull`
`ContainerHighMemory`. Routing: critical → Slack `#ml-oncall` + e-mail · ML-quality → `#ml-quality` · warning → `#ml-alerts`.

## 9 · Wire it to *your* model

1. **Gateway (no code changes):** add your model + a gateway to Compose with `UPSTREAM_URL`, `PREDICT_PATH`,
   `HEALTH_PATH`, `MODEL_NAME`, `PROB_FIELD`, `LABEL_FIELD` (see `docker-compose.second-model.yml`), add one
   `job_name` to `monitoring/prometheus/prometheus.yml`, `make reload-prom`, send traffic to the gateway.
   All operational dashboards and alerts work immediately (everything is grouped by the `model` label).
2. **Drift:** `python services/drift_exporter/build_reference.py --csv train.csv --numeric … --categorical … --model NAME --out reference/NAME.json`
   and run a drift-exporter instance with `REFERENCE_PATH`.
3. **Optional in-process metrics:** ten lines of `prometheus_client` in your service (see `services/model_api/api.py`).

Proof: `docker compose -f docker-compose.yml -f docker-compose.second-model.yml up -d` monitors the other team's
**unmodified** image next to ours.

## 10 · Deployment & CI/CD

* **Local:** Docker Compose (`make up` / `make start`), health checks, named volumes for state.
* **Cloud (live):** the model API is deployed on Render from `render.yaml` — **https://team4-churn-model-api.onrender.com**
  ([/health](https://team4-churn-model-api.onrender.com/health) · [/docs](https://team4-churn-model-api.onrender.com/docs) · [/metrics](https://team4-churn-model-api.onrender.com/metrics); free plan, first request after idle takes ~30 s).
  Point a gateway's `UPSTREAM_URL` at it to monitor the cloud instance from the stack.
* **CI (GitHub Actions):** on every push/PR — Ruff → pytest → `promtool` / `amtool` / `docker compose config` →
  dashboards reproducible from code → build all images (pushed to GHCR on `main`) → boot the whole stack in the runner
  and run the smoke test.

## 11 · Tests

| Command | What |
|---|---|
| `make test` | pytest — PSI/KS maths, drift detection, gateway helper, reference builder, ZenML validation gate, config sanity |
| `make smoke` | 25 assertions against the running stack (health, metrics, targets, rules, drift window, dashboards, MLflow) |
| `make lint` | Ruff |

## 12 · Success metrics

| Area | Metric | Target |
|---|---|---|
| Model quality | hold-out ROC-AUC (primary), F1 | ≥ 0.84 · registry gate refuses < 0.80 |
| Serving SLOs | p95 latency · 5xx ratio · availability (30 m) | < 500 ms · < 5 % · ≥ 99 % |
| Drift | PSI / KS per feature; time-to-detect | PSI > 0.25 flagged; alert within ≤ 3 min at 5 req/s |
| Saturation | host CPU / RAM / disk, container memory | alerts at 85 % / 90 % / 90 % |
| Alerting | delivery to the right receiver | 100 % of scenarios (Slack + e-mail) |
| Reusability | effort to monitor a new model | env vars + one scrape job |

Hour budget (15 h): exporters 5 · Compose 3 · dashboards 3 · Alertmanager 2 · docs/demo 2.

## 13 · Documentation

* **[OPERATIONS.md](OPERATIONS.md)** — team operations guide: start/stop, **add your e-mail/Slack for alerts**, drills, what to do when an alert fires, retrain, monitor another model.

The **project report** and the **technical guidebook** (why each tool, alternatives, PromQL cookbook, demo script,
troubleshooting, examiner Q&A) are submitted with the LMS submission.

---
*Team 4 — Gurrala Rohith Kumar · Vijeta S Alavani · Abhilash M K*
