# Team 4 · Real-Time Model Monitoring & Alerting Stack

[![CI](https://github.com/AutomationArtist01/MLOPS/actions/workflows/ci.yml/badge.svg)](https://github.com/AutomationArtist01/MLOPS/actions/workflows/ci.yml)

## 1 · Problem statement

Machine-learning services fail *silently*: the HTTP endpoint keeps returning `200 OK` while the model's
inputs drift away from the training data or a serving bug scrambles the features. Ordinary
infrastructure monitoring (CPU, memory, uptime) cannot see this. **Goal:** build a reusable,
model-agnostic observability stack that gives any deployed model — starting with the Telco
Customer-Churn classifier from the *customer-churn-mlops* team — real-time visibility of its
**operational health** (traffic, latency, errors) *and* its **ML health** (prediction distribution,
data drift), and that **alerts humans** through Slack/e-mail before business impact.

**Dataset.** IBM Telco Customer Churn (7 043 customers, 19 features, 26.5 % churners) — the dataset the
wrapped model was trained on; it is used (a) to retrain/tune the model in our ZenML + Optuna pipeline,
(b) as the drift *reference* distribution, and (c) to replay realistic traffic. No new dataset was
needed: the assignment is about the monitoring stack, not a new model.

**Success metrics (defined up-front).**

| Area | Metric | Target |
|---|---|---|
| Model quality (tuning) | ROC-AUC on the hold-out set (primary), F1 (secondary) | ≥ 0.84 AUC (other team's baseline 0.845) · registry gate refuses < 0.80 |
| Serving SLOs | p95 latency · 5xx error ratio · availability (30 m) | < 500 ms · < 5 % · ≥ 99 % |
| Drift detection | PSI per feature / KS-test; time-to-detect after a distribution change | PSI > 0.25 flagged; alert within ≤ 3 min at 5 req/s (500-row window) |
| Alerting | routing correctness (severity → receiver), grouped notifications | 100 % of test scenarios delivered to Slack + e-mail |
| Reusability | effort to wire a *new* model | env-vars + one scrape job, no model code changes (proven with a 2nd model) |

**Scope vs the 15-hour budget.** exporters 5 h · Compose 3 h · dashboards 3 h · Alertmanager 2 h ·
docs/demo 2 h — matched by the repository layout below (`services/`, `monitoring/`, `docs/`).

## 2 · What was built
A **reusable, model-agnostic observability stack** — Prometheus · Grafana · Alertmanager · custom
drift exporter (PSI/KS) · MLflow · Docker Compose — wrapped around the other team's Telco
Customer-Churn model.

📘 **Everything is explained in one document: [`docs/Team4-Monitoring-Guidebook.pdf`](docs/Team4-Monitoring-Guidebook.pdf)**
— concept → why → tools → alternatives →
trade-offs → why chosen → implementation → PromQL/code → demo → troubleshooting → examiner Q&A.

## 3 · Quick start

```bash
make up          # build + start 10 containers (first run: a few minutes)
make train       # ZenML pipeline: validate → split → Optuna (25 trials) → evaluate → register → drift reference (~4 min)
make smoke       # end-to-end check (23 assertions)
```

| UI | URL |
|---|---|
| Grafana (4 dashboards) | http://localhost:3000 — admin / admin |
| Prometheus | http://localhost:9090 |
| Alertmanager | http://localhost:9093 |
| MLflow | http://localhost:5001 |
| Model API (Swagger) | http://localhost:8000/docs |
| Gateway metrics | http://localhost:8080/metrics |
| Drift exporter metrics | http://localhost:9105/metrics |
| Load generator (scenarios) | http://localhost:8090/docs |
| Fake e-mail inbox (Mailhog) | http://localhost:8025 |
| Fake Slack (webhook catcher) | http://localhost:8091 |

Demo scenarios: `make scenario-drift | scenario-latency | scenario-errors | scenario-burst | scenario-bad | scenario-normal`
· `python3 tests/alert_status.py` shows what fired and where it was routed.

## 4 · Architecture

```
load-generator ──► GATEWAY (:8080, mlgw_* metrics) ──► model-api (:8000, other team's churn model)
                      │ async /ingest
                      ▼
               drift-exporter (:9105, PSI + KS on a 500-row rolling window vs training reference)
Prometheus (:9090) scrapes everything · recording rules (ml:*) · 14 alert rules
   ├─► Alertmanager (:9093) ──► Slack (#ml-alerts / #ml-quality / #ml-oncall) + e-mail
   └─► Grafana (:3000) — Ops · Predictions · Drift · Stack-Health dashboards
MLflow (:5001) ◄── trainer — experiment tracking + model registry
```

## 5 · Repository layout

| Path | What |
|---|---|
| `docker-compose.yml` · `docker-compose.second-model.yml` · `Makefile` · `.env.example` | orchestration, overlay for a 2nd model, commands, real Slack/SMTP settings |
| `services/model_api/` | other team's FastAPI churn service + MLflow/skops model, column-order fix, ML metrics, `/chaos` |
| `services/gateway/` | model-agnostic Prometheus exporter/proxy (request rate, latency histogram, errors, prediction distribution) |
| `services/drift_exporter/` | PSI + KS rolling-window exporter, `build_reference.py`, reference profile |
| `services/load_generator/` | traffic + demo scenarios |
| `monitoring/prometheus/` | `prometheus.yml`, `rules/recording_rules.yml`, `rules/alert_rules.yml` |
| `monitoring/alertmanager/` | routes/receivers/inhibition, Slack template, env-substituting entrypoint |
| `monitoring/grafana/` | provisioning + `build_dashboards.py` (dashboards-as-code) → `dashboards/*.json` |
| `services/mlflow/` · `services/trainer/` | MLflow server image; **ZenML pipeline** (`pipeline.py`) with data-validation gate, **Optuna** TPE tuning (25 trials, nested MLflow runs), quality-gated Model-Registry promotion, drift-reference export |
| `services/webhook_catcher/` | fake Slack sink for the demo |
| `tests/` | `test_units.py` (pytest, runs in CI), `smoke_test.py` (end-to-end), `alert_status.py` |
| `.github/workflows/ci.yml` · `render.yaml` | CI: ruff + pytest + promtool/amtool + compose smoke + image builds/push to GHCR; Render blueprint for the model API |
| `docs/` | **`Team4-Monitoring-Guidebook.pdf`** — the guidebook |
| `data/` · `artifacts/` | Telco CSV; `mlflow_runs.csv` |
| `external/customer-churn-mlops/` | the original repository, untouched (used as-is by the second-model overlay) |

## 6 · Wiring the stack to a new model

Point a gateway at it (env vars), add one scrape job, optionally build a drift reference from its
training data — see chapter 10 of the guidebook. `docker-compose.second-model.yml` shows it live with
the other team's *unmodified* image.

## 7 · Training pipeline (ZenML + Optuna + MLflow)

```
ingest_data → validate_data ─╳ halts on missing columns / nulls > 5 % / minority class < 10 %
            → split_data → tune_and_train (Optuna TPE + MedianPruner, 25 trials, 3-fold CV AUC, nested MLflow runs)
            → evaluate_model (hold-out AUC/F1/precision/recall, model logged with signature)
            → register_model (quality gate AUC ≥ 0.80 → Model Registry, alias @champion, tag stage=production)
            → build_drift_reference (writes the PSI/KS reference for the drift exporter)
            → export_runs_csv
```
`make train` runs it in a container against the MLflow server; `make train-quick` uses 8 trials.

## 8 · Deployment

* **Local / demo:** `make up` (Docker Compose, 10 services, health checks).
* **Cloud:** `render.yaml` deploys `services/model_api` to Render as a Docker web service with
  `/health` as health-check path (`CHAOS_ENABLED=0`). Point a gateway's `UPSTREAM_URL` at the Render URL
  to monitor the cloud instance from the local stack.
* **CI/CD:** every push/PR runs lint, unit tests, config validation (promtool/amtool/compose),
  builds all images and (on `main`) pushes them to GHCR; a compose smoke test boots the whole stack.

## 9 · Documentation

* **`docs/Team4-Monitoring-Guidebook.pdf`** — the full guidebook (why each tool, alternatives, PromQL, demo, Q&A).
* **`docs/Team4-Project-Report.pdf`** — the project report (objective · architecture · implementation · deployment · results · challenges · conclusion).
