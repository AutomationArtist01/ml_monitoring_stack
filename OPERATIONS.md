# Team Operations Guide — Real-Time Model Monitoring & Alerting Stack

*For whoever is on duty. Everything here is a copy-paste command. Docker Desktop must be running.*

---

## 1 · Start / stop

| Task | Command |
|---|---|
| Start the stack (images already built) | `make start` |
| Start after a code change (rebuild) | `make up` |
| Stop (keeps data) | `make down` |
| Stop and wipe all data | `make nuke` |
| Health check of everything | `make smoke` (26 checks) |
| Status one-liner | `make status` |

After `make start`, wait ~30 s, then open:

| UI | URL |
|---|---|
| Grafana (dashboards) | http://localhost:3000 — admin / admin |
| Prometheus (metrics, alerts) | http://localhost:9090 |
| Alertmanager (who got paged) | http://localhost:9093 |
| MLflow (experiments, registry) | http://localhost:5001 |
| ZenML (pipeline runs) | http://localhost:8237 |
| Model API (Swagger) | http://localhost:8000/docs |
| Cloud copy of the model API | https://team4-churn-model-api.onrender.com/health |

---

## 2 · Add your e-mail (and Slack) for alerts  ← DO THIS ONCE PER MACHINE

Alerts are routed by **Alertmanager**. By default they go to a *fake* inbox (http://localhost:8025) so the stack
works offline. To receive real e-mails, each operator fills in a private `.env` file — **never committed to GitHub**.

### Step 1 — create the file
```bash
cp .env.example .env
```
(If `.env` already exists, just open it.)

### Step 2 — edit these lines
```bash
open -e .env
```
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_TLS=true
SMTP_USER=your.gmail@gmail.com                 # the Gmail that SENDS the alerts
SMTP_PASSWORD=abcdefghijklmnop                 # Google App Password – 16 chars, NO spaces
SMTP_FROM=your.gmail@gmail.com
ALERT_EMAIL_TO=rohith@x.com,vijeta@x.com,abhilash@x.com   # who RECEIVES – comma-separated, no spaces
```

**Getting a Google App Password:** Google Account → *Security* → turn on *2-Step Verification* → *App passwords*
→ create one named `alertmanager` → copy the 16 characters into `SMTP_PASSWORD`.

**Slack (optional):** Slack → *Apps → Incoming Webhooks → Add New Webhook to Workspace* → paste the URL:
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T000/B000/XXXX
```
Leave it as-is to keep the fake-Slack page (http://localhost:8091).

### Step 3 — restart Alertmanager
```bash
make start
```
(Compose reads `.env` automatically; `docker compose restart alertmanager` also works.)

### Step 4 — test it
```bash
make scenario-drift
```
Within ~3 minutes you receive: `[ML-QUALITY FIRING] FeatureDriftHigh …`, `PredictionDrift …` and
`[CRITICAL] MultipleFeaturesDrifted`. Then:
```bash
make scenario-normal
```
→ the matching `RESOLVED` e-mails arrive a couple of minutes later.

**Not arriving?** `docker compose logs alertmanager | grep -i smtp` — usually a wrong/space-containing app password.

### Who gets what

| Alert type | Severity | Slack channel | E-mail |
|---|---|---|---|
| ModelServiceDown, UpstreamModelUnreachable, HighErrorRate, HighLatencyP99Critical, HostDiskAlmostFull, MultipleFeaturesDrifted | critical | `#ml-oncall` | ✅ |
| FeatureDriftHigh/KS/Moderate, PredictionDrift, PositiveRateAnomaly, DriftExporterStale | ML-quality | `#ml-quality` | ✅ |
| HighLatencyP95, TrafficSpike, NoTraffic, HostHighCPU/Memory, ContainerHighMemory | warning | `#ml-alerts` | — |

---

## 3 · Demo / drill scenarios

```bash
make scenario-latency     # inject 800 ms  → HighLatencyP95
make scenario-errors      # inject 30 % 5xx → HighErrorRate (critical → e-mail)
make scenario-drift       # shift the population → drift alerts (→ e-mail)
make scenario-burst       # 40 req/s → TrafficSpike
make scenario-normal      # back to normal → RESOLVED notifications
python3 tests/alert_status.py   # what is firing, where it was routed, what Slack/e-mail received
```

---

## 4 · When an alert arrives — what to do

| Alert | Meaning | First checks | Action |
|---|---|---|---|
| **ModelServiceDown / UpstreamModelUnreachable** | model container not responding | `docker compose ps`, `docker compose logs model-api --tail=50` | `docker compose restart model-api`; if it crash-loops → code/artefact problem, roll back (`git stash` / last good commit) and `make up` |
| **HighLatencyP95 / P99** | slow predictions | Grafana *1 · Model API*: is traffic up? in-process p95 vs gateway p95 | if demo: `make scenario-normal`; real: scale replicas / check model size |
| **HighErrorRate** | > 5 % 5xx | `mlgw_errors_total{reason}` panel; model logs | fix payload / model; `make scenario-normal` if demo |
| **FeatureDriftHigh / KS** | a feature's live distribution ≠ training | Grafana *3 · Data Drift*: which feature; live vs reference means | confirm with data owner → retrain (`make train`) → rebuild reference; if upstream bug → fix and `curl -X POST localhost:9105/reset` |
| **MultipleFeaturesDrifted** (critical) | ≥ 3 features drifted | same + prediction PSI | consider rollback to previous registry version / retrain |
| **PredictionDrift / PositiveRateAnomaly** | outputs shifted but inputs did not | Grafana *2 · Predictions* | serving bug (feature order, wrong version) → check `model_info`, redeploy |
| **HostHighCPU / HostHighMemory / HostDiskAlmostFull** | machine saturated | Grafana *5 · Hardware* (which container) | stop extra containers, free disk (`docker system prune`), resize Docker Desktop resources |
| **NoTraffic** | callers stopped | is the load generator / client running? | `docker compose ps load-generator` |

Silence an alert during maintenance: Alertmanager → *New silence* → matcher `alertname=...`.

---

## 5 · Retrain / change the model

```bash
make train            # ZenML: validate → Optuna (25 trials) → evaluate → register @champion → drift reference
```
Then copy `artifacts/telco_reference.json` to `services/drift_exporter/reference/` and `make up` so the drift
exporter uses the new reference. Results: MLflow http://localhost:5001, run history http://localhost:8237.

## 6 · Monitor another model

Add a gateway with `UPSTREAM_URL` / `PREDICT_PATH` / `MODEL_NAME` / `PROB_FIELD` / `LABEL_FIELD`, add one
`job_name` to `monitoring/prometheus/prometheus.yml`, `make reload-prom`. Example: `docker-compose.second-model.yml`.

## 7 · Keep GitHub clean

`.env` (your e-mail/password), `docs/` (PDFs), `artifacts/` are git-ignored — never commit them.
Every push to `main` runs CI and redeploys Render automatically.
