#!/usr/bin/env python3
"""
End-to-end smoke test for the monitoring stack (run after `make up`, wait ~60s).
Checks: services healthy → prediction through gateway → metrics exposed → Prometheus scraping →
recording rules produce values → drift exporter has a window → Grafana dashboards provisioned →
Alertmanager config loaded → MLflow reachable.
Exit code 0 = all good.
"""
import json
import sys
import urllib.request

OK, FAIL = 0, 0


def get(url, data=None, headers=None, timeout=10):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return r.status, body


def check(name, fn):
    global OK, FAIL
    try:
        detail = fn()
        print(f"  ✔ {name}" + (f"  ({detail})" if detail else ""))
        OK += 1
    except Exception as e:  # noqa: BLE001
        print(f"  ✘ {name}: {e.__class__.__name__}: {e}")
        FAIL += 1


def promq(q):
    st, b = get("http://localhost:9090/api/v1/query?query=" + urllib.parse.quote(q))
    return json.loads(b)["data"]["result"]


import urllib.parse  # noqa: E402

SAMPLE = json.load(open("tests/sample_request.json"))

print("== services")
for name, url in [("model-api", "http://localhost:8000/health"), ("gateway", "http://localhost:8080/health"),
                  ("drift-exporter", "http://localhost:9105/health"), ("load-generator", "http://localhost:8090/status"),
                  ("prometheus", "http://localhost:9090/-/ready"), ("alertmanager", "http://localhost:9093/-/ready"),
                  ("grafana", "http://localhost:3000/api/health"), ("mlflow", "http://localhost:5001/health"),
                  ("mailhog", "http://localhost:8025/api/v2/messages"), ("webhook-catcher", "http://localhost:8091/api")]:
    check(name, lambda u=url: f"HTTP {get(u)[0]}")

print("== prediction path")
def predict():
    st, b = get("http://localhost:8080/predict", data=json.dumps(SAMPLE).encode(),
                headers={"content-type": "application/json"})
    d = json.loads(b)
    assert "churn_probability" in d, d
    return f"churn_probability={d['churn_probability']}"
check("POST /predict via gateway", predict)

def gw_metrics():
    b = get("http://localhost:8080/metrics")[1].decode()
    for m in ["mlgw_requests_total", "mlgw_request_latency_seconds_bucket", "mlgw_prediction_score_bucket", "mlgw_errors_total"]:
        assert m in b, f"missing {m}"
    return "requests/latency/score/errors metrics present"
check("gateway /metrics", gw_metrics)

def api_metrics():
    b = get("http://localhost:8000/metrics")[1].decode()
    for m in ["prediction_requests_total", "prediction_probability_bucket", "http_request_duration_seconds"]:
        assert m in b, f"missing {m}"
    return "in-process metrics present"
check("model-api /metrics", api_metrics)

def drift_metrics():
    b = get("http://localhost:9105/metrics")[1].decode()
    assert "drift_psi" in b and "drift_window_rows" in b
    return "drift_psi present"
check("drift-exporter /metrics", drift_metrics)

print("== prometheus")
def targets():
    r = promq("up")
    down = [x["metric"]["job"] for x in r if x["value"][1] != "1"]
    assert not down, f"targets down: {down}"
    return f"{len(r)} targets up"
check("all scrape targets up", targets)
check("recording rule ml:request_rate:1m", lambda: f"{float(promq('ml:request_rate:1m')[0]['value'][1]):.2f} req/s")
check("recording rule ml:latency_p95:5m", lambda: f"{float(promq('ml:latency_p95:5m')[0]['value'][1])*1000:.1f} ms")
check("recording rule ml:error_ratio:5m", lambda: f"{float(promq('ml:error_ratio:5m')[0]['value'][1]):.3f}")
def rules():
    st, b = get("http://localhost:9090/api/v1/rules")
    groups = json.loads(b)["data"]["groups"]
    n = sum(len(g["rules"]) for g in groups)
    assert n >= 20, n
    return f"{n} rules in {len(groups)} groups"
check("alert + recording rules loaded", rules)
def drift_window():
    r = promq("drift_window_rows")
    rows = float(r[0]["value"][1])
    return f"{int(rows)} rows in window"
check("drift window populated", drift_window)

print("== grafana / alertmanager / mlflow")
def dashboards():
    req = urllib.request.Request("http://localhost:3000/api/search?type=dash-db")
    import base64
    req.add_header("Authorization", "Basic " + base64.b64encode(b"admin:admin").decode())
    with urllib.request.urlopen(req) as r:
        d = json.load(r)
    uids = {x["uid"] for x in d}
    for u in ["ml-ops", "ml-predictions", "ml-drift", "ml-stack"]:
        assert u in uids, f"missing dashboard {u}"
    return f"{len(d)} dashboards"
check("grafana dashboards provisioned", dashboards)
def am():
    st, b = get("http://localhost:9093/api/v2/status")
    d = json.loads(b)
    assert "team4-critical" in d["config"]["original"]
    return "receivers configured"
check("alertmanager config", am)
def mlflow():
    st, b = get("http://localhost:5001/api/2.0/mlflow/experiments/search?max_results=10")
    exps = json.loads(b).get("experiments", [])
    return f"{len(exps)} experiments"
check("mlflow API", mlflow)

print(f"\n{OK} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
