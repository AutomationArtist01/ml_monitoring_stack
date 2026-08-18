"""
Generates the provisioned Grafana dashboards (JSON) – dashboards-as-code.
Run:  python grafana/build_dashboards.py     (writes grafana/dashboards/*.json)
Edit a panel here, re-run, and Grafana picks it up within 10s (provisioning updateIntervalSeconds).
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
OUT.mkdir(exist_ok=True)
DS = {"type": "prometheus", "uid": "prometheus"}
_id = [0]


def nid():
    _id[0] += 1
    return _id[0]


def target(expr, legend="", instant=False, fmt="time_series"):
    t = {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": chr(64 + (nid() % 26 or 26))}
    if instant:
        t["instant"] = True
    if fmt != "time_series":
        t["format"] = fmt
    return t


def panel(title, ptype, targets, x, y, w, h, unit=None, desc="", thresholds=None, opts=None, overrides=None,
          decimals=None, min_=None, max_=None, color=None):
    for i, t in enumerate(targets):          # refIds A, B, C… per panel (Grafana names table columns after them)
        t["refId"] = chr(65 + i)
    p = {
        "id": nid(), "type": ptype, "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
        "fieldConfig": {"defaults": {}, "overrides": overrides or []},
        "options": opts or {},
    }
    d = p["fieldConfig"]["defaults"]
    if unit:
        d["unit"] = unit
    if decimals is not None:
        d["decimals"] = decimals
    if min_ is not None:
        d["min"] = min_
    if max_ is not None:
        d["max"] = max_
    if color:
        d["color"] = color
    if thresholds:
        d["thresholds"] = {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in thresholds]}
    if ptype == "timeseries":
        d.setdefault("custom", {"lineWidth": 2, "fillOpacity": 12, "showPoints": "never"})
        p["options"] = {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"},
                        **(opts or {})}
    if ptype == "stat":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area",
                        "textMode": "auto", **(opts or {})}
    if ptype == "gauge":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"]}, "showThresholdMarkers": True, **(opts or {})}
    if ptype == "bargauge":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"]}, "orientation": "horizontal",
                        "displayMode": "gradient", "showUnfilled": True, **(opts or {})}
    return p


def row(title, y):
    return {"id": nid(), "type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
            "panels": []}


def dashboard(uid, title, panels, tags, desc, refresh="5s", time_from="now-30m", templating=None, links=None):
    return {
        "uid": uid, "title": title, "description": desc, "tags": tags, "timezone": "browser",
        "schemaVersion": 39, "version": 1, "editable": True, "graphTooltip": 1,
        "refresh": refresh, "time": {"from": time_from, "to": "now"},
        "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "1h"]},
        "templating": {"list": templating or []},
        "annotations": {"list": [
            {"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"}, "enable": True,
             "hide": True, "iconColor": "rgba(0, 211, 255, 1)", "name": "Annotations & Alerts", "type": "dashboard"},
            {"datasource": DS, "enable": True, "name": "Critical alerts", "iconColor": "red",
             "expr": "ALERTS{alertstate=\"firing\",severity=\"critical\"}", "step": "5s", "titleFormat": "{{alertname}}",
             "textFormat": "{{model}} {{feature}}"},
        ]},
        "links": links or [
            {"title": "Model API Ops", "type": "link", "url": "/d/ml-ops"},
            {"title": "Predictions", "type": "link", "url": "/d/ml-predictions"},
            {"title": "Data Drift", "type": "link", "url": "/d/ml-drift"},
            {"title": "Stack Health", "type": "link", "url": "/d/ml-stack"},
            {"title": "Prometheus", "type": "link", "url": "http://localhost:9090", "targetBlank": True},
            {"title": "Alertmanager", "type": "link", "url": "http://localhost:9093", "targetBlank": True},
            {"title": "MLflow", "type": "link", "url": "http://localhost:5001", "targetBlank": True},
        ],
        "panels": panels,
    }


MODEL_VAR = [{"name": "model", "label": "Model", "type": "query", "datasource": DS, "refresh": 2,
              "query": "label_values(mlgw_requests_total, model)", "definition": "label_values(mlgw_requests_total, model)",
              "includeAll": True, "multi": False, "current": {"text": "All", "value": "$__all"}, "sort": 1}]
M = 'model=~"$model"'

# ============================================================================ 1. Model API Ops
p = []
y = 0
p.append(row("Golden signals – traffic · latency · errors · saturation", y)); y += 1
p += [
    panel("Request rate", "stat", [target(f'sum(rate(mlgw_requests_total{{{M}}}[1m]))', "req/s")], 0, y, 4, 4, "reqps",
          "sum(rate(mlgw_requests_total[1m])) – requests per second over the last minute", [(None, "blue")], decimals=2),
    panel("p95 latency", "stat", [target(f'ml:latency_p95:5m{{{M}}}', "p95")], 4, y, 4, 4, "s",
          "histogram_quantile(0.95, …) over 5m", [(None, "green"), (0.3, "orange"), (0.5, "red")], decimals=3),
    panel("Error ratio (5m)", "stat", [target(f'ml:error_ratio:5m{{{M}}}', "errors")], 8, y, 4, 4, "percentunit",
          "5xx / all requests", [(None, "green"), (0.01, "orange"), (0.05, "red")], decimals=2),
    panel("Availability (30m)", "gauge", [target(f'ml:availability:30m{{{M}}}', "avail")], 12, y, 4, 4, "percentunit",
          "1 – error ratio over 30 minutes (SLO-style)", [(None, "red"), (0.95, "orange"), (0.99, "green")], decimals=3,
          min_=0.9, max_=1),
    panel("Upstream model up", "stat", [target(f'mlgw_upstream_up{{{M}}}', "up")], 16, y, 4, 4, None,
          "gateway /health probe of the model", [(None, "red"), (1, "green")],
          opts={"textMode": "value"}, overrides=[{"matcher": {"id": "byName", "options": "up"},
                                                  "properties": [{"id": "mappings", "value": [
                                                      {"type": "value", "options": {"0": {"text": "DOWN"}, "1": {"text": "UP"}}}]}]}]),
    panel("In-flight", "stat", [target(f'sum(mlgw_inflight_requests{{{M}}})', "inflight")], 20, y, 4, 4, None,
          "concurrent requests inside the gateway (saturation)", [(None, "blue")]),
]
y += 4
p += [
    panel("Request rate by status", "timeseries",
          [target(f'sum by (status) (rate(mlgw_requests_total{{{M}}}[1m]))', "{{status}}")], 0, y, 12, 8, "reqps",
          "rate() = per-second average increase of a counter over the window"),
    panel("Latency p50 / p95 / p99 / avg", "timeseries",
          [target(f'ml:latency_p50:5m{{{M}}}', "p50"), target(f'ml:latency_p95:5m{{{M}}}', "p95"),
           target(f'ml:latency_p99:5m{{{M}}}', "p99"), target(f'ml:latency_avg:5m{{{M}}}', "avg")],
          12, y, 12, 8, "s", "Percentiles from the latency histogram. Note how p99 » avg under tail latency.",
          thresholds=[(None, "transparent"), (0.5, "red")],
          overrides=[{"matcher": {"id": "byName", "options": "p99"}, "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": "red"}}]},
                     {"matcher": {"id": "byName", "options": "avg"}, "properties": [{"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [10, 10]}}]}],
          opts={}),
]
y += 8
p += [
    panel("Error rate (5xx/s) & ratio", "timeseries",
          [target(f'ml:error_rate:1m{{{M}}}', "5xx / s"), target(f'ml:error_ratio:5m{{{M}}}', "ratio")], 0, y, 12, 7,
          "short", "sum(rate(mlgw_requests_total{status=~\"5..\"}[1m]))",
          overrides=[{"matcher": {"id": "byName", "options": "ratio"}, "properties": [{"id": "unit", "value": "percentunit"},
                                                                                       {"id": "custom.axisPlacement", "value": "right"}]}]),
    panel("Errors by reason", "timeseries",
          [target(f'sum by (reason) (rate(mlgw_errors_total{{{M}}}[1m]))', "{{reason}}")], 12, y, 12, 7, "reqps",
          "upstream_5xx / client_4xx / timeout / connection", opts={"stacking": {"mode": "normal"}}),
]
y += 7
p.append(row("Latency distribution (histogram buckets)", y)); y += 1
p += [
    panel("Latency heatmap", "heatmap",
          [{"datasource": DS, "expr": f'sum by (le) (increase(mlgw_request_latency_seconds_bucket{{{M}}}[$__rate_interval]))',
            "format": "heatmap", "legendFormat": "{{le}}", "refId": "A"}], 0, y, 12, 8, "s",
          "Each column = one scrape window, colour = number of requests in that latency bucket",
          opts={"calculate": False, "yAxis": {"unit": "s"}, "color": {"mode": "scheme", "scheme": "Spectral", "reverse": True},
                "cellGap": 1, "legend": {"show": True}}),
    panel("Requests per latency bucket (last 5m)", "bargauge",
          [target(f'sum by (le) (increase(mlgw_request_latency_seconds_bucket{{{M}}}[5m]))', "≤ {{le}}s", instant=True)],
          12, y, 12, 8, "short", "cumulative buckets: le=0.1 counts everything ≤ 100ms",
          thresholds=[(None, "blue")]),
]
y += 8
p.append(row("Demo controls state (load generator / chaos)", y)); y += 1
p += [
    panel("Active scenario", "stat", [target('loadgen_scenario == 1', "{{scenario}}", instant=True)], 0, y, 6, 4, None,
          "Which load-generator scenario is running", [(None, "purple")], opts={"textMode": "name"}),
    panel("Injected latency (ms)", "stat", [target('chaos_injected_latency_ms', "ms")], 6, y, 6, 4, "ms",
          "chaos endpoint on model_api", [(None, "green"), (1, "red")]),
    panel("Injected error rate", "stat", [target('chaos_injected_error_rate', "rate")], 12, y, 6, 4, "percentunit",
          "chaos endpoint on model_api", [(None, "green"), (0.01, "red")]),
    panel("Target RPS", "stat", [target('loadgen_target_rps', "rps")], 18, y, 6, 4, "reqps", "", [(None, "blue")]),
]
d1 = dashboard("ml-ops", "1 · Model API – Operational (Golden Signals)", p, ["ml", "ops", "team4"],
               "Traffic, latency percentiles, errors, saturation for any model behind the gateway.", templating=MODEL_VAR)

# ============================================================================ 2. Predictions
p = []
y = 0
p.append(row("Prediction distribution", y)); y += 1
p += [
    panel("Mean score (5m)", "stat", [target(f'ml:score_mean:5m{{{M}}}', "mean")], 0, y, 4, 4, None,
          "sum(rate(score_sum))/sum(rate(score_count)); training-time mean ≈ 0.25", [(None, "green"), (0.4, "orange"), (0.55, "red")], decimals=3),
    panel("Positive (churn=1) rate", "gauge", [target(f'ml:positive_rate:5m{{{M}}}', "pos")], 4, y, 4, 4, "percentunit",
          "share of predictions with class 1; training data ≈ 26% churners", [(None, "green"), (0.4, "orange"), (0.6, "red")], min_=0, max_=1),
    panel("Prediction PSI (score drift)", "stat", [target(f'drift_prediction_psi{{{M}}}', "psi")], 8, y, 4, 4, None,
          "PSI of live score histogram vs training-time score histogram", [(None, "green"), (0.1, "orange"), (0.25, "red")], decimals=3),
    panel("Predictions / s by class", "timeseries",
          [target(f'sum by (predicted_class) (rate(mlgw_predictions_total{{{M}}}[1m]))', "class {{predicted_class}}")],
          12, y, 12, 8, "reqps", "", opts={"stacking": {"mode": "normal"}}),
    panel("Last score", "stat", [target(f'mlgw_last_score{{{M}}}', "last")], 0, y + 4, 4, 4, None, "most recent prediction", [(None, "blue")], decimals=3),
    panel("Total predictions", "stat", [target(f'sum(mlgw_predictions_total{{{M}}})', "total")], 4, y + 4, 4, 4, "short", "", [(None, "blue")]),
    panel("Positive rate over time", "timeseries", [target(f'ml:positive_rate:5m{{{M}}}', "positive rate")], 8, y + 4, 4, 4, "percentunit", "", min_=0, max_=1),
]
y += 8
p += [
    panel("Score histogram (last 5m)", "bargauge",
          [target(f'sum by (le) (increase(mlgw_prediction_score_bucket{{{M}}}[5m]))', "≤ {{le}}", instant=True)],
          0, y, 12, 9, "short", "Cumulative histogram buckets of churn probability. Compare with the reference shape.",
          thresholds=[(None, "green")]),
    panel("Score heatmap over time", "heatmap",
          [{"datasource": DS, "expr": f'sum by (le) (increase(mlgw_prediction_score_bucket{{{M}}}[$__rate_interval]))',
            "format": "heatmap", "legendFormat": "{{le}}", "refId": "A"}], 12, y, 12, 9, None,
          "How the score distribution moves over time – drift shows as the hot band moving up",
          opts={"calculate": False, "color": {"mode": "scheme", "scheme": "Oranges"}, "cellGap": 1}),
]
y += 9
p.append(row("In-process metrics of the sample model (library-style instrumentation)", y)); y += 1
p += [
    panel("model_api prediction_requests_total rate", "timeseries",
          [target('sum by (model, version) (rate(prediction_requests_total[1m]))', "{{model}} v{{version}}")], 0, y, 8, 7, "reqps",
          "same signal measured *inside* the model container – should match the gateway"),
    panel("model_api errors by reason", "timeseries",
          [target('sum by (reason) (rate(prediction_errors_total[1m]))', "{{reason}}")], 8, y, 8, 7, "reqps", ""),
    panel("model_api p95 (in-process)", "timeseries",
          [target('histogram_quantile(0.95, sum by (le) (rate(prediction_request_latency_seconds_bucket[5m])))', "p95 in-process"),
           target(f'ml:latency_p95:5m{{{M}}}', "p95 gateway")], 16, y, 8, 7, "s",
          "gateway p95 ≥ in-process p95 (adds network + serialization) – the difference is proxy overhead"),
]
d2 = dashboard("ml-predictions", "2 · Predictions – Distribution & Model Behaviour", p, ["ml", "predictions", "team4"],
               "Is the model still predicting like it did at training time?", templating=MODEL_VAR)

# ============================================================================ 3. Drift
p = []
y = 0
p.append(row("Data drift – rolling window vs training reference (PSI / KS)", y)); y += 1
p += [
    panel("Features drifted (PSI > 0.25)", "stat", [target(f'drift_features_drifted{{{M}}}', "drifted")], 0, y, 4, 4, None,
          "", [(None, "green"), (1, "orange"), (3, "red")]),
    panel("Window rows", "stat", [target(f'drift_window_rows{{{M}}}', "rows")], 4, y, 4, 4, "short", "rows in the rolling window", [(None, "blue")]),
    panel("Prediction PSI", "stat", [target(f'drift_prediction_psi{{{M}}}', "psi")], 8, y, 4, 4, None, "", [(None, "green"), (0.1, "orange"), (0.25, "red")], decimals=3),
    panel("Seconds since last compute", "stat", [target(f'time() - drift_last_compute_timestamp{{{M}}}', "age")], 12, y, 4, 4, "s", "", [(None, "green"), (60, "orange"), (120, "red")], decimals=0),
    panel("Compute time", "stat", [target(f'drift_compute_seconds{{{M}}}', "s")], 16, y, 4, 4, "s", "", [(None, "blue")], decimals=3),
    panel("Rows ingested / s", "stat", [target(f'rate(drift_rows_ingested_total{{{M}}}[1m])', "rows/s")], 20, y, 4, 4, "short", "", [(None, "blue")], decimals=2),
]
y += 4
p += [
    panel("PSI per feature (now)", "bargauge",
          [target(f'sort_desc(drift_psi{{{M}}})', "{{feature}}", instant=True)], 0, y, 12, 12, None,
          "PSI < 0.10 stable · 0.10–0.25 moderate · > 0.25 significant",
          thresholds=[(None, "green"), (0.10, "orange"), (0.25, "red")], min_=0, decimals=3),
    panel("PSI over time (top features)", "timeseries",
          [target(f'topk(8, drift_psi{{{M}}})', "{{feature}}")], 12, y, 12, 12, None,
          "watch the lines cross the 0.25 threshold when the drift scenario is switched on",
          thresholds=[(None, "transparent"), (0.10, "orange"), (0.25, "red")],
          opts={}, overrides=[]),
]
p[-1]["fieldConfig"]["defaults"]["custom"] = {"lineWidth": 2, "fillOpacity": 0, "showPoints": "never",
                                              "thresholdsStyle": {"mode": "line+area"}}
y += 12
p += [
    panel("KS statistic (numeric features)", "timeseries",
          [target(f'drift_ks_statistic{{{M}}}', "{{feature}}")], 0, y, 8, 8, None,
          "Two-sample Kolmogorov–Smirnov D: max distance between the empirical CDFs (0 = identical)", min_=0, max_=1),
    panel("KS p-value (log)", "timeseries",
          [target(f'clamp_min(drift_ks_pvalue{{{M}}}, 1e-6)', "{{feature}}")], 8, y, 8, 8, None,
          "p < 0.05 → reject 'same distribution'. Log scale (values clamped at 1e-6 so p=0 is drawable).", thresholds=[(None, "transparent"), (0.05, "green")]),
    panel("Numeric feature means: window vs reference", "timeseries",
          [target(f'drift_window_mean{{{M}}}', "{{feature}} (live)"),
           target(f'drift_reference_mean{{{M}}}', "{{feature}} (ref)")], 16, y, 8, 8, None,
          "a quick intuition check: are the live means moving away from training-time means?"),
]
p[-2]["fieldConfig"]["defaults"]["custom"] = {"lineWidth": 2, "fillOpacity": 0, "scaleDistribution": {"type": "log", "log": 10}}
p[-1]["fieldConfig"]["overrides"] = [{"matcher": {"id": "byRegexp", "options": ".*\\(ref\\)"},
                                      "properties": [{"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [6, 6]}}]}]
y += 8
p += [
    panel("Drift table (instant)", "table",
          [target(f'sum by (model, feature) (drift_psi{{{M}}})', "", instant=True, fmt="table"),
           target(f'sum by (model, feature) (drift_ks_statistic{{{M}}})', "", instant=True, fmt="table"),
           target(f'sum by (model, feature) (drift_ks_pvalue{{{M}}})', "", instant=True, fmt="table")], 0, y, 24, 9, None,
          "Same data as a table – Prometheus 'graph vs table' point of the demo",
          thresholds=[(None, "green"), (0.10, "orange"), (0.25, "red")],
          opts={"sortBy": [{"displayName": "PSI", "desc": True}]}),
]
p[-1]["transformations"] = [
    {"id": "merge", "options": {}},
    {"id": "organize", "options": {"excludeByName": {"Time": True, "__name__": True, "instance": True, "job": True, "service": True},
                                   "renameByName": {"Value #A": "PSI", "Value #B": "KS D", "Value #C": "KS p"}}},
]
p[-1]["fieldConfig"]["overrides"] = [{"matcher": {"id": "byName", "options": "PSI"},
                                      "properties": [{"id": "custom.cellOptions", "value": {"type": "color-background"}}, {"id": "decimals", "value": 3}]},
                                     {"matcher": {"id": "byName", "options": "KS p"}, "properties": [{"id": "decimals", "value": 4}]},
                                     {"matcher": {"id": "byName", "options": "KS D"}, "properties": [{"id": "decimals", "value": 3}]}]
d3 = dashboard("ml-drift", "3 · Data Drift – PSI & KS on a Rolling Window", p, ["ml", "drift", "team4"],
               "Feature and prediction drift computed by the custom drift exporter.", templating=MODEL_VAR)

# ============================================================================ 4. Stack health
p = []
y = 0
p.append(row("Prometheus targets & alerts", y)); y += 1
p += [
    panel("Targets UP", "stat", [target('up', "{{job}}")], 0, y, 12, 5, None, "1 = scrape OK",
          [(None, "red"), (1, "green")], opts={"textMode": "value_and_name", "colorMode": "background"},
          overrides=[{"matcher": {"id": "byType", "options": "number"}, "properties": [{"id": "mappings", "value": [
              {"type": "value", "options": {"0": {"text": "DOWN"}, "1": {"text": "UP"}}}]}]}]),
    panel("Firing alerts", "stat", [target('count(ALERTS{alertstate="firing"}) or vector(0)', "firing")], 12, y, 4, 5, None, "",
          [(None, "green"), (1, "red")]),
    panel("Pending alerts", "stat", [target('count(ALERTS{alertstate="pending"}) or vector(0)', "pending")], 16, y, 4, 5, None, "",
          [(None, "green"), (1, "orange")]),
    panel("Scrape duration (max)", "stat", [target('max(scrape_duration_seconds)', "s")], 20, y, 4, 5, "s", "", [(None, "green"), (1, "red")], decimals=3),
]
y += 5
p += [
    panel("Alerts by name (firing/pending)", "table",
          [target('ALERTS', "", instant=True, fmt="table")], 0, y, 24, 8, None, "",
          opts={"sortBy": [{"displayName": "alertstate", "desc": False}]}),
]
p[-1]["transformations"] = [{"id": "organize", "options": {"excludeByName": {"Time": True, "Value": True, "__name__": True}}}]
y += 8
p += [
    panel("Prometheus samples ingested / s", "timeseries", [target('rate(prometheus_tsdb_head_samples_appended_total[1m])', "samples/s")], 0, y, 8, 7, "short", ""),
    panel("Prometheus TSDB head series", "timeseries", [target('prometheus_tsdb_head_series', "series")], 8, y, 8, 7, "short", ""),
    panel("Alertmanager notifications / s", "timeseries",
          [target('sum by (integration) (rate(alertmanager_notifications_total[5m]))', "{{integration}}"),
           target('sum by (integration) (rate(alertmanager_notifications_failed_total[5m]))', "{{integration}} FAILED")], 16, y, 8, 7, "short", ""),
]
y += 7
p += [
    panel("Scrape duration by job", "timeseries", [target('scrape_duration_seconds', "{{job}}")], 0, y, 12, 7, "s", ""),
    panel("Container process CPU (all exporters)", "timeseries",
          [target('sum by (job) (rate(process_cpu_seconds_total[1m]))', "{{job}}")], 12, y, 12, 7, "percentunit", "process_cpu_seconds_total is exposed by every prometheus_client app"),
]
d4 = dashboard("ml-stack", "4 · Stack Health – Prometheus / Alertmanager / Targets", p, ["stack", "team4"],
               "Meta-monitoring: is the monitoring itself healthy?", refresh="10s")

for d in (d1, d2, d3, d4):
    (OUT / f"{d['uid']}.json").write_text(json.dumps(d, indent=1))
    print("wrote", OUT / f"{d['uid']}.json", "-", len(d["panels"]), "panels")
