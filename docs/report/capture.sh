#!/usr/bin/env bash
# Capture report screenshots from the running stack (Grafana anonymous viewer is enabled).
set -euo pipefail; cd "$(dirname "$0")"; mkdir -p screenshots
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
shot() { "$CHROME" --headless=new --disable-gpu --no-sandbox --hide-scrollbars --window-size=1600,${3:-1200} \
  --virtual-time-budget=15000 --screenshot="screenshots/$1.png" "$2" >/dev/null 2>&1 && echo "  $1.png" || echo "  FAILED $1"; }
G="http://localhost:3000/d"; Q="orgId=1&kiosk&from=now-15m&to=now"
shot grafana-ops         "$G/ml-ops?$Q" 1500
shot grafana-drift       "$G/ml-drift?$Q" 1500
shot grafana-predictions "$G/ml-predictions?$Q" 1300
shot alertmanager        "http://localhost:9093/#/alerts" 900
shot notifications       "http://localhost:8091/" 1100
shot mlflow-runs         "http://localhost:5001/#/experiments/1/runs" 1100
shot mlflow-compare      "http://localhost:5001/#/experiments/1/runs?compareRunsMode=CHART" 1100
shot mlflow-registry     "http://localhost:5001/#/models/telco-churn-classifier" 900
