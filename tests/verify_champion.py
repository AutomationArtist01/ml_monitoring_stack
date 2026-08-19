#!/usr/bin/env python3
"""Prove that the deployed model == the MLflow registry @champion.
usage: python3 tests/verify_champion.py [API_URL] [MLFLOW_URL] [REGISTERED_NAME]
       python3 tests/verify_champion.py https://team4-churn-model-api.onrender.com
"""
import json
import sys
import urllib.parse
import urllib.request

api = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
mlf = (sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5001").rstrip("/")
name = sys.argv[3] if len(sys.argv) > 3 else "telco-churn-classifier"

def get(u):
    with urllib.request.urlopen(u, timeout=60) as r:
        return json.load(r)

health = get(api + "/health")
q = urllib.parse.urlencode({"name": name, "alias": "champion"})
mv = get(f"{mlf}/api/2.0/mlflow/registered-models/alias?{q}")["model_version"]
print(f"MLflow registry  : {name} v{mv['version']}  (alias champion, run {mv['run_id']})")
print(f"Deployed {api:<28}: {health.get('model')} v{health.get('version')}  (run {health.get('run_id') or '-'}, source {health.get('source')})")
ok = health.get("model") == name and str(health.get("version")) == str(mv["version"]) and (not health.get("run_id") or health["run_id"] == mv["run_id"])
print("RESULT           :", "✅ deployed model IS the MLflow champion" if ok else "❌ deployed model is NOT the MLflow champion")
sys.exit(0 if ok else 1)
