"""
Entry point:  python run_pipeline.py [--trials N] [--csv PATH]

Env: MLFLOW_TRACKING_URI (http://mlflow:5000) · MLFLOW_EXPERIMENT (telco-churn) · OPTUNA_TRIALS (25)
     ARTIFACT_DIR (artifacts) · REFERENCE_OUT (artifacts/telco_reference.json)
ZenML runs with its default local stack (orchestrator=local, artifact store=local); the run is
recorded in ZenML's SQLite store, so `zenml pipeline runs list` shows history.
"""
import argparse
import os

from pipeline import N_TRIALS, churn_training_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.environ.get("DATA_PATH", "data/telco_churn.csv"))
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    a = ap.parse_args()
    run = churn_training_pipeline(csv_path=a.csv, n_trials=a.trials)
    print(f"[zenml] pipeline run: {run.name} · status={run.status}")
