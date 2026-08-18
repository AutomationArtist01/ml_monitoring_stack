# run_pipeline.py — entry point: python run_pipeline.py [--trials N] [--csv PATH]
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
