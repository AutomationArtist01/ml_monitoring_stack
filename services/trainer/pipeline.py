# pipeline.py — ZenML training pipeline: ingest → validate(gate) → split → Optuna tune (MLflow) → evaluate → register → drift reference
from __future__ import annotations

import json
import os
import time
from typing import Annotated, Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from zenml import pipeline, step

# --- Schema + configuration ---
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn"
REQUIRED = NUMERIC + CATEGORICAL + [TARGET]

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "telco-churn")
REGISTERED_NAME = os.environ.get("REGISTERED_NAME", "telco-churn-classifier")
N_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "25"))
CV_FOLDS = int(os.environ.get("CV_FOLDS", "3"))
ART_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")
REFERENCE_OUT = os.environ.get("REFERENCE_OUT", os.path.join(ART_DIR, "telco_reference.json"))
EXPORT_DIR = os.environ.get("EXPORT_DIR", os.path.join(ART_DIR, "champion_model"))


# --- Helpers ---
def make_pipeline(params: dict[str, Any]) -> Pipeline:
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=42, **params))])


def _mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


# --- STEP 1: Ingest ---
@step
def ingest_data(csv_path: str) -> Annotated[pd.DataFrame, "raw_data"]:
    """Load the Telco churn CSV."""
    df = pd.read_csv(csv_path)
    print(f"[ingest] {csv_path}: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


# --- STEP 2: Validate (halts pipeline on failure) ---
@step
def validate_data(df: pd.DataFrame, min_rows: int = 1000, max_null_share: float = 0.05,
                  min_minority_share: float = 0.10) -> Annotated[pd.DataFrame, "validated_data"]:
    """Schema / nulls / class-balance checks. Raises → the pipeline HALTS (rubric: validation gates)."""
    problems: list[str] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if len(df) < min_rows:
        problems.append(f"too few rows: {len(df)} < {min_rows}")

    # TotalCharges arrives as text with blanks for brand-new customers → coerce, then check null share
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    null_share = df[REQUIRED].isna().mean() if not missing else pd.Series(dtype=float)
    bad_nulls = null_share[null_share > max_null_share]
    if len(bad_nulls):
        problems.append(f"null share too high: {bad_nulls.round(3).to_dict()}")

    if TARGET in df.columns:
        if not set(df[TARGET].dropna().unique()) <= {"Yes", "No"}:
            problems.append(f"unexpected target values: {df[TARGET].unique()[:5]}")
        minority = df[TARGET].value_counts(normalize=True).min()
        if minority < min_minority_share:
            problems.append(f"class imbalance too severe: minority share {minority:.3f}")
    for c in NUMERIC:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            problems.append(f"{c} not numeric")
    if "tenure" in df.columns and (df["tenure"] < 0).any():
        problems.append("negative tenure")

    report = {"rows": int(len(df)), "null_share": {k: round(float(v), 4) for k, v in null_share.items() if v > 0},
              "churn_rate": float((df[TARGET] == "Yes").mean()) if TARGET in df.columns else None,
              "problems": problems}
    print("[validate]", json.dumps(report))
    if problems:
        raise ValueError("DATA VALIDATION FAILED – pipeline halted: " + "; ".join(problems))

    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df = df.drop(columns=[c for c in ["customerID"] if c in df.columns])
    return df


# --- STEP 3: Split ---
@step
def split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple[
        Annotated[pd.DataFrame, "X_train"], Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"], Annotated[pd.Series, "y_test"]]:
    """Stratified train/test split; target encoded Yes/No → 1/0."""
    y = (df[TARGET] == "Yes").astype(int)
    X = df[NUMERIC + CATEGORICAL]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    print(f"[split] train {len(Xtr)} / test {len(Xte)} · churn rate {y.mean():.3f}")
    return Xtr, Xte, ytr, yte


# --- STEP 4: Optuna tuning + training (nested MLflow runs) ---
@step(enable_cache=False)
def tune_and_train(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = N_TRIALS) -> tuple[
        Annotated[Pipeline, "model"], Annotated[dict, "best_params"], Annotated[str, "mlflow_run_id"]]:
    """Optuna TPE search (MedianPruner) with CV ROC-AUC; every trial = nested MLflow run."""
    _mlflow()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    with mlflow.start_run(run_name="optuna-gbm-study") as parent:
        mlflow.set_tags({"team": "team4", "stage": "tuning", "sampler": "TPE", "pruner": "Median",
                         "dataset": "telco-churn"})
        mlflow.log_params({"n_trials": n_trials, "cv_folds": CV_FOLDS, "n_train": len(X_train)})

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
            with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
                t0 = time.time()
                scores = []
                for fold, (tr, va) in enumerate(cv.split(X_train, y_train)):
                    pipe = make_pipeline(params).fit(X_train.iloc[tr], y_train.iloc[tr])
                    auc = roc_auc_score(y_train.iloc[va], pipe.predict_proba(X_train.iloc[va])[:, 1])
                    scores.append(auc)
                    trial.report(float(np.mean(scores)), step=fold)   # pruner sees partial CV
                    if trial.should_prune():
                        mlflow.log_params(params)
                        mlflow.log_metrics({"cv_roc_auc": float(np.mean(scores)), "pruned": 1})
                        mlflow.set_tag("state", "pruned")
                        raise optuna.TrialPruned()
                cv_auc = float(np.mean(scores))
                mlflow.log_params(params)
                mlflow.log_metrics({"cv_roc_auc": cv_auc, "cv_roc_auc_std": float(np.std(scores)),
                                    "fit_seconds": time.time() - t0, "pruned": 0})
                mlflow.set_tag("state", "complete")
                return cv_auc

        study = optuna.create_study(direction="maximize", study_name="telco-churn-gbm",
                                    sampler=optuna.samplers.TPESampler(seed=42),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        print(f"[optuna] {len(study.trials)} trials ({pruned} pruned) · best cv AUC {study.best_value:.4f} · {best}")
        mlflow.log_metrics({"best_cv_roc_auc": study.best_value, "trials_pruned": pruned})
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        # param-importance for the "visualise hyper-parameter effects" whiteboard item
        try:
            imp = optuna.importance.get_param_importances(study)
            mlflow.log_dict(imp, "optuna_param_importance.json")
        except Exception:  # noqa: BLE001
            pass
        mlflow.log_dict({"trials": [{"number": t.number, "value": t.value, "state": str(t.state), **t.params}
                                    for t in study.trials]}, "optuna_trials.json")

        # retrain the best config on the whole training split
        model = make_pipeline(best).fit(X_train, y_train)
        return model, best, parent.info.run_id


# --- STEP 5: Evaluate ---
@step
def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, best_params: dict,
                   mlflow_run_id: str) -> Annotated[dict, "test_metrics"]:
    """Hold-out metrics; logged to the parent MLflow run together with the model artifact."""
    _mlflow()
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred)),
        "test_f1": float(f1_score(y_test, pred)),
    }
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_params({f"final_{k}": v for k, v in best_params.items()})
        mlflow.sklearn.log_model(model, name="model", signature=infer_signature(X_test, proba),
                                 input_example=X_test.head(3))
    print("[evaluate]", json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    return metrics


# --- STEP 6: Register (quality gate) ---
@step
def register_model(mlflow_run_id: str, test_metrics: dict, min_auc: float = 0.80) -> Annotated[str, "model_version"]:
    """Promote to the Model Registry only if the quality gate passes; alias `champion`, tag stage=production."""
    _mlflow()
    if test_metrics["test_roc_auc"] < min_auc:
        raise ValueError(f"QUALITY GATE FAILED: test AUC {test_metrics['test_roc_auc']:.4f} < {min_auc}")
    mv = mlflow.register_model(f"runs:/{mlflow_run_id}/model", REGISTERED_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_NAME, "champion", mv.version)
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "stage", "production")
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "test_roc_auc", f"{test_metrics['test_roc_auc']:.4f}")
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "validated_by", "zenml-pipeline")
    print(f"[registry] {REGISTERED_NAME} v{mv.version} → @champion (stage=production)")
    return str(mv.version)


# --- STEP 7: Drift reference ---
@step
def build_drift_reference(model: Pipeline, X_train: pd.DataFrame, model_version: str,
                          out_path: str = REFERENCE_OUT) -> Annotated[str, "reference_path"]:
    """Write the drift-exporter reference profile (feature bins + training-score histogram)."""
    rng = np.random.default_rng(42)
    ref: dict[str, Any] = {"model": REGISTERED_NAME, "model_version": model_version, "rows": int(len(X_train)),
                           "numeric": {}, "categorical": {}}
    for f in NUMERIC:
        v = X_train[f].to_numpy(dtype=float)
        edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            edges = np.array([v.min() - 0.5, v.max() + 0.5])
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(v, bins=edges)
        sample = v if len(v) <= 2000 else rng.choice(v, 2000, replace=False)
        ref["numeric"][f] = {"edges": edges.tolist(), "probs": (counts / counts.sum()).tolist(),
                             "values": np.round(sample, 4).tolist()}
    for f in CATEGORICAL:
        ref["categorical"][f] = {"probs": X_train[f].astype(str).value_counts(normalize=True).to_dict()}
    scores = model.predict_proba(X_train)[:, 1]
    edges = np.array([-np.inf] + [i / 10 for i in range(1, 10)] + [np.inf])
    counts, _ = np.histogram(scores, bins=edges)
    ref["score"] = {"edges": edges.tolist(), "probs": (counts / counts.sum()).tolist()}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(ref, fh)
    print(f"[reference] wrote {out_path}")
    return out_path


# --- STEP 8: Export the champion model for serving ---
@step(enable_cache=False)
def export_champion(model: Pipeline, model_version: str, mlflow_run_id: str, test_metrics: dict,
                    out_dir: str = EXPORT_DIR) -> Annotated[str, "champion_export"]:
    """Write the @champion model as an MLflow sklearn artifact + model_meta.json (consumed by services/model_api)."""
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)
    mlflow.sklearn.save_model(model, out_dir)
    meta = {"registered_name": REGISTERED_NAME, "version": str(model_version), "alias": "champion",
            "run_id": mlflow_run_id, "kind": "sklearn-pipeline", "features": NUMERIC + CATEGORICAL,
            "test_roc_auc": round(float(test_metrics["test_roc_auc"]), 4),
            "mlflow_tracking_uri": TRACKING_URI, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out_dir, "model_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[export] champion v{model_version} → {out_dir} (run {mlflow_run_id})")
    return out_dir


# --- STEP 9: Export runs CSV ---
@step
def export_runs_csv(model_version: str, out_dir: str = ART_DIR) -> Annotated[str, "runs_csv"]:
    """Export ALL MLflow runs of the experiment to CSV (rubric/whiteboard item)."""
    _mlflow()
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["metrics.cv_roc_auc DESC"])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "mlflow_runs.csv")
    runs.to_csv(out, index=False)
    print(f"[csv] {len(runs)} runs → {out}")
    return out


# --- Pipeline definition ---
@pipeline(enable_cache=True)
def churn_training_pipeline(csv_path: str = "data/telco_churn.csv", n_trials: int = N_TRIALS):
    df = ingest_data(csv_path=csv_path)
    df = validate_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model, best_params, run_id = tune_and_train(X_train, y_train, n_trials=n_trials)
    metrics = evaluate_model(model, X_test, y_test, best_params, run_id)
    version = register_model(run_id, metrics)
    build_drift_reference(model, X_train, version)
    export_champion(model, version, run_id, metrics)
    export_runs_csv(version)
