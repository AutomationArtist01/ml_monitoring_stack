"""
Unit tests (run in CI, no Docker needed):  pytest tests/test_units.py
Covers: PSI/KS maths of the drift exporter, gateway JSON-path helper, drift-reference builder,
data-validation gate of the training pipeline (pure-python parts), and config sanity.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telco_churn.csv"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- drift maths
@pytest.fixture(scope="module")
def drift(monkeypatch_module=None):
    # the exporter loads the reference at import; point it at the shipped one
    import os
    os.environ["REFERENCE_PATH"] = str(ROOT / "services/drift_exporter/reference/telco_reference.json")
    return load("drift_exporter", ROOT / "services/drift_exporter/drift_exporter.py")


def test_psi_identical_is_zero(drift):
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert drift.psi(p, p) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift(drift):
    ref = np.array([0.25, 0.25, 0.25, 0.25])
    small = np.array([0.30, 0.25, 0.25, 0.20])
    big = np.array([0.70, 0.10, 0.10, 0.10])
    assert 0 < drift.psi(ref, small) < 0.1 < 0.25 < drift.psi(ref, big)


def test_psi_handles_zero_bins(drift):
    ref = np.array([0.5, 0.5, 0.0])
    act = np.array([0.0, 0.5, 0.5])
    assert np.isfinite(drift.psi(ref, act))


def test_numeric_hist_probs_sum_to_one(drift):
    edges = [-np.inf, 1, 2, 3, np.inf]
    h = drift.numeric_hist(np.array([0.5, 1.5, 1.7, 2.5, 9.0]), edges)
    assert h.sum() == pytest.approx(1.0) and len(h) == 4


def test_reference_profile_shape(drift):
    assert set(drift.NUMERIC) == {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}
    assert len(drift.CATEGORICAL) == 15
    for f, r in drift.NUMERIC.items():
        assert len(r["edges"]) == len(r["probs"]) + 1
        assert sum(r["probs"]) == pytest.approx(1.0, abs=1e-6)
    assert sum(drift.SCORE_REF["probs"]) == pytest.approx(1.0, abs=1e-6)


def test_ingest_and_compute_detects_drift(drift):
    drift.window.clear()
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    rows = df.drop(columns=["customerID", "Churn"]).sample(400, random_state=1).to_dict(orient="records")
    for r in rows:                                   # baseline: same distribution → low PSI
        drift.ingest({"features": r, "score": 0.2, "prediction": 0})
    drift.compute()
    base = drift.PSI.labels(drift.MODEL, "tenure", "numeric")._value.get()
    assert base < 0.25
    drift.window.clear()
    for r in rows:                                   # shifted tenure → high PSI
        r2 = dict(r); r2["tenure"] = 1
        drift.ingest({"features": r2, "score": 0.9, "prediction": 1})
    drift.compute()
    shifted = drift.PSI.labels(drift.MODEL, "tenure", "numeric")._value.get()
    assert shifted > 0.25 and shifted > base
    assert drift.KS_P.labels(drift.MODEL, "tenure")._value.get() < 0.01


# --------------------------------------------------------------------------- gateway helper
def test_gateway_dotted_get():
    gw = load("gateway", ROOT / "services/gateway/gateway.py")
    d = {"result": {"score": 0.42, "label": "yes"}, "churn_probability": 0.1}
    assert gw._get(d, "result.score") == 0.42
    assert gw._get(d, "churn_probability") == 0.1
    assert gw._get(d, "missing.key") is None
    assert gw._get([1, 2], "a") is None


# --------------------------------------------------------------------------- reference builder
def test_build_reference_cli(tmp_path):
    br = load("build_reference", ROOT / "services/drift_exporter/build_reference.py")
    out = tmp_path / "ref.json"
    sys.argv = ["x", "--csv", str(DATA), "--out", str(out)]
    br.main()
    ref = json.loads(out.read_text())
    assert ref["rows"] == 7043 and "tenure" in ref["numeric"] and "Contract" in ref["categorical"]
    assert ref["numeric"]["tenure"]["edges"][0] == -np.inf or ref["numeric"]["tenure"]["edges"][0] == float("-inf")


# --------------------------------------------------------------------------- pipeline validation gate
@pytest.fixture(scope="module")
def pipeline_mod():
    zenml = pytest.importorskip("zenml")  # noqa: F841 – only run when zenml is installed (trainer env / CI)
    return load("trainer_pipeline", ROOT / "services/trainer/pipeline.py")


def test_validation_passes_on_real_data(pipeline_mod):
    df = pd.read_csv(DATA)
    out = pipeline_mod.validate_data.entrypoint(df)
    assert "customerID" not in out.columns and out["TotalCharges"].isna().sum() == 0


def test_validation_halts_on_missing_column(pipeline_mod):
    df = pd.read_csv(DATA).drop(columns=["tenure"])
    with pytest.raises(ValueError, match="missing columns"):
        pipeline_mod.validate_data.entrypoint(df)


def test_validation_halts_on_class_imbalance(pipeline_mod):
    df = pd.read_csv(DATA)
    df = pd.concat([df[df.Churn == "No"], df[df.Churn == "Yes"].head(50)])
    with pytest.raises(ValueError, match="imbalance"):
        pipeline_mod.validate_data.entrypoint(df)


# --------------------------------------------------------------------------- config sanity
def test_prometheus_scrapes_all_services():
    cfg = (ROOT / "monitoring/prometheus/prometheus.yml").read_text()
    for job in ["gateway", "model-api", "drift-exporter", "system-exporter", "alertmanager", "grafana", "mlflow"]:
        assert f"job_name: {job}" in cfg


def test_alert_rules_have_runbooks_and_severity():
    import yaml
    rules = yaml.safe_load((ROOT / "monitoring/prometheus/rules/alert_rules.yml").read_text())
    n = 0
    for g in rules["groups"]:
        for r in g["rules"]:
            n += 1
            assert r["labels"]["severity"] in {"info", "warning", "critical"}, r["alert"]
            assert "runbook_url" in r["annotations"], r["alert"]
            assert "for" in r, r["alert"]
    assert n >= 18


def test_dashboards_are_valid_json_with_panels():
    for f in (ROOT / "monitoring/grafana/dashboards").glob("*.json"):
        d = json.loads(f.read_text())
        assert d["uid"] and len(d["panels"]) >= 8, f.name
