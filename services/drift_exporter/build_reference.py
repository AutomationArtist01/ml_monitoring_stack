# build_reference.py — build the drift reference profile (bins, category shares, KS sample, score histogram) from training data
import argparse
import json

import numpy as np
import pandas as pd

# --- Defaults ---
TELCO_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TELCO_CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]


# --- Helpers ---
def quantile_edges(values: np.ndarray, bins: int = 10) -> list[float]:
    qs = np.quantile(values, np.linspace(0, 1, bins + 1))
    edges = np.unique(qs)  # collapse duplicate quantiles (e.g. SeniorCitizen 0/1)
    if len(edges) < 2:
        edges = np.array([values.min() - 0.5, values.max() + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf  # open outer bins so out-of-range values are counted
    return edges.tolist()


# --- CLI ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--numeric", nargs="*", default=TELCO_NUMERIC)
    ap.add_argument("--categorical", nargs="*", default=TELCO_CATEGORICAL)
    ap.add_argument("--score-csv", default=None, help="CSV with a single column of model scores on the training data")
    ap.add_argument("--model", default="CustomerChurnGradientBoosting")
    ap.add_argument("--out", default="reference/telco_reference.json")
    ap.add_argument("--sample", type=int, default=2000)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    rng = np.random.default_rng(42)
    ref = {"model": a.model, "source": a.csv, "rows": int(len(df)), "numeric": {}, "categorical": {}}

    for f in a.numeric:
        v = pd.to_numeric(df[f], errors="coerce").dropna().to_numpy(dtype=float)
        edges = quantile_edges(v)
        counts, _ = np.histogram(v, bins=np.array(edges))
        sample = v if len(v) <= a.sample else rng.choice(v, a.sample, replace=False)
        ref["numeric"][f] = {"edges": edges, "probs": (counts / counts.sum()).tolist(),
                             "values": np.round(sample, 4).tolist()}

    for f in a.categorical:
        vc = df[f].astype(str).value_counts(normalize=True)
        ref["categorical"][f] = {"probs": {k: float(v) for k, v in vc.items()}}

    if a.score_csv:
        s = pd.read_csv(a.score_csv).iloc[:, 0].to_numpy(dtype=float)
        edges = [0.0] + [round(i / 10, 1) for i in range(1, 10)] + [1.0]
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(s, bins=np.array(edges))
        ref["score"] = {"edges": edges, "probs": (counts / counts.sum()).tolist()}

    with open(a.out, "w") as fh:
        json.dump(ref, fh)
    print(f"wrote {a.out}: {len(ref['numeric'])} numeric, {len(ref['categorical'])} categorical, score={'yes' if 'score' in ref else 'no'}")


if __name__ == "__main__":
    main()
