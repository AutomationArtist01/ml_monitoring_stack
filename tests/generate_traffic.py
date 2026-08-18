#!/usr/bin/env python3
"""
Generate RANDOM customers within the real value ranges of the Telco dataset and send them
to the model through the monitoring gateway, so the dashboards/graphs fill up.

    python3 tests/generate_traffic.py                    # 300 random customers at 10 req/s
    python3 tests/generate_traffic.py --n 1000 --rps 20  # more / faster
    python3 tests/generate_traffic.py --drift             # shifted population (young, expensive, month-to-month)
    python3 tests/generate_traffic.py --forever           # keep going until Ctrl-C

How the random values are made (all learned from data/telco_churn.csv, nothing hard-coded):
  * numeric columns  -> uniform random between the column's min and max
                        (TotalCharges is kept consistent: tenure * MonthlyCharges ± noise)
  * categorical cols -> random choice weighted by how often each value appears in the dataset
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import csv
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "telco_churn.csv"
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]


def learn_ranges(rows: list[dict]):
    """min/max per numeric column and value frequencies per categorical column (no pandas needed)."""
    def num(v):
        try:
            return float(v)
        except ValueError:
            return 0.0  # blank TotalCharges for brand-new customers
    ranges = {c: (min(num(r[c]) for r in rows), max(num(r[c]) for r in rows)) for c in NUMERIC}
    cats = {}
    for c in CATEGORICAL:
        cnt = Counter(r[c] for r in rows)
        total = sum(cnt.values())
        cats[c] = {k: v / total for k, v in cnt.items()}
    return ranges, cats


def random_customer(ranges, cats, drift=False) -> dict:
    row = {}
    for c, probs in cats.items():
        row[c] = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
    row["SeniorCitizen"] = random.choice([0, 1]) if not drift else random.choices([0, 1], [0.6, 0.4])[0]
    lo, hi = ranges["tenure"]
    row["tenure"] = int(random.uniform(lo, hi)) if not drift else int(max(lo, min(hi, random.gauss(4, 3))))
    lo, hi = ranges["MonthlyCharges"]
    row["MonthlyCharges"] = round(random.uniform(lo, hi), 2) if not drift else round(random.uniform(hi * 0.7, hi), 2)
    row["TotalCharges"] = round(max(0.0, row["tenure"] * row["MonthlyCharges"] * random.uniform(0.9, 1.1)), 2)
    if drift:  # shifted population: mostly month-to-month, fibre, electronic check
        if random.random() < 0.85:
            row["Contract"] = "Month-to-month"
        if random.random() < 0.7:
            row["InternetService"] = "Fiber optic"
        if random.random() < 0.6:
            row["PaymentMethod"] = "Electronic check"
    return row


def send(url: str, customer: dict, timeout=10):
    req = urllib.request.Request(url, data=json.dumps(customer).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080/predict", help="gateway predict URL (default) or model API directly")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--rps", type=float, default=10)
    ap.add_argument("--drift", action="store_true", help="generate a shifted population to trigger drift alerts")
    ap.add_argument("--forever", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.seed is not None:
        random.seed(a.seed)

    with open(CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    ranges, cats = learn_ranges(rows)
    print("value ranges learned from the dataset:")
    for c, (lo, hi) in ranges.items():
        print(f"  {c:15s} {lo:>10.2f} .. {hi:<10.2f}")
    print(f"  {len(cats)} categorical columns, sampled by their real frequencies")
    print(f"\nsending {'∞' if a.forever else a.n} random customers to {a.url} at {a.rps} req/s"
          f"{' (DRIFTED population)' if a.drift else ''} — watch Grafana: http://localhost:3000/d/ml-ops\n")

    sent = ok = err = 0
    scores = []
    t_start = time.time()
    try:
        while a.forever or sent < a.n:
            t0 = time.perf_counter()
            cust = random_customer(ranges, cats, drift=a.drift)
            try:
                status, body = send(a.url, cust)
                ok += 1
                scores.append(body.get("churn_probability", 0))
                if sent % 25 == 0:
                    print(f"  #{sent:5d}  tenure={cust['tenure']:3d}  monthly={cust['MonthlyCharges']:7.2f}  "
                          f"contract={cust['Contract']:15s} → churn_prob={body.get('churn_probability')}")
            except urllib.error.HTTPError as e:
                err += 1
                if err <= 3:
                    print(f"  HTTP {e.code}: {e.read()[:120]}")
            except Exception as e:  # noqa: BLE001
                err += 1
                if err <= 3:
                    print("  error:", e)
            sent += 1
            time.sleep(max(0.0, 1.0 / a.rps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    dur = time.time() - t_start
    print(f"\ndone: {sent} sent, {ok} ok, {err} errors in {dur:.0f}s "
          f"({sent / max(dur, 1):.1f} req/s); mean churn probability {sum(scores) / max(len(scores), 1):.3f}, "
          f"positive rate {sum(s >= 0.5 for s in scores) / max(len(scores), 1):.1%}")
    print("Grafana: http://localhost:3000/d/ml-ops   http://localhost:3000/d/ml-predictions   http://localhost:3000/d/ml-drift")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
