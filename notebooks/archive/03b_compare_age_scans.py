"""Compare saved Mathematica and Python age-scan CSVs and report relative differences.

Usage: run from repo root or via the project's Python:

  python notebooks/03b_compare_age_scans.py

This script does NOT regenerate any scans; it only loads existing CSVs from
`results/03_calibrate_models/` and compares the Mathematica-derived files
against the Python-derived files added by the Python pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]

MATHEMATICA_PATHS = {
    "main": REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan.csv",
    "q05":  REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_05.csv",
    "q95":  REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_95.csv",
}

PYTHON_PATHS = {
    "main": REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_python.csv",
    "q05":  REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_05_python.csv",
    "q95":  REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_95_python.csv",
}


def load_matrix(p: Path) -> np.ndarray:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p, index_col=0).to_numpy(dtype=float)


def summarize(M: np.ndarray, P: np.ndarray) -> dict:
    rel = np.abs(M - P) / np.maximum(np.abs(M), 1e-12)
    return {
        "shape": list(M.shape),
        "median_rel": float(np.median(rel)),
        "p99_rel": float(np.quantile(rel, 0.99)),
        "max_rel": float(rel.max()),
    }


def main() -> int:
    report = {}
    missing = []
    for k in ("main", "q05", "q95"):
        mpath = MATHEMATICA_PATHS[k]
        ppath = PYTHON_PATHS[k]
        if not mpath.exists() or not ppath.exists():
            missing.append((k, str(mpath), str(ppath)))
            continue
        J = load_matrix(mpath)
        P = load_matrix(ppath)
        if J.shape != P.shape:
            report[k] = {"error": f"shape mismatch: {J.shape} vs {P.shape}"}
            continue
        report[k] = summarize(J, P)

    out_path = REPO / "results/03_calibrate_models/compare_age_scans_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"report": report, "missing": missing}, f, indent=2)

    # Print concise table
    if missing:
        print("Missing files for the following scans (mathematica_path, python_path):")
        for k, m, p in missing:
            print(f"  {k}: {m} , {p}")
    print(f"Wrote JSON report to: {out_path}")

    if report:
        print(f"{'scan':>5} {'shape':>14} {'median rel':>14} {'p99 rel':>14} {'max rel':>14}")
        for k, v in report.items():
            if "error" in v:
                print(f"{k:>5} ERROR: {v['error']}")
                continue
            shape = tuple(v["shape"])
            print(f"{k:>5} {str(shape):>14} {v['median_rel']:14.3e} {v['p99_rel']:14.3e} {v['max_rel']:14.3e}")

    return 0 if not missing else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print("Error:", exc, file=sys.stderr)
        raise SystemExit(1)
