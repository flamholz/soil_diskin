"""Test + benchmark for `04b_lognormal_predictions_fast_python.py`.

  1. Run the Julia version (`04b_lognormal_predictions_fast_julia.jl`) and time
     it. (Julia script as written is single-threaded.)
  2. Run the Python version twice — once serial (`n_jobs=1`) and once parallel
     (joblib `n_jobs=-1`).
  3. Compare the produced CSVs row-by-row. The 05/95 outputs only have real
     data in the first `n_backfilled` rows; the rest is uninitialized memory
     in both implementations and is excluded from the comparison.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/04_model_predictions"

JULIA_PATHS = {
    "main": RES / "04b_lognormal_cdfs_julia.csv",
    "q05":  RES / "04b_lognormal_cdfs_05_julia.csv",
    "q95":  RES / "04b_lognormal_cdfs_95_julia.csv",
}
PY_PATHS = {
    "main": RES / "04b_lognormal_cdfs_python.csv",
    "q05":  RES / "04b_lognormal_cdfs_05_python.csv",
    "q95":  RES / "04b_lognormal_cdfs_95_python.csv",
}


def run_julia():
    print("=== Running Julia (04b_lognormal_predictions_fast_julia.jl) ===")
    t0 = time.perf_counter()
    out = subprocess.run(
        ["julia", "--project=.", "notebooks/04b_lognormal_predictions_fast_julia.jl"],
        capture_output=True, text=True, cwd=REPO,
    )
    elapsed = time.perf_counter() - t0
    if out.returncode != 0:
        print("STDOUT:\n", out.stdout)
        print("STDERR:\n", out.stderr)
        raise RuntimeError("Julia run failed.")
    print(f"  Julia total wall time: {elapsed:.2f} s")
    return elapsed


def run_python(n_jobs: int):
    label = "serial" if n_jobs == 1 else f"parallel (n_jobs={n_jobs})"
    print(f"=== Running Python {label} ===")
    t0 = time.perf_counter()
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'notebooks'); "
         "from importlib import import_module; "
         f"m = import_module('04b_lognormal_predictions_fast_python'); m.main(n_jobs={n_jobs})"],
        capture_output=True, text=True, cwd=REPO,
    )
    elapsed = time.perf_counter() - t0
    if out.returncode != 0:
        print("STDOUT:\n", out.stdout)
        print("STDERR:\n", out.stderr)
        raise RuntimeError("Python run failed.")
    print(f"  Python {label} total wall time: {elapsed:.2f} s")
    # Surface inner timings printed by the script
    for line in out.stdout.splitlines():
        if "elapsed" in line.lower() or "scanning" in line.lower():
            print(" ", line)
    return elapsed


def compare_csvs():
    # n_backfilled comes from the params file; only those rows are real
    # in the q05/q95 outputs.
    params = pd.read_csv("results/03_calibrate_models/03b_lognormal_predictions_calcurve_julia.csv")
    n_backfilled = int(params["turnover_q05"].notna().sum())

    print("\n=== Numerical agreement (Julia vs Python) ===")
    print(f"{'scan':>6} {'rows compared':>14} {'cols':>6} {'median rel':>14} "
          f"{'p99 rel':>14} {'max rel':>14}")
    for k in ("main", "q05", "q95"):
        J = pd.read_csv(JULIA_PATHS[k]).to_numpy()
        P = pd.read_csv(PY_PATHS[k]).to_numpy()
        assert J.shape == P.shape, f"shape mismatch for {k}: {J.shape} vs {P.shape}"
        rows = J.shape[0] if k == "main" else n_backfilled
        Jr = J[:rows, :]
        Pr = P[:rows, :]
        denom = np.maximum(np.abs(Jr), 1e-12)
        rel = np.abs(Jr - Pr) / denom
        print(f"{k:>6} {rows:>14} {J.shape[1]:>6} "
              f"{np.median(rel):14.3e} {np.quantile(rel, 0.99):14.3e} {rel.max():14.3e}")


def main():
    t_jl = run_julia()
    print()
    t_py_par = run_python(-1)
    print()
    t_py_ser = run_python(1)

    compare_csvs()

    print("\n=== Performance summary ===")
    print(f"  Julia (single-threaded):       {t_jl:.2f} s")
    print(f"  Python (joblib, all cores):    {t_py_par:.2f} s     (Julia/Python = {t_jl/t_py_par:.2f}x)")
    print(f"  Python (serial, n_jobs=1):     {t_py_ser:.2f} s     (Julia/Python = {t_jl/t_py_ser:.2f}x)")


if __name__ == "__main__":
    main()
