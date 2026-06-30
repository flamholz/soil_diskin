"""Python port of `03b_lognormal_age_scan.jl`.

Produces the same three CSVs (matrix-shaped, no header):

  results/03_calibrate_models/03b_lognormal_model_age_scan_python.csv      (turnover)
  results/03_calibrate_models/03b_lognormal_model_age_scan_05_python.csv   (turnover_q05)
  results/03_calibrate_models/03b_lognormal_model_age_scan_95_python.csv   (turnover_q95)

`_python` suffix avoids overwriting Julia/Wolfram outputs already in place.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Allow running this file directly (`uv run .../03b_lognormal_age_scan.py`):
# add the repo root to sys.path so the `notebooks` package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from notebooks.experimental.lognormal_radiocarbon import load_atm14c, scan_ages

SITES_PATH = "results/all_sites_14C_turnover.csv"
OUT_DIR = Path("results/03_calibrate_models")
AGELIST = np.logspace(3.0, 5.5, 101)


def scan_all(atm, taus, agelist, n_jobs: int = -1, rtol: float = 1e-4) -> np.ndarray:
    rows = Parallel(n_jobs=n_jobs)(
        delayed(scan_ages)(atm, float(t), agelist, rtol) for t in taus
    )
    return np.vstack(rows)


def write_matrix_csv(path: Path, mat: np.ndarray) -> None:
    np.savetxt(path, mat, delimiter=",")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atm = load_atm14c("data/14C_atm_annot.csv")
    print(f"atm14C loaded: {len(atm.ages)} non-negative-age knots, mean_R = {atm.mean_R:.6f}")

    sites = pd.read_csv(SITES_PATH)
    print(f"Sites: {len(sites)} rows")

    n_jobs = int(os.environ.get("N_JOBS", -1))
    print(f"Using n_jobs={n_jobs}\n")

    print("=== Scanning turnover (all sites) ===")
    t0 = time.perf_counter()
    res = scan_all(atm, sites["turnover"].to_numpy(dtype=np.float64), AGELIST, n_jobs=n_jobs)
    print(f"  elapsed: {time.perf_counter() - t0:.2f} s")
    out_main = OUT_DIR / "03b_lognormal_model_age_scan_python.csv"
    write_matrix_csv(out_main, res)
    print(f"  wrote {out_main}  shape {res.shape}\n")

    backfilled = sites[sites["turnover_q05"].notna()].reset_index(drop=True)
    print(f"=== Backfilled sites: {len(backfilled)} ===")

    print("=== Scanning turnover_q05 ===")
    t0 = time.perf_counter()
    res05 = scan_all(atm, backfilled["turnover_q05"].to_numpy(dtype=np.float64), AGELIST, n_jobs=n_jobs)
    print(f"  elapsed: {time.perf_counter() - t0:.2f} s")
    out05 = OUT_DIR / "03b_lognormal_model_age_scan_05_python.csv"
    write_matrix_csv(out05, res05)
    print(f"  wrote {out05}  shape {res05.shape}\n")

    print("=== Scanning turnover_q95 ===")
    t0 = time.perf_counter()
    res95 = scan_all(atm, backfilled["turnover_q95"].to_numpy(dtype=np.float64), AGELIST, n_jobs=n_jobs)
    print(f"  elapsed: {time.perf_counter() - t0:.2f} s")
    out95 = OUT_DIR / "03b_lognormal_model_age_scan_95_python.csv"
    write_matrix_csv(out95, res95)
    print(f"  wrote {out95}  shape {res95.shape}\n")


if __name__ == "__main__":
    main()
