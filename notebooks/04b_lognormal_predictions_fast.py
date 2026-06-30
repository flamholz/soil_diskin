"""Python port of `04b_lognormal_predictions_fast_julia.jl`.

For each row in the calibration-curve predictions file, evaluate the
closed-form Diskin C(t) on a log-spaced time grid.

Inputs:
    results/03_calibrate_models/03b_lognormal_predictions_calcurve_julia.csv
        with columns turnover, pred, turnover_q05, pred_05, turnover_q95, pred_95

Outputs:
    results/04_model_predictions/04b_lognormal_cdfs_python.csv
    results/04_model_predictions/04b_lognormal_cdfs_05_python.csv
    results/04_model_predictions/04b_lognormal_cdfs_95_python.csv

The 05/95 outputs match the Julia layout: a `len(params) × ts_size` matrix
where only the first `len(backfilled)` rows hold real values; the remaining
rows hold whatever `np.empty` returns (matches Julia's `Matrix(undef, …)`).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from soil_diskin.lognormal import run_diskin_fast

DEFAULT_PARAMS_PATH = Path("results/03_calibrate_models/03b_lognormal_predictions_calcurve_julia.csv")
DEFAULT_OUT_DIR = Path("results/04_model_predictions")

TMAX = 100_000
TS_SIZE = 1000
TS = np.logspace(-1.0, np.log10(TMAX), TS_SIZE)


def _run_one(tau: float, age: float) -> np.ndarray:
    _, C = run_diskin_fast(float(tau), float(age), 1.0, tmax=TMAX, ts_size=TS_SIZE)
    return C


def _scan(taus: np.ndarray, ages: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(t, a) for t, a in zip(taus, ages)
    )
    return np.vstack(rows)


def _write_with_ts_header(path: Path, mat: np.ndarray) -> None:
    """Write `mat` with `TS` values as column headers — matches Julia's
    `Tables.table(results, header = ts)` followed by `CSV.write`.
    """
    df = pd.DataFrame(mat, columns=[str(t) for t in TS])
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params-path",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
        help="Path to calibration-curve predictions CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for model prediction outputs",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="joblib n_jobs setting",
    )
    return parser.parse_args()


def main(params_path: Path = DEFAULT_PARAMS_PATH, out_dir: Path = DEFAULT_OUT_DIR, n_jobs: int = -1) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = pd.read_csv(params_path)
    n = len(params)
    print(f"Loaded {n} param rows from {params_path}")

    # --- Main scan over all sites ---
    print("Scanning main (turnover, pred)…")
    t0 = time.perf_counter()
    results = _scan(params["turnover"].to_numpy(float), params["pred"].to_numpy(float), n_jobs=n_jobs)
    print(f"  elapsed: {time.perf_counter() - t0:.2f} s  shape {results.shape}")
    _write_with_ts_header(out_dir / "04b_lognormal_cdfs_python.csv", results)

    # --- q05 / q95 scans on backfilled rows only ---
    backfilled = params[params["turnover_q05"].notna()].reset_index(drop=True)
    nb = len(backfilled)
    print(f"Backfilled rows (non-missing turnover_q05): {nb}")

    # Match Julia: pre-allocate a (n × TS_SIZE) matrix and only fill the first
    # `nb` rows. Use np.empty so the unfilled rows hold uninitialized memory
    # (i.e., the same garbage Julia's `undef` produces).
    results_05 = np.empty((n, TS_SIZE), dtype=np.float64)
    results_95 = np.empty((n, TS_SIZE), dtype=np.float64)

    print("Scanning q05 (turnover_q05, pred_05)…")
    t0 = time.perf_counter()
    block_05 = _scan(backfilled["turnover_q05"].to_numpy(float),
                     backfilled["pred_05"].to_numpy(float), n_jobs=n_jobs)
    results_05[:nb, :] = block_05
    print(f"  elapsed: {time.perf_counter() - t0:.2f} s  filled rows {nb}")

    print("Scanning q95 (turnover_q95, pred_95)…")
    t0 = time.perf_counter()
    block_95 = _scan(backfilled["turnover_q95"].to_numpy(float),
                     backfilled["pred_95"].to_numpy(float), n_jobs=n_jobs)
    results_95[:nb, :] = block_95
    print(f"  elapsed: {time.perf_counter() - t0:.2f} s  filled rows {nb}")

    _write_with_ts_header(out_dir / "04b_lognormal_cdfs_05_python.csv", results_05)
    _write_with_ts_header(out_dir / "04b_lognormal_cdfs_95_python.csv", results_95)
    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(params_path=args.params_path, out_dir=args.out_dir, n_jobs=args.n_jobs)
