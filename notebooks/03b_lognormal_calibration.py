"""End-to-end lognormal-Diskin calibration in Python.

Merges:
  - notebooks/03b_lognormal_age_scan.py           (forward age scan)
  - notebooks/03b_calibrate_lognormal_model.py    (LOWESS inversion)

Pipeline:
  1. Build the atmospheric ¹⁴C lookup from data/14C_atm_annot.csv.
  2. For each (turnover, age) combination, predict the bulk-pool ¹⁴C ratio
     using the analytical-inner / 1-D quadrature lognormal Diskin model
     (`lognormal_radiocarbon`). This produces three matrices: main, q05, q95.
  3. Optionally write those matrices as age-scan CSVs (matches the Julia/
     Wolfram convention; useful for inspection and for cross-language tools).
  4. For each site, take its row of the relevant age scan as a calibration
     curve fm(age), smooth it with LOWESS (frac=0.2), invert to get age(fm),
     and look up the site's measured `fm` to predict the mean age.
  5. Save a single predictions CSV with columns pred / pred_05 / pred_95
     alongside the original site columns.

Runs the age scans in parallel via joblib. The `_python` suffix on outputs
keeps the Julia-derived CSVs in place for side-by-side comparison; rename or
symlink to drop the suffix if you want this script's outputs to feed
downstream `04_*` scripts.

TODO: The other calibration scripts use a class interface. To make this one 
consistent, we need to provide the AtmC14 to the lognormal diskin class. 
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import argparse

# Allow running this file directly (`uv run .../03b_lognormal_age_scan.py`):
# add the repo root to sys.path so the `notebooks` package is importable.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from soil_diskin.radiocarbon_utils import AtmC14, load_atm14c
from soil_diskin.lognormal import scan_ages

SITES_PATH = Path("results/all_sites_14C_turnover.csv")
ATM_PATH = Path("data/14C_atm_annot.csv")
OUT_DIR = Path("results/03_calibrate_models")

# Calibration grid: same as Wolfram script — 101 log-spaced ages from 10^3 to 10^5.5.
# (np.arange in the original Python script gave only 100 points; we now match
# Mathematica/Julia exactly with 101 points.)
AGELIST = np.logspace(3.0, 5.5, 101)


def run_age_scan(atm: AtmC14, taus: np.ndarray, n_jobs: int) -> np.ndarray:
    """Forward scan: rows = sites, cols = ages, values = predicted fm."""
    rows = Parallel(n_jobs=n_jobs)(
        delayed(scan_ages)(atm, float(t), AGELIST) for t in taus
    )
    return np.vstack(rows)


def predict_ages_from_curve(scan: np.ndarray, fms: np.ndarray) -> np.ndarray:
    """Per-site LOWESS smoothing + inversion. `scan[i]` is site i's curve.

    Mirrors `get_prediction` in `03b_calibrate_lognormal_model.py`.
    """
    preds = np.empty(scan.shape[0], dtype=np.float64)
    for i in range(scan.shape[0]):
        smoothed = lowess(scan[i], AGELIST, frac=0.2)
        # smoothed columns: [0] = AGELIST (x), [1] = smoothed fm (y).
        # Calibration curve maps fm -> age, so invert: x_new = fm, y_new = age.
        calcurve = interp1d(smoothed[:, 1], smoothed[:, 0], fill_value="extrapolate")
        preds[i] = float(calcurve(fms[i]))
    return preds


def main(n_jobs: int = -1, write_age_scans: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    atm = load_atm14c(str(ATM_PATH))
    print(f"atm14C: {len(atm.ages)} non-negative-age knots, mean_R = {atm.mean_R:.6f}")

    sites = pd.read_csv(SITES_PATH)
    backfilled_mask = sites["turnover_q05"].notna() & sites["turnover_q95"].notna()
    backfilled = sites[backfilled_mask].reset_index(drop=True)
    print(f"Sites: {len(sites)} total, {len(backfilled)} with q05/q95 backfill")
    print(f"Joblib n_jobs={n_jobs}\n")

    # --- Forward age scans ---
    print("Scanning turnover (all sites)…")
    t0 = time.perf_counter()
    scan_main = run_age_scan(atm, sites["turnover"].to_numpy(float), n_jobs)
    print(f"  {time.perf_counter() - t0:.2f} s, shape {scan_main.shape}")

    print("Scanning turnover_q05 (backfilled)…")
    t0 = time.perf_counter()
    scan_05 = run_age_scan(atm, backfilled["turnover_q05"].to_numpy(float), n_jobs)
    print(f"  {time.perf_counter() - t0:.2f} s, shape {scan_05.shape}")

    print("Scanning turnover_q95 (backfilled)…")
    t0 = time.perf_counter()
    scan_95 = run_age_scan(atm, backfilled["turnover_q95"].to_numpy(float), n_jobs)
    print(f"  {time.perf_counter() - t0:.2f} s, shape {scan_95.shape}")

    if write_age_scans:
        # Write CSVs with a leading `site_index` column and header row of ages
        # to match the Mathematica/Julia exports (first cell = 'site_index').
        df_main = pd.DataFrame(scan_main, columns=AGELIST)
        df_main.insert(0, "site_index", sites.index + 1)
        df_main.to_csv(OUT_DIR / "03b_lognormal_model_age_scan_python.csv", index=False)

        df_05 = pd.DataFrame(scan_05, columns=AGELIST)
        df_05.insert(0, "site_index", backfilled.index + 1)
        df_05.to_csv(OUT_DIR / "03b_lognormal_model_age_scan_05_python.csv", index=False)

        df_95 = pd.DataFrame(scan_95, columns=AGELIST)
        df_95.insert(0, "site_index", backfilled.index + 1)
        df_95.to_csv(OUT_DIR / "03b_lognormal_model_age_scan_95_python.csv", index=False)

    # --- LOWESS calibration / inversion ---
    print("\nInverting calibration curves (LOWESS frac=0.2)…")
    t0 = time.perf_counter()
    pred = predict_ages_from_curve(scan_main, sites["fm"].to_numpy(float))
    pred_05 = predict_ages_from_curve(scan_05, backfilled["fm"].to_numpy(float))
    pred_95 = predict_ages_from_curve(scan_95, backfilled["fm"].to_numpy(float))
    print(f"  {time.perf_counter() - t0:.2f} s")

    # --- Assemble output frame ---
    out = sites.copy()
    out["pred"] = pred
    backfilled_preds = pd.DataFrame({"pred_05": pred_05, "pred_95": pred_95},
                                    index=sites.index[backfilled_mask])
    out = out.join(backfilled_preds)

    out_path = OUT_DIR / "03b_lognormal_predictions_calcurve_python.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out)} rows, columns: pred, pred_05, pred_95 added)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lognormal Diskin calibration pipeline")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("N_JOBS", -1)),
                        help="Number of parallel jobs for joblib (-1 uses all cores).")
    parser.add_argument("--no-write-age-scans", action="store_true",
                        help="Do not write the intermediate age-scan CSV files.")
    args = parser.parse_args()
    main(n_jobs=args.n_jobs, write_age_scans=not args.no_write_age_scans)
