"""Compare prior (golden) and new outputs for calibration curves and age scans.

Usage:
  python notebooks/03b_compare_calcurves.py

Defaults:
  prior: results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv
  new:   results/03_calibrate_models/03b_lognormal_predictions_calcurve_python.csv

This script performs two comparisons:
    1) Calibration-curve prediction CSVs (pred, pred_05, pred_95).
    2) Python age-scan matrices versus reference age-scan matrices.

For age scans, disagreements are split into:
    - expected numeric differences (kept in post-filter error summaries), and
    - likely reference mis-convergence cells, detected by a local smoothness
      heuristic on the reference row.

It writes JSON summaries to:
    results/03_calibrate_models/compare_calcurves_report.json
    results/03_calibrate_models/compare_age_scans_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PRIOR = REPO / "results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv"
DEFAULT_NEW = REPO / "results/03_calibrate_models/03b_lognormal_predictions_calcurve_python.csv"
DEFAULT_OUT = REPO / "results/03_calibrate_models/compare_calcurves_report.json"
DEFAULT_OUT_AGE = REPO / "results/03_calibrate_models/compare_age_scans_report.json"
COMPARE_COLS = ("pred", "pred_05", "pred_95")
AGE_SCAN_FILES = (
    ("main", REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_python.csv", REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan.csv"),
    ("q05", REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_05_python.csv", REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_05.csv"),
    ("q95", REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_95_python.csv", REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_95.csv"),
)


def _load_age_scan_numeric(path: Path) -> np.ndarray:
    """Load an age-scan CSV and return numeric matrix (rows=sites, cols=ages).

    Handles both styles present in this repo:
    - headered CSVs with a `site_index` metadata column,
    - headerless pure numeric matrices.
    """
    df = pd.read_csv(path)

    # Most age-scan files include a metadata index column.
    if "site_index" in df.columns:
        df = df.drop(columns=["site_index"])

    df = df.apply(pd.to_numeric, errors="coerce")

    # Fallback for headerless numeric files.
    if np.isnan(df.to_numpy(dtype=float)).all():
        df = pd.read_csv(path, header=None)
        df = df.apply(pd.to_numeric, errors="coerce")

    # If a leading index-like column slipped through, drop it.
    if df.shape[1] > 1:
        first = df.iloc[:, 0].to_numpy(dtype=float)
        idx0 = np.arange(len(first), dtype=float)
        idx1 = idx0 + 1.0
        if np.allclose(first, idx0) or np.allclose(first, idx1):
            df = df.iloc[:, 1:]

    return df.to_numpy(dtype=float)


def _is_wolfram_smooth_local(row: np.ndarray, j: int, rel_jump: float = 0.01) -> bool:
        """Return True when reference cell ``row[j]`` is locally smooth in age.

        Heuristic:
        - Build a robust local baseline from non-adjacent anchors at offsets
            ``{-5, -3, +3, +5}`` around ``j``.
        - Use the median of available anchors as the baseline.
        - Mark the point smooth when relative jump to baseline is below
            ``rel_jump``.

        Rationale:
        Adjacent cells can belong to the same short mis-converged run, so
        non-adjacent anchors reduce sensitivity to spike clusters.
        """
        offsets = (-5, -3, 3, 5)
        anchors: list[float] = []
        n = len(row)

        for off in offsets:
                k = j + off
                if 0 <= k < n:
                        anchors.append(float(row[k]))

        if not anchors:
                return True

        baseline = float(np.median(np.asarray(anchors, dtype=float)))
        return abs(float(row[j]) - baseline) / max(abs(baseline), 1e-12) < rel_jump


def _compare_age_scan_matrix(
    name: str,
    new_path: Path,
    ref_path: Path,
    rel_err_thresh: float = 5e-3,
    rel_jump: float = 0.01,
) -> dict:
    """Compare one Python/reference age-scan pair with spike-aware filtering.

    A cell is classified as a likely reference failure when BOTH are true:
    1) relative error exceeds `rel_err_thresh`, and
    2) the reference value fails the local smoothness test.

    The report contains both:
    - overall error statistics (all cells), and
    - post-filter statistics (excluding likely reference-failure cells).
    """
    if not new_path.exists():
        raise FileNotFoundError(f"Missing Python age-scan file: {new_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference age-scan file: {ref_path}")

    new = _load_age_scan_numeric(new_path)
    ref = _load_age_scan_numeric(ref_path)
    if new.shape != ref.shape:
        raise ValueError(
            f"Shape mismatch for {name}: python={new.shape}, reference={ref.shape}"
        )

    nrows, ncols = ref.shape
    all_rel = []
    agree_rel = []
    wolfram_failures = []

    for i in range(nrows):
        ref_row = ref[i, :]
        for j in range(ncols):
            rel_err = abs(float(new[i, j]) - float(ref[i, j])) / max(abs(float(ref[i, j])), 1e-12)
            all_rel.append(rel_err)
            if rel_err > rel_err_thresh and not _is_wolfram_smooth_local(ref_row, j, rel_jump=rel_jump):
                wolfram_failures.append((i + 1, j + 1, rel_err))
            else:
                agree_rel.append(rel_err)

    all_rel_arr = np.asarray(all_rel, dtype=float)
    agree_rel_arr = np.asarray(agree_rel, dtype=float)
    total_cells = int(nrows * ncols)
    n_fail = int(len(wolfram_failures))

    report = {
        "name": name,
        "python_path": str(new_path),
        "reference_path": str(ref_path),
        "shape": [int(nrows), int(ncols)],
        "n_cells": total_cells,
        "overall_median_rel_err": float(np.median(all_rel_arr)),
        "overall_p99_rel_err": float(np.quantile(all_rel_arr, 0.99)),
        "wolfram_failures_detected": n_fail,
        "wolfram_failures_pct": float(100.0 * n_fail / max(total_cells, 1)),
        "post_filter_max_rel_err": float(np.max(agree_rel_arr)) if agree_rel_arr.size else None,
        "post_filter_p99_rel_err": float(np.quantile(agree_rel_arr, 0.99)) if agree_rel_arr.size else None,
        "post_filter_median_rel_err": float(np.median(agree_rel_arr)) if agree_rel_arr.size else None,
        "failure_examples": [
            {"row": int(r), "col": int(c), "rel_err": float(e)}
            for (r, c, e) in wolfram_failures[:20]
        ],
    }
    return report


def compare_age_scans() -> dict:
    """Run matrix comparison for main, q05, and q95 age-scan products."""
    scans = []
    for name, py_path, ref_path in AGE_SCAN_FILES:
        scans.append(_compare_age_scan_matrix(name, py_path, ref_path))
    return {"scans": scans}


def _series_metrics(prior: pd.Series, new: pd.Series) -> dict:
    prior_arr = prior.to_numpy(dtype=float)
    new_arr = new.to_numpy(dtype=float)

    both_finite = np.isfinite(prior_arr) & np.isfinite(new_arr)
    prior_nan = ~np.isfinite(prior_arr)
    new_nan = ~np.isfinite(new_arr)

    if both_finite.any():
        diff = new_arr[both_finite] - prior_arr[both_finite]
        abs_diff = np.abs(diff)
        rel = abs_diff / np.maximum(np.abs(prior_arr[both_finite]), 1e-12)
        out = {
            "n_total": int(len(prior_arr)),
            "n_both_finite": int(both_finite.sum()),
            "n_prior_nonfinite": int(prior_nan.sum()),
            "n_new_nonfinite": int(new_nan.sum()),
            "n_nonfinite_mismatch": int((prior_nan != new_nan).sum()),
            "mean_abs_err": float(abs_diff.mean()),
            "median_abs_err": float(np.median(abs_diff)),
            "p99_abs_err": float(np.quantile(abs_diff, 0.99)),
            "max_abs_err": float(abs_diff.max()),
            "mean_rel_err": float(rel.mean()),
            "median_rel_err": float(np.median(rel)),
            "p99_rel_err": float(np.quantile(rel, 0.99)),
            "max_rel_err": float(rel.max()),
        }
    else:
        out = {
            "n_total": int(len(prior_arr)),
            "n_both_finite": 0,
            "n_prior_nonfinite": int(prior_nan.sum()),
            "n_new_nonfinite": int(new_nan.sum()),
            "n_nonfinite_mismatch": int((prior_nan != new_nan).sum()),
            "error": "no overlapping finite values",
        }
    return out


def compare(prior_path: Path, new_path: Path) -> dict:
    if not prior_path.exists():
        raise FileNotFoundError(f"Missing prior file: {prior_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"Missing new file: {new_path}")

    prior = pd.read_csv(prior_path)
    new = pd.read_csv(new_path)

    n_rows = min(len(prior), len(new))
    if len(prior) != len(new):
        print(
            f"Warning: row-count mismatch ({len(prior)} vs {len(new)}); comparing first {n_rows} rows only.",
            file=sys.stderr,
        )

    prior = prior.iloc[:n_rows].reset_index(drop=True)
    new = new.iloc[:n_rows].reset_index(drop=True)

    shared_cols = [c for c in COMPARE_COLS if c in prior.columns and c in new.columns]
    missing_in_prior = [c for c in COMPARE_COLS if c not in prior.columns]
    missing_in_new = [c for c in COMPARE_COLS if c not in new.columns]

    results = {}
    for col in shared_cols:
        results[col] = _series_metrics(prior[col], new[col])

    return {
        "prior_path": str(prior_path),
        "new_path": str(new_path),
        "rows_compared": int(n_rows),
        "missing_in_prior": missing_in_prior,
        "missing_in_new": missing_in_new,
        "column_results": results,
    }


def write_overlay_plot(prior: pd.DataFrame, new: pd.DataFrame, columns: list[str], out_path: Path) -> None:
    if not columns:
        return

    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(11, max(3.0, 2.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    x = np.arange(len(prior))
    for ax, col in zip(axes, columns):
        prior_vals = prior[col].to_numpy(dtype=float)
        new_vals = new[col].to_numpy(dtype=float)
        prior_mask = np.isfinite(prior_vals)
        new_mask = np.isfinite(new_vals)
        both_mask = prior_mask & new_mask

        ax.plot(x[prior_mask], prior_vals[prior_mask], label="prior", linewidth=1.2)
        ax.plot(x[new_mask], new_vals[new_mask], label="new", linewidth=1.2, alpha=0.85)
        ax.text(
            0.01,
            0.98,
            f"finite prior/new/both: {prior_mask.sum()}/{new_mask.sum()}/{both_mask.sum()}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("row index")
    axes[0].legend(loc="best")
    fig.suptitle("Calibration predictions: prior vs new", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, default=DEFAULT_PRIOR, help="Path to prior/golden CSV")
    p.add_argument("--new", type=Path, default=DEFAULT_NEW, help="Path to new CSV")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Path to output JSON report")
    p.add_argument(
        "--out-age",
        type=Path,
        default=DEFAULT_OUT_AGE,
        help="Path to output JSON report for age-scan comparison",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = compare(args.prior, args.new)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    prior_df = pd.read_csv(args.prior).iloc[: report["rows_compared"]].reset_index(drop=True)
    new_df = pd.read_csv(args.new).iloc[: report["rows_compared"]].reset_index(drop=True)
    plot_cols = list(report["column_results"].keys())
    plot_path = args.out.with_name(f"{args.out.stem}.png")
    write_overlay_plot(prior_df, new_df, plot_cols, plot_path)

    print(f"Wrote report: {args.out}")
    print(f"Wrote plot:   {plot_path}")
    print(f"Rows compared: {report['rows_compared']}")

    if report["missing_in_prior"]:
        print(f"Missing in prior: {report['missing_in_prior']}")
    if report["missing_in_new"]:
        print(f"Missing in new: {report['missing_in_new']}")

    if not report["column_results"]:
        print("No shared comparison columns among pred/pred_05/pred_95.")
        return 2

    header = f"{'column':>8} {'n':>6} {'mae':>12} {'p99 abs':>12} {'max abs':>12} {'median rel':>12} {'max rel':>12}"
    print(header)
    for col, stats in report["column_results"].items():
        if "error" in stats:
            print(f"{col:>8} ERROR: {stats['error']}")
            continue
        print(
            f"{col:>8} {stats['n_both_finite']:6d} "
            f"{stats['mean_abs_err']:12.4e} {stats['p99_abs_err']:12.4e} {stats['max_abs_err']:12.4e} "
            f"{stats['median_rel_err']:12.4e} {stats['max_rel_err']:12.4e}"
        )

    age_report = compare_age_scans()
    args.out_age.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_age, "w") as f:
        json.dump(age_report, f, indent=2)

    print(f"Wrote age-scan report: {args.out_age}")
    for scan in age_report["scans"]:
        print(f"=== {scan['name']} ===")
        print(
            f"grid: {scan['shape'][0]} x {scan['shape'][1]} ({scan['n_cells']} cells)"
        )
        print(f"overall median rel err: {scan['overall_median_rel_err']:.3e}")
        print(f"overall 99th pctile:    {scan['overall_p99_rel_err']:.3e}")
        print(
            "Wolfram failures detected: "
            f"{scan['wolfram_failures_detected']} cells ({scan['wolfram_failures_pct']:.2f}%)"
        )
        if scan["post_filter_median_rel_err"] is not None:
            print("After removing Wolfram failures:")
            print(f"  max rel err:  {scan['post_filter_max_rel_err']:.3e}")
            print(f"  99th pctile:  {scan['post_filter_p99_rel_err']:.3e}")
            print(f"  median:       {scan['post_filter_median_rel_err']:.3e}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
