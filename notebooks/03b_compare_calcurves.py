"""Compare prior (golden) and new Python calibration-curve prediction CSVs.

Usage:
  python notebooks/03b_compare_calcurves.py

Defaults:
  prior: results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv
  new:   results/03_calibrate_models/03b_lognormal_predictions_calcurve_python.csv

This script aligns rows by index and compares shared prediction columns:
  pred, pred_05, pred_95

It writes a JSON summary to:
  results/03_calibrate_models/compare_calcurves_report.json
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
COMPARE_COLS = ("pred", "pred_05", "pred_95")


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

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
