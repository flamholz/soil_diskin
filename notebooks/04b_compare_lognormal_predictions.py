"""Compare Julia and Python 04b lognormal CDF prediction outputs.

Compares the three output matrices:
  - main: 04b_lognormal_cdfs.csv vs 04b_lognormal_cdfs_python.csv
  - q05:  04b_lognormal_cdfs_05.csv vs 04b_lognormal_cdfs_05_python.csv
  - q95:  04b_lognormal_cdfs_95.csv vs 04b_lognormal_cdfs_95_python.csv

For q05/q95, only the first N backfilled rows are compared, where N is inferred
from `turnover_q05.notna()` in the calibration-curve CSV.

Usage:
  python notebooks/04b_compare_lognormal_predictions.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "julia_main": REPO / "results/04_model_predictions/04b_lognormal_cdfs.csv",
    "python_main": REPO / "results/04_model_predictions/04b_lognormal_cdfs_python.csv",
    "julia_q05": REPO / "results/04_model_predictions/04b_lognormal_cdfs_05.csv",
    "python_q05": REPO / "results/04_model_predictions/04b_lognormal_cdfs_05_python.csv",
    "julia_q95": REPO / "results/04_model_predictions/04b_lognormal_cdfs_95.csv",
    "python_q95": REPO / "results/04_model_predictions/04b_lognormal_cdfs_95_python.csv",
    "calcurve": REPO / "results/03_calibrate_models/03b_lognormal_predictions_calcurve_python.csv",
    "out": REPO / "results/04_model_predictions/compare_lognormal_cdfs_report.json",
}


def _load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path).to_numpy(dtype=float)


def _metrics(j: np.ndarray, p: np.ndarray) -> dict:
    both = np.isfinite(j) & np.isfinite(p)
    j_nonfinite = ~np.isfinite(j)
    p_nonfinite = ~np.isfinite(p)

    out = {
        "shape": [int(j.shape[0]), int(j.shape[1])],
        "n_total": int(j.size),
        "n_both_finite": int(both.sum()),
        "n_julia_nonfinite": int(j_nonfinite.sum()),
        "n_python_nonfinite": int(p_nonfinite.sum()),
        "n_nonfinite_mismatch": int((j_nonfinite != p_nonfinite).sum()),
    }

    if both.any():
        d = p[both] - j[both]
        ad = np.abs(d)
        rel = ad / np.maximum(np.abs(j[both]), 1e-12)
        out.update(
            {
                "mean_abs_err": float(ad.mean()),
                "median_abs_err": float(np.median(ad)),
                "p99_abs_err": float(np.quantile(ad, 0.99)),
                "max_abs_err": float(ad.max()),
                "mean_rel_err": float(rel.mean()),
                "median_rel_err": float(np.median(rel)),
                "p99_rel_err": float(np.quantile(rel, 0.99)),
                "max_rel_err": float(rel.max()),
            }
        )
    else:
        out["error"] = "no overlapping finite entries"

    return out


def _trim_pair(j: np.ndarray, p: np.ndarray, n_rows: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = min(j.shape[0], p.shape[0])
    cols = min(j.shape[1], p.shape[1])

    if n_rows is not None:
        rows = min(rows, n_rows)

    info = {
        "julia_shape": [int(j.shape[0]), int(j.shape[1])],
        "python_shape": [int(p.shape[0]), int(p.shape[1])],
        "rows_compared": int(rows),
        "cols_compared": int(cols),
    }

    return j[:rows, :cols], p[:rows, :cols], info


def _infer_backfilled_rows(calcurve_path: Path) -> int:
    if not calcurve_path.exists():
        raise FileNotFoundError(f"Missing calibration file: {calcurve_path}")
    df = pd.read_csv(calcurve_path)
    if "turnover_q05" not in df.columns:
        raise ValueError(f"Expected column 'turnover_q05' in: {calcurve_path}")
    return int(df["turnover_q05"].notna().sum())


def _plot_summary(results: dict, plot_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    order = ["main", "q05", "q95"]

    for i, key in enumerate(order):
        mat_j = np.asarray(results[key]["_julia_matrix"])
        mat_p = np.asarray(results[key]["_python_matrix"])

        # Left: row means overlay
        ax = axes[i, 0]
        j_row = np.nanmean(mat_j, axis=1)
        p_row = np.nanmean(mat_p, axis=1)
        x = np.arange(len(j_row))
        ax.plot(x, j_row, label="julia", linewidth=1.2)
        ax.plot(x, p_row, label="python", linewidth=1.2, alpha=0.85)
        ax.set_title(f"{key}: row-mean overlay")
        ax.set_xlabel("row")
        ax.set_ylabel("mean C(t)")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="best")

        # Right: parity scatter (sampled)
        ax = axes[i, 1]
        jf = mat_j.ravel()
        pf = mat_p.ravel()
        mask = np.isfinite(jf) & np.isfinite(pf)
        jf = jf[mask]
        pf = pf[mask]
        if jf.size > 0:
            if jf.size > 20000:
                idx = np.linspace(0, jf.size - 1, 20000, dtype=int)
                jf = jf[idx]
                pf = pf[idx]
            ax.scatter(jf, pf, s=3, alpha=0.25)
            lo = float(min(jf.min(), pf.min()))
            hi = float(max(jf.max(), pf.max()))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(f"{key}: parity scatter")
        ax.set_xlabel("julia")
        ax.set_ylabel("python")
        ax.grid(True, alpha=0.25)

    fig.suptitle("04b lognormal CDF comparison: Julia vs Python", y=0.995)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--julia-main", type=Path, default=DEFAULTS["julia_main"])
    p.add_argument("--python-main", type=Path, default=DEFAULTS["python_main"])
    p.add_argument("--julia-q05", type=Path, default=DEFAULTS["julia_q05"])
    p.add_argument("--python-q05", type=Path, default=DEFAULTS["python_q05"])
    p.add_argument("--julia-q95", type=Path, default=DEFAULTS["julia_q95"])
    p.add_argument("--python-q95", type=Path, default=DEFAULTS["python_q95"])
    p.add_argument("--calcurve", type=Path, default=DEFAULTS["calcurve"])
    p.add_argument("--out", type=Path, default=DEFAULTS["out"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    nb = _infer_backfilled_rows(args.calcurve)

    pairs = {
        "main": (_load_matrix(args.julia_main), _load_matrix(args.python_main), None),
        "q05": (_load_matrix(args.julia_q05), _load_matrix(args.python_q05), nb),
        "q95": (_load_matrix(args.julia_q95), _load_matrix(args.python_q95), nb),
    }

    report = {
        "inputs": {
            "julia_main": str(args.julia_main),
            "python_main": str(args.python_main),
            "julia_q05": str(args.julia_q05),
            "python_q05": str(args.python_q05),
            "julia_q95": str(args.julia_q95),
            "python_q95": str(args.python_q95),
            "calcurve": str(args.calcurve),
        },
        "backfilled_rows_used_for_q05_q95": nb,
        "results": {},
    }

    plot_cache = {}
    for key, (j, p, forced_rows) in pairs.items():
        jt, pt, trim_info = _trim_pair(j, p, n_rows=forced_rows)
        met = _metrics(jt, pt)
        report["results"][key] = {**trim_info, **met}
        plot_cache[key] = {"_julia_matrix": jt, "_python_matrix": pt}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    plot_path = args.out.with_name(f"{args.out.stem}.png")
    _plot_summary(plot_cache, plot_path)

    print(f"Backfilled rows used for q05/q95: {nb}")
    print(f"Wrote report: {args.out}")
    print(f"Wrote plot:   {plot_path}")

    print(f"{'pair':>6} {'rows':>6} {'cols':>6} {'mae':>12} {'p99 abs':>12} {'max abs':>12} {'median rel':>12} {'max rel':>12}")
    for key in ("main", "q05", "q95"):
        r = report["results"][key]
        if "error" in r:
            print(f"{key:>6} ERROR: {r['error']}")
            continue
        print(
            f"{key:>6} {r['rows_compared']:6d} {r['cols_compared']:6d} "
            f"{r['mean_abs_err']:12.4e} {r['p99_abs_err']:12.4e} {r['max_abs_err']:12.4e} "
            f"{r['median_rel_err']:12.4e} {r['max_rel_err']:12.4e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
