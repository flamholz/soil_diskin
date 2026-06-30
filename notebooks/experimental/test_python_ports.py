"""Compare Python ports against Julia results, and benchmark.

Two checks:
  1. `diskin_utils_fast.py` vs `diskin_utils_fast.jl` — closed-form fixed-input
     Diskin model. Run both on the same (tau, age) grid and compare element-
     wise; report timings.
  2. `03b_lognormal_age_scan.py` vs `03b_lognormal_age_scan.jl` — full age scan
     using saved Julia output CSVs as the reference.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from notebooks.experimental.diskin_utils_fast import run_diskin_fast as run_diskin_fast_py  # noqa: E402
from notebooks.experimental.lognormal_radiocarbon import load_atm14c, scan_ages  # noqa: E402


REPO = Path(__file__).resolve().parents[1]


# --- Section 1: diskin_utils_fast comparison ---

def diskin_julia_batch(cases, tmax: float = 1000.0, ts_size: int = 100):
    """Run all cases in a single Julia process to amortize startup cost.

    Returns dict mapping (tau, age) -> (ts, C, elapsed_seconds).
    Output goes to a tmp directory: ts.csv (one column), then per-case files
    named `<tau>_<age>.csv` (one column each), and a `times.csv` mapping
    case -> elapsed.
    """
    tmpdir = REPO / "_tmp_julia_diskin_bench"
    tmpdir.mkdir(exist_ok=True)
    cases_julia = "[" + ",".join(f"({tau},{age})" for tau, age in cases) + "]"
    julia_src = f"""
        using Pkg; Pkg.instantiate()
        include(joinpath("notebooks", "diskin_utils_fast.jl"))
        using DelimitedFiles
        cases = {cases_julia}
        # Warm-up: same code path, throwaway timing.
        for (tau, age) in cases
            run_diskin_fast(tau, age; tmax={tmax}, ts_size=10)
        end
        global ts_ref = nothing
        times = Tuple{{Float64,Float64,Float64}}[]
        for (tau, age) in cases
            t = @elapsed (ts, C) = run_diskin_fast(tau, age; tmax={tmax}, ts_size={ts_size})
            global ts_ref = ts
            writedlm(joinpath("{tmpdir}", "C_$(tau)_$(age).csv"), C)
            push!(times, (tau, age, t))
        end
        writedlm(joinpath("{tmpdir}", "ts.csv"), ts_ref)
        open(joinpath("{tmpdir}", "times.csv"), "w") do io
            for (tau, age, t) in times
                println(io, "$tau,$age,$t")
            end
        end
    """
    out = subprocess.run(
        ["julia", "--project=.", "-e", julia_src],
        capture_output=True, text=True, cwd=REPO,
    )
    if out.returncode != 0:
        print(out.stdout); print(out.stderr)
        raise RuntimeError("Julia failed.")
    ts = np.loadtxt(tmpdir / "ts.csv")
    times = {}
    with open(tmpdir / "times.csv") as f:
        for line in f:
            tau, age, t = line.strip().split(",")
            times[(float(tau), float(age))] = float(t)
    result = {}
    for tau, age in cases:
        C = np.loadtxt(tmpdir / f"C_{tau}_{age}.csv")
        result[(tau, age)] = (ts, C, times[(tau, age)])
    return result


def section_diskin_utils_fast():
    print("=" * 72)
    print("SECTION 1 — diskin_utils_fast.{jl,py}")
    print("=" * 72)
    cases = [(10.0, 100.0), (1.0, 50.0), (50.0, 500.0), (5.0, 1000.0)]
    tmax, ts_size = 1000.0, 100

    print("Running Julia batch...")
    julia_results = diskin_julia_batch(cases, tmax=tmax, ts_size=ts_size)

    print(f"\n{'(tau, age)':>16} {'jl_time':>10} {'py_time':>10} "
          f"{'jl/py':>8} {'max_abs':>10} {'max_rel':>10}")
    for tau, age in cases:
        ts_j, C_j, t_jl = julia_results[(tau, age)]
        # Warm-up Python (small grid; not timed)
        run_diskin_fast_py(tau, age, tmax=tmax, ts_size=10)
        t0 = time.perf_counter()
        ts_p, C_p = run_diskin_fast_py(tau, age, tmax=tmax, ts_size=ts_size)
        t_py = time.perf_counter() - t0
        assert np.allclose(ts_j, ts_p)
        abs_err = np.abs(C_j - C_p)
        rel_err = abs_err / np.maximum(np.abs(C_j), 1e-12)
        ratio = t_jl / t_py if t_py > 0 else float("inf")
        print(f"({tau:6.1f}, {age:7.1f}) {t_jl:10.4f} {t_py:10.4f} "
              f"{ratio:8.2f}  {abs_err.max():10.2e} {rel_err.max():10.2e}")
    print()


# --- Section 2: full age-scan comparison + benchmark ---

def section_full_age_scan():
    print("=" * 72)
    print("SECTION 2 — 03b_lognormal_age_scan.{jl,py}")
    print("=" * 72)

    julia_paths = {
        "main": REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan.csv",
        "q05":  REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_05.csv",
        "q95":  REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_95.csv",
    }
    python_paths = {
        "main": REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_python.csv",
        "q05":  REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_05_python.csv",
        "q95":  REPO / "results/03_calibrate_models/03b_lognormal_model_age_scan_95_python.csv",
    }

    # Run Python full scan
    print("Running Python full age scan (this is also the Python benchmark)...")
    t0 = time.perf_counter()
    out = subprocess.run(
        [sys.executable, "notebooks/03b_lognormal_age_scan.py"],
        capture_output=True, text=True, cwd=REPO,
    )
    py_elapsed = time.perf_counter() - t0
    if out.returncode != 0:
        print("STDOUT:\n", out.stdout)
        print("STDERR:\n", out.stderr)
        raise RuntimeError("Python age scan failed.")
    # Surface per-section timings reported by the Python script itself
    for line in out.stdout.splitlines():
        if "elapsed" in line or "Scanning" in line:
            print(" ", line)
    print(f"  Python total wall time: {py_elapsed:.2f} s")
    print()

    # Compare results to Julia CSVs
    print(f"{'scan':>5} {'shape':>14} {'median rel':>14} {'p99 rel':>14} {'max rel':>14}")
    for k in ("main", "q05", "q95"):
        J = pd.read_csv(julia_paths[k], header=None).to_numpy()
        P = pd.read_csv(python_paths[k], header=None).to_numpy()
        assert J.shape == P.shape, f"shape mismatch for {k}: {J.shape} vs {P.shape}"
        rel = np.abs(J - P) / np.maximum(np.abs(J), 1e-12)
        print(f"{k:>5} {str(J.shape):>14} "
              f"{np.median(rel):14.3e} {np.quantile(rel, 0.99):14.3e} {rel.max():14.3e}")


if __name__ == "__main__":
    section_diskin_utils_fast()
    section_full_age_scan()
