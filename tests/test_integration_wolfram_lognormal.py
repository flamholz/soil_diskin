"""Integration test: compare Mathematica (WolframScript) and Python lognormal scans.

This test runs a small Mathematica age-scan script (`test_lognormal_age_scan_small.wls`),
which performs a 5-point lognormal age scan and writes its results to
`tmp_age_scan.json` in the working directory. The test then loads the same
atmospheric data via `load_atm14c()` and evaluates `lognormal_radiocarbon` for
the same ages in Python. It prints a concise summary and a table of values
showing `fm` from both implementations, absolute and relative errors, and
still asserts numerical agreement within a modest tolerance (1e-2).

This file-based integration approach avoids parsing intermixed Mathematica
warnings on stdout and makes failures easy to inspect by preserving the
JSON output until the test removes it.
"""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from soil_diskin.lognormal import lognormal_radiocarbon
from soil_diskin.radiocarbon_utils import load_atm14c


def test_wolfram_vs_python_lognormal():
    """Run the small WolframScript scan and compare to Python implementation.

    This is an integration-style test and is skipped if `wolframscript` is
    not available on the PATH.
    """
    if shutil.which("wolframscript") is None:
        pytest.skip("wolframscript not available")

    script = os.path.join(os.path.dirname(__file__), "test_lognormal_age_scan_small.wls")
    # Ensure no leftover file
    out_path = os.path.join(os.getcwd(), "tmp_age_scan.json")
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except OSError:
        pass

    try:
        # Run the WolframScript; it will write `tmp_age_scan.json` in cwd.
        subprocess.run(["wolframscript", "-file", script], check=True, timeout=120)

        if not os.path.exists(out_path):
            pytest.fail("wolframscript did not produce tmp_age_scan.json")

        with open(out_path, "r") as fh:
            data = json.load(fh)
    finally:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass

    tau = float(data["tau"])
    ages = [float(x) for x in data["ages"]]
    fm_wolfram = np.asarray(data["fm"], dtype=float)

    # Load atmospheric data the same way the Python code does and evaluate.
    atm = load_atm14c()
    fm_python = np.asarray([lognormal_radiocarbon(atm, tau, a, rtol=1e-6) for a in ages], dtype=float)

    # Allow small numerical differences between integrations.
    # Compute errors and print a concise summary + table for diagnostics.
    abs_err = np.abs(fm_python - fm_wolfram)
    # avoid division by zero for relative error
    denom = np.maximum(np.abs(fm_wolfram), 1e-15)
    rel_err = abs_err / denom

    # Summary
    print("\n=== Age-scan comparison summary ===")
    print(f"tau = {tau}")
    print(f"n_ages = {len(ages)}")
    print(f"mean_abs_err = {abs_err.mean():.6e}")
    imax = int(np.argmax(abs_err))
    print(f"max_abs_err = {abs_err[imax]:.6e} at age {ages[imax]}")
    print(f"mean_rel_err = {rel_err.mean():.6e}")
    jmax = int(np.argmax(rel_err))
    print(f"max_rel_err = {rel_err[jmax]:.6e} at age {ages[jmax]}")

    # Table header
    print("\n age | fm_wolfram | fm_python | abs_err | rel_err")
    for a, fw, fp, ae, re in zip(ages, fm_wolfram, fm_python, abs_err, rel_err):
        print(f"{a:8.2f} | {fw:10.6f} | {fp:10.6f} | {ae:9.6e} | {re:9.6e}")

    np.testing.assert_allclose(fm_python, fm_wolfram, rtol=1e-2, atol=0.0)

