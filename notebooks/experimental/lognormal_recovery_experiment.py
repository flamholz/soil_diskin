"""Recovery experiment for the lognormal Diskin calibration pipeline.

Procedure
---------
1. Draw 100 random (mu, sigma) pairs from independent uniform distributions:
       mu    ~ U(-4, 6)
       sigma ~ U( 1, 4)
2. Convert each (mu, sigma) to the (tau, age) parametrization the rest of the
   code uses (see `lognormal_radiocarbon.lognormal_radiocarbon`):
       tau = exp(sigma^2 / 2 - mu)              # E[1/k] for LogNormal(mu,sigma)
       age = exp(3 sigma^2 / 2 - mu)            # mass-weighted mean age
3. Compute the predicted bulk-pool fraction modern at (tau, age) using the
   forward radiocarbon model — call this fm_true.
4. Run exactly the calibration step from `03b_lognormal_calibration.py`:
   build the fm(age) curve at the same tau on the canonical age grid, LOWESS
   smooth (frac=0.2), interpolate, and invert at fm_true to get age_est.
5. Back-solve (mu_est, sigma_est) from (tau, age_est):
       sigma_est = sqrt(log(age_est / tau))
       mu_est    = sigma_est^2 / 2 - log(tau)
6. Save everything (drawn parameters, derived (tau, age), fm_true, age_est,
   recovered (mu, sigma), and per-draw recovery errors) to a CSV.

Output: `results/03_calibrate_models/lognormal_recovery_experiment.csv`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from notebooks.experimental.lognormal_radiocarbon import load_atm14c, lognormal_radiocarbon, scan_ages  # noqa: E402

# --- Experiment configuration ---
RNG_SEED = 42
N_DRAWS = 100
MU_RANGE = (-4.0, 6.0)
SIGMA_RANGE = (1.0, 4.0)

# Calibration grid. The canonical script uses AGELIST = np.logspace(3, 5.5, 101)
# because real soil sites have τ << 10^3 yr. The (μ, σ) box requested for this
# experiment yields τ values up to ~exp(12)≈1.6e5 yr, so for many draws every
# point in the canonical grid satisfies age < τ — and then σ = √log(age/τ)
# blows up. Instead we build a *per-draw* grid that:
#   - starts just above τ (so log(age/τ) > 0 is well-defined),
#   - spans the true age with at least one decade of headroom on each side,
#   - is log-spaced and uses the same number of points (101) as the canonical
#     grid, so the LOWESS+interp1d step is identical in shape.
N_GRID = 101
LOWESS_FRAC = 0.2

ATM_PATH = Path("data/14C_atm_annot.csv")
OUT = Path("results/03_calibrate_models/lognormal_recovery_experiment.csv")


def _build_agelist(tau: float, age: float, n: int = N_GRID) -> np.ndarray:
    """Per-draw log-spaced age grid that brackets the true age and lies
    strictly above tau so that sigma = √log(age/tau) is well-defined.

    For σ ∈ [1, 4], age/tau = exp(σ²) ∈ [exp(1), exp(16)], so true age is
    always at least 0.43 decades above tau but can be up to ~7 decades above.
    """
    log10_tau = np.log10(tau)
    log10_age = np.log10(age)
    # Lower bound: a small ε above tau (mandatory for σ > 0).
    lo = log10_tau + 0.01
    # Upper bound: at least 2 decades above the true age, and a minimum span
    # of 6 decades so LOWESS sees enough curve.
    hi = max(log10_age + 2.0, lo + 6.0)
    return np.logspace(lo, hi, n)


def _recover_one(atm, mu: float, sigma: float) -> dict:
    """Forward-then-invert one (mu, sigma) draw."""
    tau = float(np.exp(sigma ** 2 / 2 - mu))
    age = float(np.exp(3 * sigma ** 2 / 2 - mu))

    # 1. Forward: predicted fm at the true (tau, age).
    fm_true = float(lognormal_radiocarbon(atm, tau, age))

    # 2. Build calibration grid for this (tau, age) and the fm(age) curve.
    agelist = _build_agelist(tau, age)
    curve = scan_ages(atm, tau, agelist)

    # 3. LOWESS smooth + invert (same as 03b_lognormal_calibration.py).
    smoothed = lowess(curve, agelist, frac=LOWESS_FRAC)
    # smoothed[:, 0] = agelist, smoothed[:, 1] = smoothed fm.
    inv = interp1d(smoothed[:, 1], smoothed[:, 0], fill_value="extrapolate")
    age_est = float(inv(fm_true))

    # 4. Back-solve (mu, sigma) from (tau, age_est). For valid sigma > 0 we
    #    must have age_est > tau (since age/tau = exp(sigma^2) > 1).
    if age_est > tau:
        sigma_est = float(np.sqrt(np.log(age_est / tau)))
        mu_est = sigma_est ** 2 / 2 - float(np.log(tau))
    else:
        sigma_est = float("nan")
        mu_est = float("nan")

    return dict(
        mu=mu, sigma=sigma,
        tau=tau, age=age, fm_true=fm_true,
        age_est=age_est, mu_est=mu_est, sigma_est=sigma_est,
        agelist_lo=float(agelist[0]), agelist_hi=float(agelist[-1]),
    )


def main(n_jobs: int = -1) -> None:
    atm = load_atm14c(str(ATM_PATH))
    print(f"atm14C: {len(atm.ages)} non-negative-age knots, mean_R = {atm.mean_R:.6f}")

    rng = np.random.default_rng(RNG_SEED)
    mus = rng.uniform(*MU_RANGE, size=N_DRAWS)
    sigmas = rng.uniform(*SIGMA_RANGE, size=N_DRAWS)
    print(f"Drew {N_DRAWS} samples: mu ~ U{MU_RANGE}, sigma ~ U{SIGMA_RANGE}, seed={RNG_SEED}")

    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_recover_one)(atm, float(m), float(s)) for m, s in zip(mus, sigmas)
    )

    df = pd.DataFrame(rows)
    df["mu_err"] = df["mu_est"] - df["mu"]
    df["sigma_err"] = df["sigma_est"] - df["sigma"]
    df["mu_relerr"] = df["mu_err"] / df["mu"].abs().replace(0, np.nan)
    df["sigma_relerr"] = df["sigma_err"] / df["sigma"].abs()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(df)} rows)")

    print("\nSummary (recovery error statistics, finite recoveries only):")
    finite = df[df["mu_est"].notna() & df["sigma_est"].notna()]
    print(f"  Finite recoveries: {len(finite)} / {len(df)}")
    print(finite[["mu_err", "sigma_err"]].describe(percentiles=[0.5, 0.9, 0.99]))


if __name__ == "__main__":
    main()
