"""Multi-root version of the lognormal recovery experiment.

Same forward setup as `lognormal_recovery_experiment.py`, but the inversion
step enumerates *all* solutions of  smoothed_fm(age) = fm_target  on the
LOWESS-smoothed calibration curve, instead of using `scipy.interpolate.interp1d`
(which silently picks one branch when the curve is non-monotonic).

For each root we back-solve (mu_est, sigma_est) and record:
  - n_candidates: how many distinct ages the calibration can map fm_target to,
  - candidate_<k>: each candidate's (age, mu, sigma) for k = 1..n,
  - best_*: the candidate closest to the true (mu, sigma) in Euclidean
    distance — what recovery accuracy looks like when the operator picks
    the right branch.

Output: results/03_calibrate_models/lognormal_recovery_experiment_multiroot.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.nonparametric.smoothers_lowess import lowess

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from notebooks.experimental.lognormal_radiocarbon import load_atm14c, lognormal_radiocarbon, scan_ages  # noqa: E402

RNG_SEED = 42
N_DRAWS = 100
MU_RANGE = (-4.0, 6.0)
SIGMA_RANGE = (1.0, 4.0)
N_GRID = 101
LOWESS_FRAC = 0.2
ATM_PATH = Path("data/14C_atm_annot.csv")
OUT = Path("results/03_calibrate_models/lognormal_recovery_experiment_multiroot.csv")
FIG_OUT = Path("figures/lognormal_recovery_experiment_multiroot.png")
FIG_CORR_OUT = Path("figures/lognormal_recovery_experiment_multiroot_musigma_corr.png")


def _build_agelist(tau: float, age: float, n: int = N_GRID) -> np.ndarray:
    log10_tau = np.log10(tau)
    log10_age = np.log10(age)
    lo = log10_tau + 0.01
    hi = max(log10_age + 2.0, lo + 6.0)
    return np.logspace(lo, hi, n)


def _all_roots(fm_curve: np.ndarray, age_grid: np.ndarray, target: float) -> list[float]:
    """All ages where the (smoothed) calibration curve crosses `target`,
    via linear interpolation inside each bracketing segment.
    """
    roots: list[float] = []
    for i in range(len(fm_curve) - 1):
        f0, f1 = fm_curve[i], fm_curve[i + 1]
        if (f0 - target) * (f1 - target) <= 0.0 and f0 != f1:
            frac = (target - f0) / (f1 - f0)
            roots.append(float(age_grid[i] + frac * (age_grid[i + 1] - age_grid[i])))
    return roots


def _back_solve(tau: float, age_est: float) -> tuple[float, float]:
    """(mu_est, sigma_est) from (tau, age_est); NaN if age_est <= tau."""
    if age_est <= tau:
        return float("nan"), float("nan")
    sigma = float(np.sqrt(np.log(age_est / tau)))
    mu = sigma ** 2 / 2 - float(np.log(tau))
    return mu, sigma


def _recover_one(atm, mu: float, sigma: float) -> dict:
    tau = float(np.exp(sigma ** 2 / 2 - mu))
    age = float(np.exp(3 * sigma ** 2 / 2 - mu))
    fm_true = float(lognormal_radiocarbon(atm, tau, age))

    agelist = _build_agelist(tau, age)
    curve = scan_ages(atm, tau, agelist)
    smoothed = lowess(curve, agelist, frac=LOWESS_FRAC)
    sm_age = smoothed[:, 0]   # x (already AGELIST, sorted)
    sm_fm = smoothed[:, 1]    # y

    roots = _all_roots(sm_fm, sm_age, fm_true)

    # Back-solve all candidates and pick the one closest to the true (mu, sigma).
    candidates = []
    for r in roots:
        m_est, s_est = _back_solve(tau, r)
        candidates.append({"age_est": r, "mu_est": m_est, "sigma_est": s_est})

    # Pick the closest-to-truth candidate (oracle); NaNs sort to the end.
    best = {"age_est": float("nan"), "mu_est": float("nan"), "sigma_est": float("nan"),
            "best_dist": float("nan"), "best_idx": -1}
    if candidates:
        finite = [(i, c) for i, c in enumerate(candidates)
                  if np.isfinite(c["mu_est"]) and np.isfinite(c["sigma_est"])]
        if finite:
            i_best, c_best = min(finite, key=lambda ic: (ic[1]["mu_est"] - mu) ** 2
                                                     + (ic[1]["sigma_est"] - sigma) ** 2)
            best = {**c_best,
                    "best_dist": float(np.hypot(c_best["mu_est"] - mu,
                                                c_best["sigma_est"] - sigma)),
                    "best_idx": int(i_best)}

    return {
        "mu": mu, "sigma": sigma, "tau": tau, "age": age, "fm_true": fm_true,
        "n_candidates": len(candidates),
        "candidates_json": json.dumps(candidates),  # full set for downstream
        "best_age_est": best["age_est"],
        "best_mu_est": best["mu_est"],
        "best_sigma_est": best["sigma_est"],
        "best_dist": best["best_dist"],
        "best_idx": best["best_idx"],
        "agelist_lo": float(agelist[0]),
        "agelist_hi": float(agelist[-1]),
    }


def plot_recovery(df: pd.DataFrame, out: Path = FIG_OUT) -> None:
    """Scatter of simulated vs. recovered (mu, sigma) using the oracle-best
    candidate per draw."""
    import matplotlib.pyplot as plt

    finite = df.dropna(subset=["best_mu_est", "best_sigma_est"])

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for ax, true_col, est_col, label in (
        (axs[0], "mu", "best_mu_est", r"$\mu$"),
        (axs[1], "sigma", "best_sigma_est", r"$\sigma$"),
    ):
        x = finite[true_col]
        y = finite[est_col]
        ax.scatter(x, y, s=20, alpha=0.6, edgecolors="none")
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax.set_xlabel(f"simulated {label}")
        ax.set_ylabel(f"recovered {label}")
        ax.set_title(label)
        ax.legend(loc="upper left", frameon=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _bin_fits(mu: np.ndarray, sigma: np.ndarray, A: np.ndarray, n_bins: int = 10):
    """Bin samples by A and, in each bin, fit  mu + ln(A_mid) = slope * sigma^2
    (no intercept). Returns {A_midpoint: (slope, ln_midpoint)}."""
    from scipy.stats import gmean

    bins = np.logspace(np.log10(A.min()), np.log10(A.max()), n_bins)
    idx = np.digitize(A, bins)
    fits = {}
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() <= 5:
            continue
        A_mid = float(gmean([A[sel].min(), A[sel].max()]))
        ln_mid = np.log(A_mid)
        x = (sigma[sel] ** 2)[:, None]
        y = mu[sel] + ln_mid
        slope = float(np.linalg.lstsq(x, y, rcond=None)[0].item())
        fits[A_mid] = (slope, ln_mid)
    return fits


def plot_musigma_corr(df: pd.DataFrame, out: Path = FIG_CORR_OUT) -> None:
    """Two-panel sigma^2-vs-mu scatter (simulated | recovered), colored by A
    (= the steady-state amount, here the `age` column), with per-A-bin linear
    fits. For fixed A, theory predicts mu = (3/2) sigma^2 - ln(A)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    finite = df.dropna(subset=["best_mu_est", "best_sigma_est"]).copy()
    A = finite["age"].to_numpy()  # A = tau * exp(sigma^2) = exp(3 sigma^2/2 - mu)
    norm = LogNorm(vmin=A.min(), vmax=A.max())

    panels = [
        ("simulated", finite["mu"].to_numpy(), finite["sigma"].to_numpy()),
        ("recovered", finite["best_mu_est"].to_numpy(), finite["best_sigma_est"].to_numpy()),
    ]

    # y-limits from the scatter (with a small margin) so the long fit lines,
    # which span ~12 decades of A here, don't compress the points.
    all_mu = np.concatenate([mu for _, mu, _ in panels])
    pad = 0.05 * (all_mu.max() - all_mu.min())
    ylim = (all_mu.min() - pad, all_mu.max() + pad)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True,
                            constrained_layout=True)
    scatter = None
    for ax, (title, mu, sigma) in zip(axs, panels):
        scatter = ax.scatter(sigma ** 2, mu, c=A, cmap="viridis", norm=norm, s=20)
        for A_mid, (slope, ln_mid) in _bin_fits(mu, sigma, A).items():
            s2 = np.linspace((sigma ** 2).min(), (sigma ** 2).max(), 100)
            ax.plot(s2, slope * s2 - ln_mid, color=plt.cm.viridis(norm(A_mid)),
                    lw=1, label=f"A≈{A_mid:.1g}; slope={slope:.1f}")
        ax.set(xlabel=r"$\sigma^2$", ylabel=r"$\mu$", title=title, ylim=ylim)
        ax.legend(fontsize=6, loc="lower right")

    fig.colorbar(scatter, ax=axs, label="A")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


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
    df["best_mu_err"]    = df["best_mu_est"]    - df["mu"]
    df["best_sigma_err"] = df["best_sigma_est"] - df["sigma"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(df)} rows)")

    plot_recovery(df)
    plot_musigma_corr(df)

    # Compact summary
    print(f"\nMulti-root statistics:")
    print(df["n_candidates"].value_counts().sort_index().to_string())

    finite_best = df.dropna(subset=["best_mu_est", "best_sigma_est"])
    print(f"\nWith oracle pick of best candidate:")
    print(f"  Finite recoveries: {len(finite_best)} / {len(df)}")
    ok = (finite_best["best_mu_err"].abs() < 0.1) & (finite_best["best_sigma_err"].abs() < 0.05)
    print(f"  |mu_err|<0.1 AND |sigma_err|<0.05:  {ok.sum()} / {len(finite_best)} ({100*ok.sum()/len(finite_best):.0f}%)")
    print()
    print(finite_best[["best_mu_err", "best_sigma_err"]].describe(percentiles=[.5, .9, .99]).round(4))


if __name__ == "__main__":
    main()
