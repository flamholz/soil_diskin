"""Python port of `lognormal_radiocarbon.jl`.

Predicted bulk-pool ¹⁴C activity ratio for the lognormal Diskin model.

    r(tau, age) = exp(mu - sigma^2 / 2)
                  * integral over u of phi_N(u; mu, sigma) * I(exp(u)) du

where I(k) = integral over a from 0 to inf of atm14C(a) * exp(-(1/8267 + k) a) da
is computed analytically because atm14C is piecewise constant.
"""
import numpy as np
import pandas as pd
from math import sqrt, log, exp
from scipy.integrate import quad


C14_MEAN_LIFE = 8267.0  # years
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


class AtmC14:
    """Piecewise-constant atmospheric ¹⁴C lookup with a constant tail."""

    __slots__ = ("ages", "fm", "mean_R")

    def __init__(self, ages: np.ndarray, fm: np.ndarray, mean_R: float):
        self.ages = np.ascontiguousarray(ages, dtype=np.float64)
        self.fm = np.ascontiguousarray(fm, dtype=np.float64)
        self.mean_R = float(mean_R)


def load_atm14c(path: str = "data/14C_atm_annot.csv") -> AtmC14:
    """Replicates the Mathematica/Julia setup:

    - read CSV with header
    - take columns 4 (`years_before_2000`) and 5 (`R_14C`)
    - mean_R = mean of R_14C over the LAST 50 000 rows (in original file order)
    - keep only ages >= 0, sorted ascending
    """
    df = pd.read_csv(path)
    raw_age = df.iloc[:, 3].to_numpy(dtype=np.float64)  # years_before_2000
    raw_fm = df.iloc[:, 4].to_numpy(dtype=np.float64)   # R_14C
    mean_R = float(raw_fm[-50_000:].mean())
    perm = np.argsort(raw_age)
    a = raw_age[perm]
    f = raw_fm[perm]
    mask = a >= 0.0
    return AtmC14(a[mask], f[mask], mean_R)


def inner_integral(atm: AtmC14, alpha: float) -> float:
    """Closed-form for I(alpha) = ∫_0^∞ atm14C(a) exp(-alpha a) da.

    O(N) vectorized in numpy.
    """
    ages = atm.ages
    fm = atm.fm
    e = np.exp(-alpha * ages)
    # sum_i fm[i] * (e[i] - e[i+1])  for i = 0 .. N-2,
    # plus mean_R * e[N-1] for the constant tail beyond the last knot.
    body = fm[:-1] * (e[:-1] - e[1:])
    return (body.sum() + atm.mean_R * e[-1]) / alpha


def lognormal_radiocarbon(atm: AtmC14, tau: float, age: float, rtol: float = 1e-4) -> float:
    """Predicted bulk-pool ¹⁴C activity ratio for the lognormal Diskin model."""
    sigma = sqrt(log(age / tau))
    mu = -log(sqrt(tau ** 3 / age))
    u_lo = mu - 10.0 * sigma
    u_hi = mu + 10.0 * sigma
    inv_sigma = 1.0 / sigma

    def integrand(u):
        z = (u - mu) * inv_sigma
        phi = _INV_SQRT_2PI * np.exp(-0.5 * z * z) * inv_sigma
        return phi * inner_integral(atm, 1.0 / C14_MEAN_LIFE + np.exp(u))

    val, _ = quad(integrand, u_lo, u_hi, epsrel=rtol, limit=200)
    return val * exp(mu - 0.5 * sigma * sigma)


def scan_ages(atm: AtmC14, tau: float, agelist, rtol: float = 1e-4) -> np.ndarray:
    return np.array([lognormal_radiocarbon(atm, tau, float(a), rtol=rtol) for a in agelist])
