"""Lognormal model-related utility functions.

Provides `inner_integral`, `lognormal_radiocarbon`, `scan_ages`, and
closed-form fixed-input concentration helpers.
These are used by the calibration and recovery scripts.
"""
from __future__ import annotations

import numpy as np
from math import sqrt, log, exp
from scipy.integrate import quad

from .radiocarbon_utils import AtmC14

__all__ = [
    "inner_integral",
    "lognormal_radiocarbon",
    "scan_ages",
    "diskin_C_of_t",
    "run_diskin_fast",
]


C14_MEAN_LIFE = 8267.0  # years
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def inner_integral(atm: AtmC14, alpha: float) -> float:
    """Closed-form for I(alpha) = ∫_0^∞ atm14C(a) exp(-alpha a) da.

    Vectorized in NumPy; returns a scalar.
    """
    ages = atm.ages
    fm = atm.fm
    e = np.exp(-alpha * ages)
    body = fm[:-1] * (e[:-1] - e[1:])
    return (body.sum() + atm.mean_R * e[-1]) / alpha


def lognormal_radiocarbon(
    atm: AtmC14,
    tau: float,
    age: float,
    rtol: float = 1e-4,
) -> float:
    """Predicted bulk-pool 14C activity ratio for the lognormal Diskin model.

    Computed as a normalized double integral:
    fm = exp(mu - 0.5*sigma^2) * integral_u[ phi(u; mu, sigma) * I(lambda + exp(u)) du ],
    with inner term I(alpha) = integral_a[ atm14C(a) * exp(-alpha*a) da ].
    Here u = ln(k), k = exp(u), and lambda = 1 / C14_MEAN_LIFE.

    Parameters
    ----------
    atm: AtmC14
        Atmospheric lookup
    tau: float
        Turnover time (mean residence time) of the bulk pool
    age: float
        Mass-weighted mean age at steady state
    rtol: float
        Relative tolerance passed to the outer quadrature
    """
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
    # Normalize by bulk C_ss via 1 / E[1/k] for lognormal k, i.e. exp(mu - 0.5*sigma^2).
    # This computes the activity ratio under the standard trace-isotope assumption (14C << 12C).
    return val * exp(mu - 0.5 * sigma * sigma)


def scan_ages(atm: AtmC14, tau: float, agelist, rtol: float = 1e-4) -> np.ndarray:
    """Compute predicted fm for each age in `agelist` at fixed `tau`.

    Returns a NumPy 1-D array with the same length as `agelist`.
    """
    return np.array([lognormal_radiocarbon(atm, tau, float(a), rtol=rtol) for a in agelist])


def diskin_C_of_t(
    ts: np.ndarray,
    mu: float,
    sigma: float,
    input_: float = 1.0,
    rtol: float = 1e-10,
) -> np.ndarray:
    """Evaluate closed-form fixed-input C(t) for a lognormal Diskin pool.

    Parameters
    ----------
    ts: np.ndarray
        Time points at which to evaluate concentration.
    mu: float
        Mean of log(k).
    sigma: float
        Standard deviation of log(k).
    input_: float
        Constant input rate.
    rtol: float
        Relative tolerance passed to quadrature.
    """
    u_lo = mu - sigma * sigma - 10.0 * sigma
    u_hi = mu + 10.0 * sigma
    out = np.empty(len(ts), dtype=np.float64)
    inv_sigma = 1.0 / sigma

    for i, t in enumerate(ts):
        def integrand(u, t=t):
            z = (u - mu) * inv_sigma
            phi = _INV_SQRT_2PI * np.exp(-0.5 * z * z) * inv_sigma
            return np.exp(-u) * phi * (-np.expm1(-np.exp(u) * t))

        val, _ = quad(integrand, u_lo, u_hi, epsrel=rtol, limit=200)
        out[i] = input_ * val

    return out


def run_diskin_fast(
    tau: float,
    age: float,
    input_: float = 1.0,
    tmax: float = 100.0,
    ts_size: int = 1000,
    rtol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(ts, C)` on a log-spaced time grid from 0.1 to `tmax`."""
    sigma = sqrt(log(age / tau))
    mu = -log(sqrt(tau ** 3 / age))
    ts = np.logspace(-1.0, np.log10(tmax), ts_size)
    return ts, diskin_C_of_t(ts, mu, sigma, input_=input_, rtol=rtol)
