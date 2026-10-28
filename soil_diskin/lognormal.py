"""Closed-form calculations for the lognormal ("Diskin") continuum model.

The model: carbon enters the soil at rate I and each parcel is assigned, once,
a first-order decay rate k drawn from a lognormal distribution, ln k ~ N(mu, sigma).
Equivalently, the soil is a continuum of independent pools, one per k, each
receiving the fraction p(k) dk of the input. A pool with rate k holds I p(k)/k
at steady state, so slow pools dominate the stock even when they receive
little of the input.

Two summary times fix (mu, sigma) at steady-state:
    T = E[1/k]               = exp(-mu + sigma^2/2)  (turnover: stock / input)
    A = mean age of the stock = T exp(sigma^2)
so sigma^2 = ln(A/T) and mu = sigma^2/2 - ln T. The functions here that take
(tau, age) use these relations (written as mu = -ln sqrt(tau^3/age)).

The same model is LognormalDisKin in continuum_models.py, which is what the
calibration (soil_diskin.calibration) uses; LognormalDisKin.cdfA calls
diskin_C_of_t below. The remaining functions are used by the sensitivity
analyses (04b, 06a, 06b), figures 1 and 2, and the tests.
"""
from __future__ import annotations

import numpy as np
from math import sqrt, log, exp
from scipy.integrate import quad

from .radiocarbon_utils import AtmC14

__all__ = [
    "lognormal_radiocarbon",
    "scan_ages",
    "diskin_C_of_t",
    "run_diskin_fast",
]


# 14C mean life, 1/lambda, in years; the same value as constants.LAMBDA_14C.
C14_MEAN_LIFE = 8267.0  # years
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def lognormal_radiocarbon(
    atm: AtmC14,
    tau: float,
    age: float,
    rtol: float = 1e-4,
) -> float:
    """Steady-state radiocarbon ratio (14C/12C relative to standard) of the whole soil.

    Each pool with rate k is at steady state under a constant input whose 14C
    ratio is the atmospheric record R(a), a years before present. Material that
    entered a years ago has decayed by exp(-k a) (all carbon) and by a further
    exp(-lambda a) (14C only), so per unit input the pool holds
        12C:  1/k
        14C:  integral of R(a) exp(-(k + lambda) a) da = I(k + lambda),
    where I is the Laplace transform of the atmospheric record (AtmC14.laplace).
    Summing over pools and dividing the 14C stock by the 12C stock gives
        fm = integral of p(k) I(k + lambda) dk / integral of p(k)/k dk
           = E[I(k + lambda)] / T,
    the expectation taken over the input's lognormal rate distribution. The
    integral is evaluated in u = ln k over mu +/- 10 sigma by adaptive
    quadrature; 1/T = exp(mu - sigma^2/2). Treating 14C as a trace isotope,
    this ratio is the fraction modern that would be measured on the bulk soil.

    LognormalDisKin.calc_radiocarbon_ratio_ss computes the same quantity from a
    precomputed table of I on a fixed grid in u, which is faster and is what
    the calibration uses. This function is used by scan_ages (and so 06a) and by
    the Mathematica comparison test.

    Parameters
    ----------
    atm: AtmC14
        Atmospheric 14C record.
    tau: float
        Turnover time T (stock / input) of the soil, in years.
    age: float
        Mean age A of the steady-state stock, in years; requires age > tau.
    rtol: float
        Relative tolerance of the quadrature over u.

    Returns
    -------
    float
        The steady-state radiocarbon ratio of the bulk soil.
    """
    sigma = sqrt(log(age / tau))
    mu = -log(sqrt(tau ** 3 / age))
    u_lo = mu - 10.0 * sigma
    u_hi = mu + 10.0 * sigma

    inv_sigma = 1.0 / sigma

    def integrand(u):
        # Input-weighted density of u = ln k times the 14C held per unit input.
        z = (u - mu) * inv_sigma
        phi = _INV_SQRT_2PI * np.exp(-0.5 * z * z) * inv_sigma
        return phi * atm.laplace(1.0 / C14_MEAN_LIFE + np.exp(u))

    val, _ = quad(integrand, u_lo, u_hi, epsrel=rtol, limit=200)
    # Divide the 14C stock by the 12C stock, T = E[1/k] = exp(-mu + sigma^2/2).
    return val * exp(mu - 0.5 * sigma * sigma)


def scan_ages(atm: AtmC14, tau: float, agelist, rtol: float = 1e-4) -> np.ndarray:
    """Radiocarbon ratio as a function of mean age, at a fixed turnover time.

    Evaluates lognormal_radiocarbon at each mean age in `agelist` with the
    turnover time held at `tau`. The result is a site's calibration curve:
    the measured radiocarbon ratio fixes the mean age by inverting it, which
    06a does by interpolation. The inversion assumes fm is monotonic in age
    over `agelist`, which can fail for fast sites near the bomb peak, where
    two ages can give the same fm.

    Returns
    -------
    np.ndarray
        The radiocarbon ratio at each age in `agelist`.
    """
    return np.array([lognormal_radiocarbon(atm, tau, float(a), rtol=rtol) for a in agelist])


def diskin_C_of_t(
    ts: np.ndarray,
    mu: float,
    sigma: float,
    input_: float = 1.0,
    rtol: float = 1e-10,
) -> np.ndarray:
    """Carbon stock C(t) of a pool started empty and fed a constant input.

    Starting with no carbon at t = 0, the pool with rate k fills toward I/k as
        I (1 - exp(-k t)) / k,
    so summing over the lognormal continuum of pools,
        C(t) = I integral of p(k) (1 - exp(-k t)) / k dk.
    C(0) = 0 and C(t) -> I T as t -> infinity.

    C(t) is also the amount of carbon younger than t in the steady-state soil:
    all of it entered within the last t years, exactly as in the pool started
    empty. So C(t) / (I T) is the steady-state age CDF (LognormalDisKin.cdfA),
    and its value at a labeling duration is the predicted fraction of new
    carbon, f_new.

    The integral is over u = ln k, with -expm1 for accuracy when k t is small.
    The 1/k weighting shifts the stock toward slow rates, centring it on
    mu - sigma^2, so the lower limit is mu - sigma^2 - 10 sigma. Each t is a
    separate adaptive quadrature at rtol, accurate but slow for long grids.

    Parameters
    ----------
    ts: np.ndarray
        Times (years) at which to evaluate the stock.
    mu: float
        Mean of ln k (k in 1/yr).
    sigma: float
        Standard deviation of ln k.
    input_: float
        Constant input rate I.
    rtol: float
        Relative tolerance of each quadrature.

    Returns
    -------
    np.ndarray
        C(t) at each time in `ts`.
    """
    u_lo = mu - sigma * sigma - 10.0 * sigma
    u_hi = mu + 10.0 * sigma
    out = np.empty(len(ts), dtype=np.float64)
    inv_sigma = 1.0 / sigma

    for i, t in enumerate(ts):
        def integrand(u, t=t):
            # p(u) * (1/k) * (1 - exp(-k t)) with k = e^u.
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
    """C(t) on a log-spaced time grid, for a soil given by its turnover time and mean age.

    Converts (tau, age) to (mu, sigma) and evaluates diskin_C_of_t at
    `ts_size` times log-spaced from 0.1 to `tmax` years. Dividing C by
    input_ * tau gives the steady-state age CDF on that grid, which the
    figures plot and the sensitivity analyses interpolate at labeling
    durations to get f_new. "Fast" contrasts this closed form with the ODE
    solution of the Julia code it replaced.

    Returns
    -------
    tuple of np.ndarray
        (ts, C): the time grid and C(t) on it.
    """
    sigma = sqrt(log(age / tau))
    mu = -log(sqrt(tau ** 3 / age))
    ts = np.logspace(-1.0, np.log10(tmax), ts_size)
    return ts, diskin_C_of_t(ts, mu, sigma, input_=input_, rtol=rtol)
