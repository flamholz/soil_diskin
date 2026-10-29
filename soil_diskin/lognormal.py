"""Radiocarbon calculations for the lognormal ("Diskin") continuum model.

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

The same model is LognormalDisKin in continuum_models.py, which the
calibration (soil_diskin.calibration), predictions and figures use; its
stock_younger_than and cdfA give the age distribution. The functions here are
used by the 06a sensitivity analysis and the tests.
"""
from __future__ import annotations

import numpy as np
from math import sqrt, log, exp
from scipy.integrate import quad

from .constants import LAMBDA_14C
from .radiocarbon_utils import AtmC14

__all__ = [
    "lognormal_radiocarbon",
    "scan_ages",
]


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
        return phi * atm.laplace(LAMBDA_14C + np.exp(u))

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
