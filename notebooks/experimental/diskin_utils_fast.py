"""Python port of `diskin_utils_fast.jl`.

Closed-form / 1-D quadrature evaluation of the lognormal Diskin model for a
constant input. See `notebooks/diskin_closed_form_derivation.md` for the math.

    C(t) = I * integral over u of exp(-u) * phi_N(u; mu, sigma)
                                 * (1 - exp(-exp(u) * t)) du,    u = log k
"""
import numpy as np
from math import sqrt, log, exp
from scipy.integrate import quad


_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _normal_pdf(u, mu, sigma):
    z = (u - mu) / sigma
    return _INV_SQRT_2PI * np.exp(-0.5 * z * z) / sigma


def diskin_C_of_t(ts, mu, sigma, input_=1.0, rtol=1e-10):
    """Evaluate C(t) at each t in `ts` (vectorized over t in Python loop)."""
    u_lo = mu - sigma * sigma - 10.0 * sigma
    u_hi = mu + 10.0 * sigma
    out = np.empty(len(ts), dtype=np.float64)
    for i, t in enumerate(ts):
        # -np.expm1(-x) = 1 - exp(-x), accurate for small x.
        def integrand(u, t=t):
            return np.exp(-u) * _normal_pdf(u, mu, sigma) * (-np.expm1(-np.exp(u) * t))
        val, _ = quad(integrand, u_lo, u_hi, epsrel=rtol, limit=200)
        out[i] = input_ * val
    return out


def run_diskin_fast(tau, age, input_=1.0, tmax=100.0, ts_size=1000, rtol=1e-10):
    """Returns (ts, C) on a log-spaced time grid from 0.1 to tmax."""
    sigma = sqrt(log(age / tau))
    mu = -log(sqrt(tau ** 3 / age))
    ts = np.logspace(-1.0, np.log10(tmax), ts_size)
    return ts, diskin_C_of_t(ts, mu, sigma, input_, rtol=rtol)
