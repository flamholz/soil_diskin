"""Ten-layer log-normal input model; cm, years, and kg C / m² throughout.

Transport preserves decomposition rate and closes both column boundaries.
See docs/notes/modeling/layered_lognormal_design.md for the governing equations.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .lognormal import C14_MEAN_LIFE
from .radiocarbon_utils import AtmC14

Array = NDArray[np.float64]
N_LAYERS = 10
DZ = 10.0
MU_BOUNDS = (-15.0, 10.0)
SIGMA_BOUNDS = (0.05, 5.0)


def _expm(matrix: Array) -> Array:
    # macOS BLAS can emit spurious matmul floating-point warnings inside expm.
    # Check its output explicitly so real numerical failure is never hidden.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        result = expm(matrix)
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite matrix exponential; check transport and time scales')
    return result


def transport_matrix(diffusion: float, velocity: float) -> Array:
    """Conservative operator on layer *stocks*, with downward velocity >= 0."""
    if not np.isfinite([diffusion, velocity]).all() or min(diffusion, velocity) < 0:
        raise ValueError('diffusion and velocity must be finite and nonnegative')
    down = diffusion / DZ**2 + velocity / DZ
    up = diffusion / DZ**2
    operator = np.zeros((N_LAYERS, N_LAYERS))
    for i in range(N_LAYERS - 1):
        operator[i + 1, i] += down
        operator[i, i] -= down
        operator[i, i + 1] += up
        operator[i + 1, i + 1] -= up
    return operator


def input_weights(input_depth: float) -> Array:
    """Integrated exponential inputs, normalized over the 100 cm column."""
    if not np.isfinite(input_depth) or input_depth <= 0:
        raise ValueError('input_depth must be finite and positive')
    z = np.arange(N_LAYERS) * DZ
    return (np.exp(-z / input_depth) * -np.expm1(-DZ / input_depth)
            / -np.expm1(-N_LAYERS * DZ / input_depth))


def _resolvent(rates: Array, diffusion: float, velocity: float) -> Array:
    """Return (k I - T)^-1 without subtracting nearly equal slow-rate pivots.

    The final Thomas pivot is O(k). Computing it by subtraction loses it when
    k is much smaller than transport. Carry its positive recurrence instead.
    Solve for k I first, giving bounded absorption probabilities, then / k.
    This also works for pure advection and zero transport.
    """
    down = diffusion / DZ**2 + velocity / DZ
    up = diffusion / DZ**2
    rhs = rates[:, None, None] * np.broadcast_to(np.eye(N_LAYERS),
                                                (len(rates), N_LAYERS, N_LAYERS))
    pivots = np.empty((len(rates), N_LAYERS))
    delta = rates.copy()
    for i in range(N_LAYERS - 1):
        pivots[:, i] = down + delta
        rhs[:, i + 1] += (down / pivots[:, i])[:, None] * rhs[:, i]
        delta = rates + up * (delta / pivots[:, i])
    pivots[:, -1] = delta
    rhs[:, -1] /= delta[:, None]
    for i in range(N_LAYERS - 2, -1, -1):
        rhs[:, i] = (rhs[:, i] + up * rhs[:, i + 1]) / pivots[:, i, None]
    return rhs / rates[:, None, None]


@dataclass
class Prediction:
    """Layer stocks and fraction modern; fnew has shape (times, layers)."""

    stocks: Array
    fm: Array
    times: Array
    fnew: Array


class LayeredLognormal:
    """Reusable response kernels for one supplied (D, v, h) triple.

    The uniform log-rate quadrature covers twelve normal standard deviations
    around both input and stock-weighted distributions, including the shifted
    slow-rate tail. ``log_rate_step`` must not exceed the minimum sigma.
    Halve it to check convergence independently of optimization.
    """

    def __init__(self, diffusion: float, velocity: float, input_depth: float,
                 atmosphere: AtmC14, *, log_rate_step: float = 0.05,
                 mu_bounds: tuple[float, float] = MU_BOUNDS,
                 sigma_bounds: tuple[float, float] = SIGMA_BOUNDS):
        self.transport = transport_matrix(diffusion, velocity)
        self.weights = input_weights(input_depth)
        self.diffusion, self.velocity, self.input_depth = diffusion, velocity, input_depth
        self.mu_bounds, self.sigma_bounds = mu_bounds, sigma_bounds
        bounds = np.asarray([mu_bounds, sigma_bounds], dtype=float)
        if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] >= bounds[:, 1]) or sigma_bounds[0] <= 0):
            raise ValueError('ordered finite bounds and positive sigma are required')
        if not np.isfinite(log_rate_step) or not 0 < log_rate_step <= sigma_bounds[0]:
            raise ValueError('log_rate_step must be positive and <= minimum sigma')
        lo = mu_bounds[0] - sigma_bounds[1]**2 - 12*sigma_bounds[1]
        hi = mu_bounds[1] + 12*sigma_bounds[1]
        if lo < -600 or hi > 600 or (hi-lo)/log_rate_step > 100_000:
            raise ValueError('quadrature range or size exceeds supported numerical limits')
        self.u = np.linspace(lo, hi, int(np.ceil((hi-lo)/log_rate_step)) + 1)
        self.log_rate_step = float(self.u[1] - self.u[0])
        self.rates = np.exp(self.u)
        self._carbon = _resolvent(self.rates, diffusion, velocity)
        self._radio = self._radiocarbon_kernel(atmosphere)

    def observables_and_jacobian(self, parameters: Array, npp: float) -> tuple[Array, Array]:
        """Return [stocks, fm] and its Jacobian for [ten mu, ten sigma]."""
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (2*N_LAYERS,):
            raise ValueError('parameters must contain ten mu followed by ten sigma')
        mu, sigma = parameters[:N_LAYERS], parameters[N_LAYERS:]
        density = self._input_density(mu, sigma, npp)
        z = (self.u[:, None]-mu)/sigma
        derivatives = [density*z/sigma, density*(z*z-1)/sigma]
        stocks = np.einsum('kij,kj->i', self._carbon, density, optimize=True)
        radio = np.einsum('kij,kj->i', self._radio, density, optimize=True)
        if not np.isfinite(stocks).all() or np.any(stocks <= 0):
            raise FloatingPointError('modeled layer stock is nonpositive or nonfinite')
        stock_jac = np.concatenate([np.einsum('kij,kj->ij', self._carbon, d,
                                             optimize=True) for d in derivatives], axis=1)
        radio_jac = np.concatenate([np.einsum('kij,kj->ij', self._radio, d,
                                             optimize=True) for d in derivatives], axis=1)
        fm = radio/stocks
        fm_jac = (radio_jac-fm[:, None]*stock_jac)/stocks[:, None]
        return np.r_[stocks, fm], np.vstack([stock_jac, fm_jac])

    def _radiocarbon_kernel(self, atmosphere: AtmC14) -> Array:
        ages = np.asarray(atmosphere.ages, dtype=float)
        values = np.asarray(atmosphere.fm, dtype=float)
        if (ages.ndim != 1 or len(ages) == 0 or values.shape != ages.shape
                or not np.isfinite(ages).all() or not np.isfinite(values).all()
                or ages[0] != 0 or np.any(np.diff(ages) <= 0)
                or np.any(values < 0) or not np.isfinite(atmosphere.mean_R)
                or atmosphere.mean_R < 0):
            raise ValueError('atmosphere needs increasing ages from zero and finite nonnegative fm')
        # Match AtmC14 exactly: last knot begins the constant mean_R tail.
        levels = np.r_[values[:-1], atmosphere.mean_R]
        jumps = np.diff(levels)
        keep = jumps != 0
        jump_ages, jumps = ages[1:][keep], jumps[keep]
        lam = 1 / C14_MEAN_LIFE
        alpha = self.rates + lam
        response = levels[0] * _resolvent(alpha, self.diffusion, self.velocity)
        if len(jumps) == 0:
            return response

        # Below this threshold replacing k+lambda by lambda changes a positive
        # Laplace integral by negligible relative error; upper rates have no
        # measurable contribution from even the youngest atmospheric jump.
        low = self.rates < lam * 1e-13
        active = (~low) & (alpha * jump_ages[0] < 40)
        indices = np.flatnonzero(active)
        alphas = np.r_[lam, alpha[indices]]
        if not np.any(self.transport):
            for start in range(0, len(alphas), 64):
                a = alphas[start:start + 64]
                integral = (levels[0] + (np.exp(-a[:, None]*jump_ages)*jumps).sum(axis=1)) / a
                matrices = integral[:, None, None] * np.eye(N_LAYERS)
                if start == 0:
                    response[low] = matrices[0]
                    response[indices[:len(a)-1]] = matrices[1:]
                else:
                    response[indices[start-1:start+len(a)-1]] = matrices
            return response

        transitions = np.empty((len(jumps), N_LAYERS, N_LAYERS))
        current = np.eye(N_LAYERS)
        previous_age = 0.0
        steps: dict[float, Array] = {}
        for i, age in enumerate(jump_ages):
            step = float(age - previous_age)
            if step not in steps:
                steps[step] = _expm(self.transport * step)
            current = steps[step] @ current
            transitions[i] = current
            previous_age = float(age)
        for start in range(0, len(alphas), 64):
            a = alphas[start:start + 64]
            factors = np.exp(-a[:, None]*jump_ages) * jumps
            boundary = levels[0]*np.eye(N_LAYERS) + np.einsum(
                'ka,aij->kij', factors, transitions, optimize=True)
            matrices = _resolvent(a, self.diffusion, self.velocity) @ boundary
            if start == 0:
                response[low] = matrices[0]
                response[indices[:len(a)-1]] = matrices[1:]
            else:
                response[indices[start-1:start+len(a)-1]] = matrices
        return response

    def _input_density(self, mu: Array, sigma: Array, npp: float) -> Array:
        mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
        if (mu.shape != (N_LAYERS,) or sigma.shape != (N_LAYERS,)
                or not np.isfinite(mu).all() or not np.isfinite(sigma).all()
                or np.any(mu < self.mu_bounds[0]) or np.any(mu > self.mu_bounds[1])
                or np.any(sigma < self.sigma_bounds[0]) or np.any(sigma > self.sigma_bounds[1])):
            raise ValueError('mu and sigma must each have ten finite values inside model bounds')
        if not np.isfinite(npp) or npp <= 0:
            raise ValueError('npp must be finite and positive (kg C/m²/yr)')
        z = (self.u[:, None] - mu) / sigma
        density = np.exp(-0.5*z*z) / (np.sqrt(2*np.pi)*sigma)
        density *= self.weights*npp*self.log_rate_step
        density[[0, -1]] *= 0.5
        return density

    @lru_cache(maxsize=4)
    def _new_carbon_kernel(self, time: float) -> Array:
        if time == 0:
            return np.zeros_like(self._carbon)
        if not np.any(self.transport):
            return (-np.expm1(-self.rates*time)/self.rates)[:, None, None]*np.eye(N_LAYERS)
        result = np.empty_like(self._carbon)
        kt = self.rates*time
        high = kt > 40
        result[high] = self._carbon[high]
        regular = (kt >= 0.01) & ~high
        transition = _expm(self.transport*time)
        result[regular] = self._carbon[regular] - np.exp(-kt[regular, None, None]) * (
            transition @ self._carbon[regular])
        # Block exponential integrates the source without cancellation at k*t≈0.
        block = np.zeros((2*N_LAYERS, 2*N_LAYERS))
        block[:N_LAYERS, N_LAYERS:] = np.eye(N_LAYERS)
        block[:N_LAYERS, :N_LAYERS] = self.transport
        small = kt < 1e-11
        result[small] = _expm(block*time)[:N_LAYERS, N_LAYERS:]
        for i in np.flatnonzero(~small & ~regular & ~high):
            block[:N_LAYERS, :N_LAYERS] = self.transport - self.rates[i]*np.eye(N_LAYERS)
            result[i] = _expm(block*time)[:N_LAYERS, N_LAYERS:]
        return result

    def predict(self, mu: Array, sigma: Array, npp: float,
                times: Array | tuple[float, ...] = ()) -> Prediction:
        """Predict stocks, fraction modern, and new fractions at elapsed years."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError('times must be a finite nonnegative one-dimensional array')
        density = self._input_density(mu, sigma, npp)
        stocks = np.einsum('kij,kj->i', self._carbon, density, optimize=True)
        if not np.isfinite(stocks).all() or np.any(stocks <= 0):
            raise FloatingPointError('modeled layer stock is nonpositive or nonfinite')
        radio = np.einsum('kij,kj->i', self._radio, density, optimize=True)
        new = np.array([np.einsum('kij,kj->i', self._new_carbon_kernel(float(t)),
                                 density, optimize=True) for t in times]).reshape(-1, N_LAYERS)
        return Prediction(stocks, radio/stocks, times, new/stocks)
