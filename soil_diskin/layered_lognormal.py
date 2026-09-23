"""Independent log-normal soil layers: input allocation, prediction, and fitting.

Units: depth in cm, time in years, stocks in kg C/m², inputs in kg C/m²/year.
mu and sigma describe log(k) in the INPUT; resident carbon has mean mu-sigma².
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from .lognormal import C14_MEAN_LIFE, inner_integral
from .radiocarbon_utils import AtmC14

Array = NDArray[np.float64]
N_LAYERS, DZ = 10, 10.0
MU_BOUNDS, SIGMA_BOUNDS = (-15., 10.), (.05, 5.)


def input_weights(input_depth: float, *, surface_fraction: float = 0.) -> Array:
    """Allocate a direct top-layer fraction plus an exponential over all ten layers."""
    if not np.isfinite(input_depth) or input_depth <= 0:
        raise ValueError('input_depth must be finite and positive')
    if not np.isfinite(surface_fraction) or not 0 <= surface_fraction < 1:
        raise ValueError('surface_fraction must be finite and in [0, 1)')
    tops = np.arange(N_LAYERS)*DZ
    weights = (np.exp(-tops/input_depth)*-np.expm1(-DZ/input_depth)
               / -np.expm1(-N_LAYERS*DZ/input_depth))
    weights *= 1-surface_fraction
    weights[0] += surface_fraction
    return weights


@dataclass
class Prediction:
    stock: float
    fm: float
    times: Array
    fnew: Array


class LayerLognormal:
    """One independent layer, reusing the atmospheric response across all fits.

    A normal density on log(k) is integrated on a fixed grid. The grid covers
    12 standard deviations of every allowed resident-carbon distribution.
    Halving log_rate_step provides an independent numerical accuracy check.
    """

    def __init__(self, atmosphere: AtmC14, *, log_rate_step: float = .05,
                 mu_bounds: tuple[float, float] = MU_BOUNDS,
                 sigma_bounds: tuple[float, float] = SIGMA_BOUNDS):
        self.mu_bounds, self.sigma_bounds = mu_bounds, sigma_bounds
        bounds = np.asarray([mu_bounds, sigma_bounds])
        if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] >= bounds[:, 1]) or sigma_bounds[0] <= 0):
            raise ValueError('ordered finite bounds and positive sigma required')
        if not np.isfinite(log_rate_step) or not 0 < log_rate_step <= sigma_bounds[0]:
            raise ValueError('log_rate_step must be positive and <= minimum sigma')
        ages, fm = atmosphere.ages, atmosphere.fm
        if (ages.ndim != 1 or not len(ages) or fm.shape != ages.shape
                or not np.isfinite(ages).all() or not np.isfinite(fm).all()
                or ages[0] != 0 or np.any(np.diff(ages) <= 0) or np.any(fm < 0)
                or not np.isfinite(atmosphere.mean_R) or atmosphere.mean_R < 0):
            raise ValueError('atmosphere needs increasing ages from zero and finite nonnegative fm')
        lo = mu_bounds[0]-sigma_bounds[1]**2-12*sigma_bounds[1]
        hi = mu_bounds[1]+12*sigma_bounds[1]
        if lo < -600 or hi > 600 or (hi-lo)/log_rate_step > 100_000:
            raise ValueError('quadrature range or size exceeds numerical limits')
        self.log_rates = np.linspace(lo, hi, int(np.ceil((hi-lo)/log_rate_step))+1)
        self.step = float(self.log_rates[1]-self.log_rates[0])
        self.rates = np.exp(self.log_rates)
        # A rate class's fraction modern: k ∫ F_atm(age) exp(-(k+lambda) age) d(age).
        self.radio_response = np.array([k*inner_integral(atmosphere, k+1/C14_MEAN_LIFE)
                                        for k in self.rates])

    def predict(self, mu: float, sigma: float, input_rate: float,
                times: tuple[float, ...] | Array = ()) -> Prediction:
        """Steady stock, historical radiocarbon, and new-carbon fractions."""
        times = np.asarray(times, dtype=float)
        if (not np.isfinite([mu, sigma, input_rate]).all() or input_rate <= 0
                or not self.mu_bounds[0] <= mu <= self.mu_bounds[1]
                or not self.sigma_bounds[0] <= sigma <= self.sigma_bounds[1]):
            raise ValueError('mu/sigma must be within bounds and input_rate positive')
        if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError('times must be a finite nonnegative one-dimensional array')
        stock = input_rate*np.exp(-mu+sigma**2/2)
        # Weighting the input distribution by 1/k shifts its normal mean by -sigma².
        z = (self.log_rates-(mu-sigma**2))/sigma
        weights = np.exp(-.5*z*z)*self.step/(sigma*np.sqrt(2*np.pi))
        weights[[0, -1]] *= .5
        fm = float(np.sum(weights*self.radio_response))
        with np.errstate(over='ignore'):
            fnew = np.sum(weights*(-np.expm1(-times[:, None]*self.rates)), axis=1)
        if (not np.isfinite(stock) or stock <= 0 or not np.isfinite(fm) or fm < 0
                or not np.isfinite(fnew).all() or np.any(fnew < 0) or np.any(fnew > 1+1e-8)):
            raise FloatingPointError('nonfinite or unphysical prediction')
        return Prediction(float(stock), fm, times, fnew)


def fit_layer(model: LayerLognormal, stock: float, fm: float, input_rate: float, *,
              max_nfev: int = 500, stock_relative_scale: float = .1,
              fm_scale: float = .02) -> list[dict]:
    """Fit mu/sigma to stock and radiocarbon; return all three starts, best first.

    Fixed starting sigmas make runs reproducible. Initialization uses the exact
    no-transport identity stock/input = exp(-mu+sigma²/2). The two observations
    still enter the same scaled least-squares objective as the previous model.
    Observed f_new is deliberately absent from this interface.
    """
    if (not np.isfinite([stock, fm, input_rate, stock_relative_scale, fm_scale]).all()
            or min(stock, input_rate, stock_relative_scale, fm_scale) <= 0
            or not isinstance(max_nfev, int) or max_nfev < 1):
        raise ValueError('finite observations, positive stock/input/scales, and positive max_nfev required')
    lower = np.array([model.mu_bounds[0], model.sigma_bounds[0]])
    upper = np.array([model.mu_bounds[1], model.sigma_bounds[1]])
    scales = np.array([stock_relative_scale*stock, fm_scale])

    def residual(parameters):
        prediction = model.predict(*parameters, input_rate)
        return (np.array([prediction.stock, prediction.fm])-[stock, fm])/scales

    candidates = []
    for start_id, sigma0 in enumerate([2.5, 1., 4.]):
        sigma0 = np.clip(sigma0, lower[1], upper[1])
        mu0 = sigma0**2/2 + np.log(input_rate)-np.log(stock)
        row: dict = {'start_id': start_id, 'mu': np.nan, 'sigma': np.nan,
               'stock_pred_kg_m2': np.nan, 'fm_pred': np.nan,
               'stock_scaled_residual': np.nan, 'fm_scaled_residual': np.nan,
               'success': False, 'objective': np.inf, 'nfev': 0,
               'mu_at_bound': False, 'sigma_at_bound': False, 'jacobian_rank': 0}
        try:
            fit = least_squares(residual, np.clip([mu0, sigma0], lower, upper),
                                bounds=(lower, upper), x_scale='jac', max_nfev=max_nfev,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            pred = model.predict(float(fit.x[0]), float(fit.x[1]), input_rate)
            singular = np.linalg.svd(fit.jac*(upper-lower), compute_uv=False)
            at_bound = np.minimum(fit.x-lower, upper-fit.x)/(upper-lower) < 1e-5
            row.update(mu=fit.x[0], sigma=fit.x[1], stock_pred_kg_m2=pred.stock,
                       fm_pred=pred.fm, stock_scaled_residual=fit.fun[0],
                       fm_scaled_residual=fit.fun[1], success=bool(fit.success),
                       objective=float(fit.fun@fit.fun), nfev=fit.nfev,
                       mu_at_bound=bool(at_bound[0]), sigma_at_bound=bool(at_bound[1]),
                       jacobian_rank=int(np.count_nonzero(singular > singular[0]*1e-8)),
                       message=fit.message)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            row['message'] = str(error)
        candidates.append(row)
    candidates.sort(key=lambda row: row['objective'])
    for index, row in enumerate(candidates):
        row['candidate_id'] = index
        row['near_best'] = row['success'] and row['objective'] <= candidates[0]['objective']*1.01+1e-6
    return candidates
