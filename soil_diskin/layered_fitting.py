"""Conditional profile fits; new-carbon observations never enter this module."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .layered_lognormal import Array, LayeredLognormal, N_LAYERS, Prediction


@dataclass(frozen=True)
class FitSettings:
    n_starts: int = 4
    seed: int = 0
    max_nfev: int = 500
    stock_relative_scale: float = 0.10
    fm_scale: float = 0.02
    near_relative: float = 0.01
    near_absolute: float = 1e-6
    distinct_distance: float = 1e-3

    def __post_init__(self) -> None:
        if (not isinstance(self.n_starts, int) or self.n_starts < 1
                or not isinstance(self.max_nfev, int) or self.max_nfev < 1):
            raise ValueError('n_starts and max_nfev must be positive integers')
        scales = [self.stock_relative_scale, self.fm_scale, self.distinct_distance]
        tolerances = [self.near_relative, self.near_absolute]
        if (not np.isfinite(scales+tolerances).all() or min(scales) <= 0
                or min(tolerances) < 0):
            raise ValueError('scales must be positive and near-fit tolerances nonnegative')


@dataclass
class FitCandidate:
    mu: Array
    sigma: Array
    prediction: Prediction
    residuals: Array
    objective: float
    success: bool
    message: str
    start_index: int
    nfev: int
    optimality: float
    at_bounds: Array
    singular_values: Array
    jacobian_rank: int
    near_best: bool = False


@dataclass
class ProfileFit:
    candidates: list[FitCandidate]
    attempts: list[dict] = field(default_factory=list)

    @property
    def best(self) -> FitCandidate:
        """Lowest objective found, which may still have success=False."""
        if not self.candidates:
            raise ValueError('no evaluable candidate was found; inspect attempts')
        return self.candidates[0]


def fit_profile(model: LayeredLognormal, stocks: Array, fm: Array, npp: float, *,
                settings: FitSettings | None = None,
                initial_parameters: Array | None = None) -> ProfileFit:
    """Fit 20 local parameters for fixed shared hyperparameters.

    Scales define a working objective, not a measurement-error likelihood.
    Repeated starting points are recorded even when their solutions coincide.
    Failed optimization attempts do not prevent other starts from running.
    """
    settings = settings or FitSettings()
    stocks, fm = np.asarray(stocks, dtype=float), np.asarray(fm, dtype=float)
    if (stocks.shape != (N_LAYERS,) or fm.shape != (N_LAYERS,)
            or not np.isfinite(stocks).all() or not np.isfinite(fm).all()
            or np.any(stocks <= 0)
            or not np.isfinite(npp) or npp <= 0):
        raise ValueError('ten positive stocks, ten finite fm values, and positive NPP required')
    lower = np.r_[np.full(N_LAYERS, model.mu_bounds[0]),
                  np.full(N_LAYERS, model.sigma_bounds[0])]
    upper = np.r_[np.full(N_LAYERS, model.mu_bounds[1]),
                  np.full(N_LAYERS, model.sigma_bounds[1])]
    span = upper-lower
    scales = np.r_[settings.stock_relative_scale*stocks, np.full(N_LAYERS, settings.fm_scale)]
    observed = np.r_[stocks, fm]
    if initial_parameters is None:
        sigma0 = np.clip(2.5, model.sigma_bounds[0], model.sigma_bounds[1])
        mu0 = sigma0**2/2 + np.log(np.maximum(npp*model.weights, 1e-300))-np.log(stocks)
        initial = np.clip(np.r_[mu0, np.full(N_LAYERS, sigma0)], lower, upper)
    else:
        initial = np.asarray(initial_parameters, dtype=float)
        if (initial.shape != (2*N_LAYERS,) or not np.isfinite(initial).all()
                or np.any(initial < lower) or np.any(initial > upper)):
            raise ValueError('initial_parameters must contain twenty values within model bounds')
    rng = np.random.default_rng(settings.seed)
    starts = [initial]
    for i in range(1, settings.n_starts):
        if i % 2 == 0:
            starts.append(rng.uniform(lower, upper))
        else:
            jitter = rng.normal(size=2*N_LAYERS)*np.r_[np.full(N_LAYERS, 1.5),
                                                     np.full(N_LAYERS, 0.5)]
            starts.append(np.clip(initial+jitter, lower, upper))

    cached_x: Array | None = None
    cached_residual: Array | None = None
    cached_jacobian: Array | None = None

    def evaluate(x: Array) -> tuple[Array, Array]:
        nonlocal cached_x, cached_residual, cached_jacobian
        if cached_x is None or not np.array_equal(x, cached_x):
            values, jac = model.observables_and_jacobian(x, npp)
            cached_x = x.copy()
            cached_residual = (values-observed)/scales
            cached_jacobian = jac/scales[:, None]
        assert cached_residual is not None and cached_jacobian is not None
        return cached_residual, cached_jacobian

    candidates: list[FitCandidate] = []
    attempts: list[dict] = []
    for index, start in enumerate(starts):
        try:
            fit = least_squares(lambda x: evaluate(x)[0], start,
                                jac=lambda x: evaluate(x)[1], bounds=(lower, upper),
                                x_scale='jac', max_nfev=settings.max_nfev,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            residual, jac = evaluate(fit.x)
            singular = np.linalg.svd(jac*span, compute_uv=False)
            rank = int(np.count_nonzero(singular > singular[0]*1e-8))
            candidate = FitCandidate(
                fit.x[:N_LAYERS].copy(), fit.x[N_LAYERS:].copy(),
                model.predict(fit.x[:N_LAYERS], fit.x[N_LAYERS:], npp),
                residual.copy(), float(residual@residual), bool(fit.success), str(fit.message),
                index, int(fit.nfev), float(fit.optimality),
                np.minimum((fit.x-lower)/span, (upper-fit.x)/span) < 1e-5,
                singular, rank)
            candidates.append(candidate)
            attempts.append({'start_index': index, 'success': candidate.success,
                             'objective': candidate.objective, 'nfev': candidate.nfev,
                             'message': candidate.message})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            attempts.append({'start_index': index, 'success': False, 'message': str(error)})

    distinct: list[FitCandidate] = []
    for candidate in sorted(candidates, key=lambda c: c.objective):
        x = np.r_[candidate.mu, candidate.sigma]
        if not any(np.max(np.abs((x-np.r_[c.mu, c.sigma])/span)) < settings.distinct_distance
                   for c in distinct):
            distinct.append(candidate)
    if distinct:
        threshold = distinct[0].objective*(1+settings.near_relative)+settings.near_absolute
        for candidate in distinct:
            candidate.near_best = candidate.success and candidate.objective <= threshold
    return ProfileFit(distinct, attempts)
