"""Shared calibration of continuum models to turnover and radiocarbon.

Every model is fit the same way. For an observation (turnover T_obs and
fraction modern fm_obs), find the parameters minimizing

    ((T - T_obs) / T_obs)^2 + ((fm - fm_obs) / fm_obs)^2

by bounded least squares from several starting points. A ModelSpec holds what
differs between models: the free parameters, the space they are optimized in,
bounds, starting points, and optionally a tie-break among exact fits.

calibrate() fits each site's central turnover and its q05/q95 stock scenarios;
calibrate_sweep() repeats that over a fixed parameter (e.g. alpha). Sites are
independent, so both run in parallel over sites with n_jobs. A warm start only
ever passes a site's own earlier fit to its next fit (one chain per site), so
results do not depend on n_jobs.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import IntegrationWarning
from scipy.optimize import least_squares

from soil_diskin.continuum_models import LognormalDisKin
from soil_diskin.radiocarbon_utils import default_atmosphere

__all__ = ['ModelSpec', 'SCENARIOS', 'FIT_COLUMNS', 'exact_fm', 'fit_observation',
           'calibrate', 'calibrate_sweep', 'lognormal_spec']

# (turnover column, output suffix) for the central fit and the stock scenarios.
SCENARIOS = (('turnover', ''), ('turnover_q05', '_05'), ('turnover_q95', '_95'))
FIT_COLUMNS = ['model_T', 'model_A', 'model_fm', 'relative_turnover_error',
               'relative_fm_error', 'objective', 'calibration_status']
LARGE_RESIDUAL = 1e6


@dataclass(frozen=True)
class ModelSpec:
    """How to calibrate one model.

    Attributes:
        model_class: continuum model class, constructed as
            model_class(**free, **fixed, atm=atm).
        param_names: free parameter names, in optimization-vector order.
        lower, upper: bounds in optimization space.
        starts: (turnover, fm_obs) -> starting points in optimization space.
        fixed: constructor arguments held fixed, e.g. {'alpha': 0.5}.
        to_params, from_params: maps between the optimization vector and the
            free parameter values (default: identity).
        fast_fm: model -> fm used during optimization (default: exact_fm).
            Reported values always use exact_fm.
        prefer: free parameters -> sort key; among exact fits, the smallest
            key wins. Without it, every fit competes by objective.
        exact_tol: a fit is 'calibrated' when its relative turnover error and
            absolute fm error are both below this.
        least_squares_options: passed to scipy.optimize.least_squares.
    """
    model_class: type
    param_names: tuple
    lower: tuple
    upper: tuple
    starts: Callable
    fixed: Mapping = field(default_factory=dict)
    to_params: Callable | None = None
    from_params: Callable | None = None
    fast_fm: Callable | None = None
    prefer: Callable | None = None
    exact_tol: float = 1e-10
    least_squares_options: Mapping = field(default_factory=dict)

    def params(self, x):
        values = x if self.to_params is None else self.to_params(x)
        return dict(zip(self.param_names, (float(v) for v in values)))

    def vector(self, params):
        values = [params[name] for name in self.param_names]
        return np.asarray(values if self.from_params is None else self.from_params(*values), dtype=float)

    def model(self, x, atm):
        return self.model_class(**self.params(x), **self.fixed, atm=atm)


def exact_fm(model):
    """Steady-state fm, with a tighter age quadrature than the model default."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntegrationWarning)
        return model.calc_radiocarbon_ratio_ss(quad_limit=2000, quad_epsabs=1e-6)[0]


def _residuals(x, spec, turnover, fm_obs, atm, fm_function):
    """Relative turnover and fm errors; a large constant where the model is unusable."""
    try:
        model = spec.model(x, atm)
        if not model.params_valid() or not np.isfinite(model.T):
            return np.full(2, LARGE_RESIDUAL)
        fm = fm_function(model)
    except (ArithmeticError, ValueError):
        return np.full(2, LARGE_RESIDUAL)
    if not np.isfinite(fm):
        return np.full(2, LARGE_RESIDUAL)
    return (np.array([model.T, fm]) - [turnover, fm_obs]) / np.array([turnover, fm_obs])


def _unique_starts(spec, starts):
    """Clip starting points into bounds and drop duplicates, keeping order."""
    unique, seen = [], set()
    for start in starts:
        start = np.clip(np.asarray(start, dtype=float), spec.lower, spec.upper)
        key = tuple(np.round(start, 12))
        if key not in seen:
            seen.add(key)
            unique.append(start)
    return unique


def _empty_record(spec, status):
    """NaN values in the same column order as a fit: free, fixed, FIT_COLUMNS."""
    return {**dict.fromkeys(spec.param_names, np.nan), **spec.fixed,
            **dict.fromkeys(FIT_COLUMNS, np.nan), 'calibration_status': status}


def fit_observation(spec, turnover, fm_obs, atm=None, warm_start=None):
    """Fit one observation; returns a dict of parameters and FIT_COLUMNS.

    warm_start: free parameter values (dict) tried before spec.starts.
    """
    if not np.isfinite([turnover, fm_obs]).all() or min(turnover, fm_obs) <= 0:
        raise ValueError('positive finite turnover and fm required')
    atm = default_atmosphere() if atm is None else atm
    fast_fm = spec.fast_fm or exact_fm
    starts = [] if warm_start is None else [spec.vector(warm_start)]
    starts += list(spec.starts(turnover, fm_obs))

    best, best_key = None, (1, np.inf)
    for start in _unique_starts(spec, starts):
        try:
            fit = least_squares(_residuals, start, args=(spec, turnover, fm_obs, atm, fast_fm),
                                bounds=(spec.lower, spec.upper), **spec.least_squares_options)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        exact = (fit.success and abs(fit.fun[0]) < spec.exact_tol
                 and abs(fit.fun[1] * fm_obs) < spec.exact_tol)
        key = (0, spec.prefer(spec.params(fit.x))) if exact and spec.prefer else (1, float(fit.fun @ fit.fun))
        if key < best_key:
            best, best_key = fit, key
    if best is None or not best.success:
        return _empty_record(spec, 'fit_failed')

    model = spec.model(best.x, atm)
    model_fm = exact_fm(model)
    relative_turnover_error = (model.T - turnover) / turnover
    relative_fm_error = (model_fm - fm_obs) / fm_obs
    exact = abs(relative_turnover_error) < spec.exact_tol and abs(model_fm - fm_obs) < spec.exact_tol
    return {**spec.params(best.x), **spec.fixed,
            'model_T': model.T, 'model_A': model.A, 'model_fm': model_fm,
            'relative_turnover_error': relative_turnover_error,
            'relative_fm_error': relative_fm_error,
            'objective': relative_turnover_error**2 + relative_fm_error**2,
            'calibration_status': 'calibrated' if exact else 'approximate_fit'}


def _fit_site(specs, turnovers, fm_obs, atm, warm_start_scenarios, initial):
    """Fit one site under each spec in turn: a chain of warm starts.

    turnovers maps scenario suffix -> turnover (NaN when missing). The central
    fit under each spec starts from the previous spec's central fit (or
    initial); scenarios start from the same spec's central fit if
    warm_start_scenarios.
    """
    records, previous = [], initial
    for spec in specs:
        by_suffix = {}
        for suffix, turnover in turnovers.items():
            if not np.isfinite([turnover, fm_obs]).all() or min(turnover, fm_obs) <= 0:
                by_suffix[suffix] = _empty_record(spec, 'invalid_input')
                continue
            central = by_suffix.get('')
            if suffix and warm_start_scenarios and central and central['calibration_status'] != 'fit_failed':
                warm = central
            else:
                warm = None if suffix else previous
            by_suffix[suffix] = fit_observation(spec, turnover, fm_obs, atm, warm)
        if by_suffix[''].get('calibration_status') not in ('fit_failed', 'invalid_input'):
            previous = by_suffix['']
        records.append(by_suffix)
    return records


def _fit_sites(specs, sites, atm, n_jobs, warm_start_scenarios, initial):
    atm = default_atmosphere() if atm is None else atm
    if any(spec.model_class.log_rate_density is not None for spec in specs):
        atm.log_rate_response()  # Tabulate once here rather than in every worker.
    tasks = []
    for index, row in sites.iterrows():
        turnovers = {suffix: float(row[column]) if column in sites else np.nan
                     for column, suffix in SCENARIOS}
        warm = None if initial is None or index not in initial.index else initial.loc[index].to_dict()
        tasks.append(delayed(_fit_site)(specs, turnovers, float(row['fm_obs']), atm,
                                        warm_start_scenarios, warm))
    return Parallel(n_jobs=n_jobs)(tasks)


def _to_frame(records, index):
    """One row per site, with scenario suffixes on every column."""
    frames = [pd.DataFrame([site[suffix] for site in records], index=index).add_suffix(suffix)
              for _, suffix in SCENARIOS]
    return pd.concat(frames, axis=1)


def calibrate(spec, sites, atm=None, n_jobs=1, warm_start_scenarios=False, initial=None):
    """Fit every site's central turnover and stock scenarios.

    sites needs fm_obs and turnover; turnover_q05/turnover_q95 are optional.
    Rows that cannot be fit are kept, with calibration_status 'invalid_input'
    or 'fit_failed' and NaN values. initial optionally gives per-site free
    parameters (indexed like sites) to warm-start the central fits.

    Returns a frame indexed like sites with the parameters and FIT_COLUMNS for
    each scenario, suffixed '', '_05' and '_95'.
    """
    records = _fit_sites([spec], sites, atm, n_jobs, warm_start_scenarios, initial)
    return _to_frame([site[0] for site in records], sites.index)


def calibrate_sweep(spec_for, values, sites, name='alpha', atm=None, n_jobs=1,
                    warm_start_scenarios=True):
    """calibrate() for each fixed value, in order, warm-starting along the sweep.

    spec_for: value -> ModelSpec. Each site's central fit at one value starts
    from its fit at the previous value. Returns a long frame with one row per
    (value, site): leading columns name and site_index, where site_index is
    the row label in sites.
    """
    specs = [spec_for(value) for value in values]
    records = _fit_sites(specs, sites, atm, n_jobs, warm_start_scenarios, None)
    frames = []
    for i, value in enumerate(values):
        frame = _to_frame([site[i] for site in records], sites.index).rename_axis('site_index').reset_index()
        frame.insert(0, name, value)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def lognormal_spec():
    """ln k ~ N(mu, sigma), fit directly in (mu, sigma).

    Starts at sigma = 2.5, 1 and 4, with mu chosen to match the observed
    turnover. When several fits match both observations exactly (a fast site
    can match post-bomb radiocarbon two ways), the highest sigma is kept.
    Within these bounds the atmosphere's log-rate grid resolves the
    distribution (see AtmC14.LOG_RATE_GRID).
    """
    def starts(turnover, fm_obs):
        return [[sigma**2 / 2 - np.log(turnover), sigma] for sigma in (2.5, 1., 4.)]

    return ModelSpec(
        model_class=LognormalDisKin, param_names=('mu', 'sigma'),
        lower=(-15., .01), upper=(10., 5.), starts=starts,
        prefer=lambda params: -params['sigma'],
        least_squares_options=dict(x_scale='jac', max_nfev=2000, ftol=1e-13, xtol=1e-13, gtol=1e-13))
