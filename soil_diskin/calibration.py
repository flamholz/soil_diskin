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
from scipy.special import gamma

from soil_diskin.constants import LAMBDA_14C
from soil_diskin.continuum_models import (GeneralPowerLawDisKin, LognormalDisKin, LogUniformDisKin,
                                          PowerLawDisKin, SoftCutoffPowerLawDisKin, WeibullDisKin)
from soil_diskin.radiocarbon_utils import default_atmosphere

__all__ = ['ModelSpec', 'SCENARIOS', 'FIT_COLUMNS', 'exact_fm', 'age_grid_fm', 'fit_observation',
           'calibrate', 'calibrate_sweep', 'lognormal_spec', 'general_powerlaw_spec',
           'soft_cutoff_powerlaw_spec', 'power_law_spec', 'weibull_spec', 'loguniform_spec']

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


# Ages (yr) for age_grid_fm: dense near zero, where power-law age densities
# diverge, and out to 60 kyr.
FAST_AGES = np.unique(np.concatenate([
    [0.], np.geomspace(1e-8, 1e-2, 300), np.geomspace(1e-2, 1., 300), np.geomspace(1., 100., 600),
    np.geomspace(100., 5000., 1000), np.geomspace(5000., 60000., 1500)]))
_fast_weights = (None, None)  # (atmosphere, R(a) exp(-lambda a) on FAST_AGES), last one used


def age_grid_fm(model):
    """Fast fm: trapezoid of pA(a) R(a) exp(-lambda a) on FAST_AGES (~1e-4 accurate)."""
    global _fast_weights
    if _fast_weights[0] is not model.atm:
        _fast_weights = (model.atm, model.atm.R(FAST_AGES) * np.exp(-LAMBDA_14C * FAST_AGES))
    return float(np.trapezoid(_fast_weights[1] * model.pA(FAST_AGES), FAST_AGES))


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


def atmospheres_by_year(years, path=None):
    """Atmospheres with ages counted back from each year: year -> AtmC14, or None where
    the record ends before the year (or the year is missing). path defaults to
    constants.C14_DATA_PATH."""
    from soil_diskin import constants
    from soil_diskin.radiocarbon_utils import load_atm14c
    path = constants.C14_DATA_PATH if path is None else path
    atmospheres = {}
    for year in pd.unique(pd.Series(years).dropna()):
        try:
            atmospheres[year] = load_atm14c(path, reference_year=year)
        except ValueError:
            atmospheres[year] = None
    return atmospheres


def _unfittable(specs, turnovers, status):
    """_fit_site's records for a site that cannot be fit at all."""
    return [{suffix: _empty_record(spec, status) for suffix in turnovers} for spec in specs]


def _fit_sites(specs, sites, atm, n_jobs, warm_start_scenarios, initial, atms=None):
    """Fit each site; atms optionally gives each row its own atmosphere (None: unfittable)."""
    if atms is None:
        atms = [default_atmosphere() if atm is None else atm]*len(sites)
    if any(spec.model_class.log_rate_density is not None for spec in specs):
        for each in {id(a): a for a in atms if a is not None}.values():
            each.log_rate_response()  # Tabulate once here rather than in every worker.
    tasks = []
    for (index, row), site_atm in zip(sites.iterrows(), atms):
        turnovers = {suffix: float(row[column]) if column in sites else np.nan
                     for column, suffix in SCENARIOS}
        if site_atm is None:
            tasks.append(delayed(_unfittable)(specs, turnovers, 'no_atmosphere'))
            continue
        warm = None if initial is None or index not in initial.index else initial.loc[index].to_dict()
        tasks.append(delayed(_fit_site)(specs, turnovers, float(row['fm_obs']), site_atm,
                                        warm_start_scenarios, warm))
    return Parallel(n_jobs=n_jobs)(tasks)


def _to_frame(records, index):
    """One row per site, with scenario suffixes on every column."""
    frames = [pd.DataFrame([site[suffix] for site in records], index=index).add_suffix(suffix)
              for _, suffix in SCENARIOS]
    return pd.concat(frames, axis=1)


def calibrate(spec, sites, atm=None, n_jobs=-1, warm_start_scenarios=False, initial=None,
              year_column=None, atm_path=None):
    """Fit every site's central turnover and stock scenarios.

    sites needs fm_obs and turnover; turnover_q05/turnover_q95 are optional.
    Rows that cannot be fit are kept, with calibration_status 'invalid_input',
    'fit_failed' or 'no_atmosphere' and NaN values. initial optionally gives
    per-site free parameters (indexed like sites) to warm-start the central fits.

    By default every site uses `atm` (the project's record if None), with ages
    counted back from 2000. With year_column, each site instead uses the
    atmosphere of the year in that column (e.g. its sampling year), from the
    record at atm_path (constants.C14_DATA_PATH if None); sites sampled after the
    record ends get 'no_atmosphere'.

    Returns a frame indexed like sites with the parameters and FIT_COLUMNS for
    each scenario, suffixed '', '_05' and '_95'. n_jobs = -1 uses all cores.
    """
    atms = None
    if year_column is not None:
        by_year = atmospheres_by_year(sites[year_column], atm_path)
        atms = [by_year.get(year) for year in sites[year_column]]
    records = _fit_sites([spec], sites, atm, n_jobs, warm_start_scenarios, initial, atms)
    return _to_frame([site[0] for site in records], sites.index)


def calibrate_sweep(spec_for, values, sites, name='alpha', atm=None, n_jobs=-1,
                    warm_start_scenarios=True):
    """calibrate() for each fixed value, in order, warm-starting along the sweep.

    spec_for: value -> ModelSpec. Each site's central fit at one value starts
    from its fit at the previous value. Returns a long frame with one row per
    (value, site): leading columns name and site_index, where site_index is
    the row label in sites. If name is also a fixed model parameter, that
    column is moved to the front rather than repeated.
    """
    specs = [spec_for(value) for value in values]
    records = _fit_sites(specs, sites, atm, n_jobs, warm_start_scenarios, None)
    frames = []
    for i, value in enumerate(values):
        frame = _to_frame([site[i] for site in records], sites.index).rename_axis('site_index').reset_index()
        frame.insert(0, name, frame.pop(name) if name in frame else value)
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


# Power laws are fit in (log10 t_min, log10(t_max - t_min)), which keeps
# t_min < t_max and scales parameters spanning many decades.
POWER_LAW_STARTS = ((-4., -1.), (-3.5, 3.), (-2., 1.), (-1., 2.), (0., 2.5), (.5, 3.), (1., 3.))


def _log_timescales(x):
    t_min = 10 ** x[0]
    return t_min, t_min + 10 ** x[1]


def _log_timescales_inverse(t_min, t_max):
    return np.log10(max(t_min, 1e-8)), np.log10(max(t_max - t_min, 1e-8))


def _power_law_spec(model_class, alpha):
    """t_min and t_max at fixed alpha, searched with the fast age-grid fm.

    exact_tol is 1e-3: the age grid is ~1e-4 accurate, so fits converge to
    ~1e-4 of the observations under the exact fm reported afterwards.
    """
    return ModelSpec(
        model_class=model_class, param_names=('t_min', 't_max'), fixed={'alpha': alpha},
        lower=(-8., -8.), upper=(8., 8.), starts=lambda turnover, fm_obs: POWER_LAW_STARTS,
        to_params=_log_timescales, from_params=_log_timescales_inverse,
        fast_fm=age_grid_fm, exact_tol=1e-3,
        least_squares_options=dict(xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=150))


def general_powerlaw_spec(alpha):
    """GeneralPowerLawDisKin (hard cutoffs) at fixed alpha in (0, 1)."""
    return _power_law_spec(GeneralPowerLawDisKin, alpha)


def soft_cutoff_powerlaw_spec(alpha):
    """SoftCutoffPowerLawDisKin at fixed alpha >= 0."""
    return _power_law_spec(SoftCutoffPowerLawDisKin, alpha)


# Models without a rate-space fm or a fast approximation accurate enough for
# their fits are searched with the exact (quadrature) fm, as they were before.
EXACT_FM_TOL = 1e-6
EXACT_FM_OPTIONS = dict(x_scale='jac', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300)


def power_law_spec():
    """PowerLawDisKin (alpha = 1), in the same timescale space as the other power laws."""
    return ModelSpec(
        model_class=PowerLawDisKin, param_names=('t_min', 't_max'),
        lower=(-8., -8.), upper=(8., 8.),
        starts=lambda turnover, fm_obs: POWER_LAW_STARTS + ((np.log10(.5), 4.),),
        to_params=_log_timescales, from_params=_log_timescales_inverse,
        exact_tol=EXACT_FM_TOL, least_squares_options=EXACT_FM_OPTIONS)


# Shapes to start the Weibull fit from, each with k matching the turnover.
WEIBULL_STARTING_ALPHAS = (.1, .15, .2, .3, .5, 1.)


def weibull_spec():
    """WeibullDisKin, fit in (log10 k, alpha)."""
    def starts(turnover, fm_obs):
        return [[np.log10(gamma(1 + 1 / alpha) / turnover), alpha] for alpha in WEIBULL_STARTING_ALPHAS]

    return ModelSpec(
        model_class=WeibullDisKin, param_names=('k', 'alpha'),
        lower=(-8., .01), upper=(6., 5.), starts=starts,
        to_params=lambda x: (10 ** x[0], x[1]), from_params=lambda k, alpha: (np.log10(k), alpha),
        exact_tol=EXACT_FM_TOL, least_squares_options=EXACT_FM_OPTIONS)


# Log-rate widths to start the log-uniform fit from, each with k_min matching the turnover.
LOGUNIFORM_STARTING_WIDTHS = (10., 100., 1000.)


def loguniform_spec():
    """LogUniformDisKin, fit in (log10 k_min, log10 log_width); log_width <= 5000."""
    def starts(turnover, fm_obs):
        return [[np.log10(-np.expm1(-width) / (turnover * width)), np.log10(width)]
                for width in LOGUNIFORM_STARTING_WIDTHS]

    return ModelSpec(
        model_class=LogUniformDisKin, param_names=('k_min', 'log_width'),
        lower=(-10., -3.), upper=(4., np.log10(5000.)), starts=starts,
        to_params=lambda x: (10 ** x[0], 10 ** x[1]),
        from_params=lambda k_min, log_width: (np.log10(k_min), np.log10(log_width)),
        exact_tol=EXACT_FM_TOL, least_squares_options=EXACT_FM_OPTIONS)
