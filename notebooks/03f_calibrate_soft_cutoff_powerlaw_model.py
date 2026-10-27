import warnings
import numpy as np
import pandas as pd

from os import path
from scipy.integrate import IntegrationWarning
from scipy.optimize import least_squares
from soil_diskin.constants import INTERP_R_14C, LAMBDA_14C
from soil_diskin.continuum_models import SoftCutoffPowerLawDisKin
from tqdm import tqdm

"""
Script should be run from the project root directory.

Calibrates the soft-cutoff power law model (SoftCutoffPowerLawDisKin) with
alpha fixed, sweeping alpha from 0.2 to 1.0 in steps of 0.1. For each alpha,
t_min and t_max are fit per site to turnover and radiocarbon, following
03c_calibrate_generalized_powerlaw_model.py.

All alpha values are written to a single long-format file with one row per
(alpha, site).
"""

DATA_PATH = 'results/all_sites_14C_turnover.csv'
OUTPUT_PATH = 'results/03_calibrate_models/soft_cutoff_powerlaw_model_alpha_sweep.csv'
# alpha < 0.2 is excluded: near alpha = 0 (a log-uniform rate distribution) A/T
# grows only logarithmically in t_max/t_min, too slowly to match the observed
# radiocarbon. In a 0.0-1.0 sweep, t_min hit its 1e-8 yr lower bound and only
# 6/99 (alpha = 0.0) and 49/99 (alpha = 0.1) sites fit, vs. 98-99/99 for alpha >= 0.2.
ALPHA_VALUES = np.round(np.linspace(0.2, 1.0, 9), 1)
OBJECTIVE_THRESHOLD = 1e-4
QUAD_LIMIT = 2000
QUAD_EPSABS = 1e-6
LARGE_RESIDUAL = 1e6
LOWER_BOUNDS = np.array([-8.0, -8.0])
UPPER_BOUNDS = np.array([8.0, 8.0])
FAST_INTEGRATION_AGES = np.unique(
    np.concatenate(
        [
            np.array([0.0]),
            np.geomspace(1e-8, 1e-2, 300),
            np.geomspace(1e-2, 1.0, 300),
            np.geomspace(1.0, 100.0, 600),
            np.geomspace(100.0, 5000.0, 1000),
            np.geomspace(5000.0, 60000.0, 1500),
        ]
    )
)
FAST_INTEGRATION_WEIGHTS = INTERP_R_14C(FAST_INTEGRATION_AGES) * np.exp(
    -LAMBDA_14C * FAST_INTEGRATION_AGES
)

# Optimize in log space to improve scaling and parameter hierarchy:
# x[0] = log10(t_min), x[1] = log10(t_max - t_min)
DEFAULT_INITIAL_GUESSES = (
    (-4.0, -1.0),
    (-3.5, 3.0),
    (-2.0, 1.0),
    (-1.0, 2.0),
    (0.0, 2.5),
    (0.5, 3.0),
    (1.0, 3.0),
)


def unpack_optimization_params(params):
    """Convert optimization-space parameters into physical model parameters."""
    log10_t_min, log10_t_gap = params
    t_min = 10 ** log10_t_min
    t_gap = 10 ** log10_t_gap
    t_max = t_min + t_gap
    return t_min, t_max


def pack_optimization_params(t_min, t_max):
    """Convert physical model parameters into optimization-space parameters."""
    t_min = max(float(t_min), 1e-8)
    t_gap = max(float(t_max) - float(t_min), 1e-8)
    return np.array([np.log10(t_min), np.log10(t_gap)], dtype=float)


def approximate_radiocarbon_ratio(model):
    """Fast radiocarbon approximation used inside the optimizer."""
    integrand_values = FAST_INTEGRATION_WEIGHTS * model.pA(FAST_INTEGRATION_AGES)
    return float(np.trapezoid(integrand_values, FAST_INTEGRATION_AGES))


def get_model_predictions(t_min, t_max, alpha, use_fast_14c=False):
    """Return the model, predicted turnover, and steady-state radiocarbon."""
    model = SoftCutoffPowerLawDisKin(t_min=t_min, t_max=t_max, alpha=alpha)
    if use_fast_14c:
        predicted_14c_ratio = approximate_radiocarbon_ratio(model)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            predicted_14c_ratio = model.calc_radiocarbon_ratio_ss(
                quad_limit=QUAD_LIMIT,
                quad_epsabs=QUAD_EPSABS,
            )[0]
    return model, predicted_14c_ratio


def residuals_function(params, alpha, site_data):
    """Return normalized residuals for turnover and radiocarbon at one site."""
    try:
        t_min, t_max = unpack_optimization_params(params)
        model, predicted_14c_ratio = get_model_predictions(
            t_min=t_min,
            t_max=t_max,
            alpha=alpha,
            use_fast_14c=True,
        )
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
        return np.array([LARGE_RESIDUAL, LARGE_RESIDUAL], dtype=float)

    if (
        not model.params_valid()
        or not np.isfinite(model.T)
        or not np.isfinite(predicted_14c_ratio)
    ):
        return np.array([LARGE_RESIDUAL, LARGE_RESIDUAL], dtype=float)

    turnover_scale = max(abs(site_data['turnover']), 1e-6)
    fm_scale = max(abs(site_data['fm']), 1e-6)
    turnover_residual = (model.T - site_data['turnover']) / turnover_scale
    fm_residual = (predicted_14c_ratio - site_data['fm']) / fm_scale
    return np.array([turnover_residual, fm_residual], dtype=float)


def build_initial_guesses(warm_start=None):
    """Return a de-duplicated list of starting points."""
    initial_guesses = []
    if warm_start is not None:
        initial_guesses.append(tuple(np.asarray(warm_start, dtype=float)))
    initial_guesses.extend(DEFAULT_INITIAL_GUESSES)

    unique_guesses = []
    seen = set()
    for guess in initial_guesses:
        rounded_guess = tuple(np.round(guess, 12))
        if rounded_guess in seen:
            continue
        seen.add(rounded_guess)
        unique_guesses.append(np.asarray(guess, dtype=float))

    return unique_guesses


def fit_single_site(site_data, alpha, warm_start=None):
    """Fit one site using multiple starting points and keep the best result."""
    best_result = None
    best_objective = np.inf
    best_start = None

    for start in build_initial_guesses(warm_start=warm_start):
        fit_result = least_squares(
            residuals_function,
            x0=start,
            args=(alpha, site_data),
            bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=150,
        )
        objective_value = float(np.sum(fit_result.fun ** 2))
        if objective_value < best_objective:
            best_objective = objective_value
            best_result = fit_result
            best_start = start

    t_min, t_max = unpack_optimization_params(best_result.x)
    model, predicted_14c_ratio = get_model_predictions(
        t_min=t_min,
        t_max=t_max,
        alpha=alpha,
        use_fast_14c=False,
    )
    turnover_scale = max(abs(site_data['turnover']), 1e-6)
    fm_scale = max(abs(site_data['fm']), 1e-6)
    exact_objective_value = float(
        ((model.T - site_data['turnover']) / turnover_scale) ** 2
        + ((predicted_14c_ratio - site_data['fm']) / fm_scale) ** 2
    )
    return {
        'objective_value': exact_objective_value,
        'fit_success': best_result.success,
        'fit_status': best_result.status,
        'fit_message': best_result.message,
        'n_function_evaluations': best_result.nfev,
        'initial_log10_t_min': float(best_start[0]),
        'initial_log10_t_gap': float(best_start[1]),
        't_min': t_min,
        't_max': t_max,
        'alpha': alpha,
        'modeled_T': model.T,
        'modeled_14C': predicted_14c_ratio,
        'modeled_A': model.A,
        'params_valid': model.params_valid(),
    }


def fit_sites(site_data, alpha, description, initialization_df=None):
    """Run the per-site calibration for all rows in a DataFrame."""
    results = []
    for idx, row in tqdm(site_data.iterrows(), total=len(site_data), desc=description):
        warm_start = None
        if initialization_df is not None and idx in initialization_df.index:
            init_row = initialization_df.loc[idx]
            warm_start = pack_optimization_params(init_row['t_min'], init_row['t_max'])
        results.append(fit_single_site(row, alpha=alpha, warm_start=warm_start))

    return results_to_dataframe(results, site_data)


def results_to_dataframe(results, site_data):
    """Attach modeled values and relative errors to the fit results."""
    result_df = pd.DataFrame(results, index=site_data.index)
    turnover_scale = site_data['turnover'].abs().clip(lower=1e-6)
    fm_scale = site_data['fm'].abs().clip(lower=1e-6)
    result_df['relative_turnover_error'] = (
        result_df['modeled_T'] - site_data['turnover']
    ) / turnover_scale
    result_df['relative_14C_error'] = (
        result_df['modeled_14C'] - site_data['fm']
    ) / fm_scale
    return result_df


def summarize_fit(result_df, label):
    """Print a compact summary of fit quality."""
    successful_fits = int(result_df['fit_success'].sum())
    below_threshold = int((result_df['objective_value'] < OBJECTIVE_THRESHOLD).sum())
    print(f'{label}:')
    print(f'  Successful fits: {successful_fits}/{len(result_df)}')
    print(f'  Median objective value: {result_df["objective_value"].median():.6e}')
    print(f'  Maximum objective value: {result_df["objective_value"].max():.6e}')
    print(f'  Sites below {OBJECTIVE_THRESHOLD:.1e}: {below_threshold}/{len(result_df)}')


def build_uncertainty_frame(site_data, turnover_column):
    """Return a copy of the site data with a substituted turnover column."""
    uncertainty_df = site_data.copy()
    uncertainty_df['turnover'] = uncertainty_df[turnover_column]
    return uncertainty_df


def run_single_alpha_calibration(merged_site_data, alpha, backfilled_sites, initialization_df=None):
    """Run central, q05, and q95 fits for one fixed alpha value.

    initialization_df optionally warm-starts the central fit, e.g. from the
    previous alpha in the sweep.
    """
    print(f'Running optimization for soft-cutoff power law with alpha = {alpha}...')
    result_df = fit_sites(
        merged_site_data,
        alpha=alpha,
        description=f'alpha={alpha}',
        initialization_df=initialization_df,
    )
    summarize_fit(result_df, f'Central fit summary for alpha = {alpha}')

    results_by_suffix = {}
    for suffix in ('05', '95'):
        uncertainty_df = build_uncertainty_frame(backfilled_sites, f'turnover_q{suffix}')
        results_by_suffix[suffix] = fit_sites(
            uncertainty_df,
            alpha=alpha,
            description=f'alpha={alpha} q{suffix}',
            initialization_df=result_df.loc[backfilled_sites.index],
        )

    merged_result_df = pd.concat([result_df, merged_site_data[['fm', 'turnover']]], axis=1)
    for suffix, uncertainty_result_df in results_by_suffix.items():
        merged_result_df = pd.merge(
            merged_result_df,
            uncertainty_result_df.add_suffix(f'_{suffix}'),
            left_index=True,
            right_index=True,
            how='left',
        )

    return result_df, merged_result_df


def main():
    merged_site_data = pd.read_csv(DATA_PATH)
    backfilled_sites = merged_site_data[
        merged_site_data['turnover_q05'].notna() & merged_site_data['turnover_q95'].notna()
    ]

    sweep_results = []
    previous_result_df = None
    for alpha in ALPHA_VALUES:
        previous_result_df, merged_result_df = run_single_alpha_calibration(
            merged_site_data=merged_site_data,
            alpha=float(alpha),
            backfilled_sites=backfilled_sites,
            initialization_df=previous_result_df,
        )
        # site_index matches row order in DATA_PATH and in the per-alpha 03c outputs
        sweep_results.append(merged_result_df.rename_axis('site_index').reset_index())

    sweep_df = pd.concat(sweep_results, ignore_index=True)
    leading_columns = ['alpha', 'site_index', 't_min', 't_max']
    sweep_df = sweep_df[leading_columns + [c for c in sweep_df.columns if c not in leading_columns]]
    sweep_df.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved results to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
