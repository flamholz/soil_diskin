import warnings
from os import makedirs, path

import numpy as np
import pandas as pd

from scipy.integrate import IntegrationWarning
from scipy.optimize import least_squares, minimize_scalar
from soil_diskin.constants import INTERP_R_14C, LAMBDA_14C
from soil_diskin.continuum_models import GaussianDisKin, LogUniformRateDisKin
from tqdm import tqdm

"""
Script should be run from the project root directory.

Calibrates two rate-distribution continuum models:
1. A bounded Gaussian rate distribution with fixed k bounds.
2. A log-uniform rate distribution with fitted k_min and k_max.
"""

DATA_PATH = 'results/all_sites_14C_turnover.csv'
OUTPUT_DIR = 'results/03_calibrate_models/'
QUAD_LIMIT = 2000
QUAD_EPSABS = 1e-6
LARGE_RESIDUAL = 1e6
OBJECTIVE_THRESHOLD = 1e-5
MIN_LOGUNIFORM_RATIO = 1.0 + 1e-9
MIN_LOG10_LOGUNIFORM_RATIO = np.log10(MIN_LOGUNIFORM_RATIO)

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


def unpack_gaussian_params(params):
    """Convert optimization-space parameters into Gaussian model parameters."""
    log10_mu, log10_sigma = params
    return {
        'mu': 10 ** log10_mu,
        'sigma': 10 ** log10_sigma,
    }


def pack_gaussian_params(mu, sigma):
    """Convert Gaussian model parameters into optimization-space parameters."""
    mu = max(float(mu), 1e-8)
    sigma = max(float(sigma), 1e-8)
    return np.array([np.log10(mu), np.log10(sigma)], dtype=float)


def unpack_loguniform_params(params):
    """Convert optimization-space parameters into log-uniform model parameters."""
    log10_k_min, log10_k_ratio = params
    k_min = 10 ** log10_k_min
    k_ratio = max(10 ** log10_k_ratio, MIN_LOGUNIFORM_RATIO)
    k_max = k_min * k_ratio
    return {
        'k_min': k_min,
        'k_max': k_max,
    }


def pack_loguniform_params(k_min, k_max):
    """Convert log-uniform model parameters into optimization-space parameters."""
    k_min = max(float(k_min), 1e-12)
    k_ratio = max(float(k_max) / k_min, MIN_LOGUNIFORM_RATIO)
    return np.array([np.log10(k_min), np.log10(k_ratio)], dtype=float)


def get_loguniform_turnover_matched_params(turnover, log10_k_ratio):
    """Return log-uniform parameters that match a target turnover exactly."""
    turnover = max(float(turnover), 1e-6)
    k_ratio = max(10 ** log10_k_ratio, MIN_LOGUNIFORM_RATIO)
    log_k_ratio = np.log(k_ratio)
    k_min = (1.0 - 1.0 / k_ratio) / (turnover * log_k_ratio)
    k_max = k_min * k_ratio
    return {
        'k_min': k_min,
        'k_max': k_max,
    }


MODEL_CONFIGS = (
    {
        'label': 'Gaussian rate',
        'model_class': GaussianDisKin,
        'param_names': ('mu', 'sigma'),
        'unpack': unpack_gaussian_params,
        'pack': pack_gaussian_params,
        'use_fast_14c': True,
        'max_nfev': 150,
        'lower_bounds': np.array([-4.0, -4.0]),
        'upper_bounds': np.array([0.0, 0.0]),
        'default_initial_guesses': (
            (-2.0, -1.5),
            (-1.5, -1.3),
            (-1.0, -1.0),
            (-0.5, -1.0),
            (-2.5, -2.0),
        ),
        'output_name': 'gaussian_rate_model_optimization_results.csv',
    },
    {
        'label': 'Log-uniform rate',
        'model_class': LogUniformRateDisKin,
        'param_names': ('k_min', 'k_max'),
        'unpack': unpack_loguniform_params,
        'pack': pack_loguniform_params,
        'use_fast_14c': False,
        'max_nfev': 300,
        'lower_bounds': np.array([-12.0, MIN_LOG10_LOGUNIFORM_RATIO]),
        'upper_bounds': np.array([2.0, 300.0]),
        'default_initial_guesses': (
            (-6.0, 1.0),
            (-4.0, 2.0),
            (-3.0, 4.0),
            (-2.0, 6.0),
            (-1.0, 8.0),
        ),
        'output_name': 'loguniform_rate_model_optimization_results.csv',
    },
)


def approximate_radiocarbon_ratio(model):
    """Fast radiocarbon approximation used inside the optimizer."""
    integrand_values = FAST_INTEGRATION_WEIGHTS * model.pA(FAST_INTEGRATION_AGES)
    return float(np.trapezoid(integrand_values, FAST_INTEGRATION_AGES))


def get_model_predictions(model_config, params, use_fast_14c=False):
    """Return the model and steady-state radiocarbon prediction."""
    model_params = model_config['unpack'](params)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        model = model_config['model_class'](**model_params)
        if use_fast_14c:
            predicted_14c_ratio = approximate_radiocarbon_ratio(model)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', IntegrationWarning)
                predicted_14c_ratio = model.calc_radiocarbon_ratio_ss(
                    quad_limit=QUAD_LIMIT,
                    quad_epsabs=QUAD_EPSABS,
                )[0]
    return model, predicted_14c_ratio


def residuals_function(params, model_config, site_data, use_fast_14c=None):
    """Return normalized residuals for turnover and radiocarbon at one site."""
    if use_fast_14c is None:
        use_fast_14c = model_config.get('use_fast_14c', False)

    try:
        model, predicted_14c_ratio = get_model_predictions(
            model_config=model_config,
            params=params,
            use_fast_14c=use_fast_14c,
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


def optimize_loguniform_turnover_matched_guess(
    model_config,
    site_data,
    return_search_result=False,
):
    """Return a turnover-matched initial guess that minimizes the exact fm error."""
    if model_config['model_class'] is not LogUniformRateDisKin:
        return (None, None) if return_search_result else None

    fm_scale = max(abs(site_data['fm']), 1e-6)
    lower_bound = float(model_config['lower_bounds'][1])
    upper_bound = float(model_config['upper_bounds'][1])

    def objective(log10_k_ratio):
        params = get_loguniform_turnover_matched_params(
            site_data['turnover'],
            log10_k_ratio,
        )
        try:
            with np.errstate(
                over='ignore',
                invalid='ignore',
                divide='ignore',
                under='ignore',
            ):
                model = LogUniformRateDisKin(**params)
            if not model.params_valid() or not np.isfinite(model.T):
                return LARGE_RESIDUAL ** 2
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', IntegrationWarning)
                predicted_14c_ratio = model.calc_radiocarbon_ratio_ss(
                    quad_limit=QUAD_LIMIT,
                    quad_epsabs=QUAD_EPSABS,
                )[0]
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            return LARGE_RESIDUAL ** 2

        if not np.isfinite(predicted_14c_ratio):
            return LARGE_RESIDUAL ** 2

        return float(((predicted_14c_ratio - site_data['fm']) / fm_scale) ** 2)

    search_result = minimize_scalar(
        objective,
        bounds=(lower_bound, upper_bound),
        method='bounded',
        options={'xatol': 1e-6, 'maxiter': 500},
    )
    if not search_result.success:
        return (None, search_result) if return_search_result else None

    params = get_loguniform_turnover_matched_params(
        site_data['turnover'],
        search_result.x,
    )
    packed_params = model_config['pack'](params['k_min'], params['k_max'])
    if return_search_result:
        return packed_params, search_result
    return packed_params


def site_specific_initial_guess(model_config, site_data):
    """Return a starting point based on the site's turnover value."""
    turnover = max(float(site_data['turnover']), 1e-6)
    central_rate = np.clip(1 / turnover, 1e-4, 1.0)

    if model_config['model_class'] is GaussianDisKin:
        sigma = np.clip(0.5 * central_rate, 1e-4, 1.0)
        return pack_gaussian_params(central_rate, sigma)

    k_min = np.clip(1 / (10 * turnover), 1e-8, 1.0)
    k_max = np.clip(10 / turnover, k_min + 1e-8, 10.0)
    return pack_loguniform_params(k_min, k_max)


def build_initial_guesses(model_config, site_data, warm_start=None):
    """Return a de-duplicated list of starting points."""
    initial_guesses = [site_specific_initial_guess(model_config, site_data)]
    turnover_matched_guess = optimize_loguniform_turnover_matched_guess(
        model_config,
        site_data,
    )
    if turnover_matched_guess is not None:
        initial_guesses.insert(0, turnover_matched_guess)
    if warm_start is not None:
        initial_guesses.insert(0, np.asarray(warm_start, dtype=float))
    initial_guesses.extend(
        np.asarray(guess, dtype=float)
        for guess in model_config['default_initial_guesses']
    )

    unique_guesses = []
    seen = set()
    lower_bounds = model_config['lower_bounds']
    upper_bounds = model_config['upper_bounds']
    for guess in initial_guesses:
        clipped_guess = np.clip(guess, lower_bounds, upper_bounds)
        rounded_guess = tuple(np.round(clipped_guess, 12))
        if rounded_guess in seen:
            continue
        seen.add(rounded_guess)
        unique_guesses.append(clipped_guess)

    return unique_guesses


def fit_single_site(site_data, model_config, warm_start=None):
    """Fit one site using multiple starting points and keep the best result."""
    if model_config['model_class'] is LogUniformRateDisKin:
        best_params, search_result = optimize_loguniform_turnover_matched_guess(
            model_config,
            site_data,
            return_search_result=True,
        )
        if best_params is not None:
            evaluation_mode = 'exact'
            try:
                model, predicted_14c_ratio = get_model_predictions(
                    model_config=model_config,
                    params=best_params,
                    use_fast_14c=False,
                )
            except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
                model, predicted_14c_ratio = get_model_predictions(
                    model_config=model_config,
                    params=best_params,
                    use_fast_14c=True,
                )
                evaluation_mode = 'fast_approximation'

            if (
                not model.params_valid()
                or not np.isfinite(model.T)
                or not np.isfinite(predicted_14c_ratio)
            ):
                exact_objective_value = np.inf
                predicted_14c_ratio = np.nan
                evaluation_mode = 'failed'
            else:
                turnover_scale = max(abs(site_data['turnover']), 1e-6)
                fm_scale = max(abs(site_data['fm']), 1e-6)
                exact_objective_value = float(
                    ((model.T - site_data['turnover']) / turnover_scale) ** 2
                    + ((predicted_14c_ratio - site_data['fm']) / fm_scale) ** 2
                )

            model_params = model_config['unpack'](best_params)
            return {
                'objective_value': exact_objective_value,
                'fit_success': bool(search_result.success),
                'fit_status': getattr(search_result, 'status', int(search_result.success)),
                'fit_message': str(search_result.message),
                'n_function_evaluations': int(search_result.nfev),
                'radiocarbon_evaluation_mode': evaluation_mode,
                'initial_param_1': np.nan,
                'initial_param_2': np.nan,
                'modeled_T': model.T,
                'modeled_14C': predicted_14c_ratio,
                'params_valid': model.params_valid(),
                **model_params,
            }

    best_result = None
    best_objective = np.inf
    best_start = None

    for start in build_initial_guesses(model_config, site_data, warm_start=warm_start):
        fit_result = least_squares(
            residuals_function,
            x0=start,
            args=(model_config, site_data),
            bounds=(model_config['lower_bounds'], model_config['upper_bounds']),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=model_config.get('max_nfev', 150),
        )
        objective_value = float(np.sum(fit_result.fun ** 2))
        if objective_value < best_objective:
            best_objective = objective_value
            best_result = fit_result
            best_start = start

    evaluation_mode = 'exact'
    try:
        model, predicted_14c_ratio = get_model_predictions(
            model_config=model_config,
            params=best_result.x,
            use_fast_14c=False,
        )
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
        model, predicted_14c_ratio = get_model_predictions(
            model_config=model_config,
            params=best_result.x,
            use_fast_14c=True,
        )
        evaluation_mode = 'fast_approximation'

    if (
        not model.params_valid()
        or not np.isfinite(model.T)
        or not np.isfinite(predicted_14c_ratio)
    ):
        predicted_14c_ratio = np.nan
        exact_objective_value = np.inf
        evaluation_mode = 'failed'
    else:
        turnover_scale = max(abs(site_data['turnover']), 1e-6)
        fm_scale = max(abs(site_data['fm']), 1e-6)
        exact_objective_value = float(
            ((model.T - site_data['turnover']) / turnover_scale) ** 2
            + ((predicted_14c_ratio - site_data['fm']) / fm_scale) ** 2
        )

    model_params = model_config['unpack'](best_result.x)
    result = {
        'objective_value': exact_objective_value,
        'fit_success': best_result.success,
        'fit_status': best_result.status,
        'fit_message': best_result.message,
        'n_function_evaluations': best_result.nfev,
        'radiocarbon_evaluation_mode': evaluation_mode,
        'initial_param_1': float(best_start[0]),
        'initial_param_2': float(best_start[1]),
        'modeled_T': model.T,
        'modeled_14C': predicted_14c_ratio,
        'params_valid': model.params_valid(),
    }
    result.update(model_params)
    return result


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


def fit_sites(site_data, model_config, description, initialization_df=None):
    """Run the per-site calibration for all rows in a DataFrame."""
    results = []
    for idx, row in tqdm(site_data.iterrows(), total=len(site_data), desc=description):
        warm_start = None
        if initialization_df is not None and idx in initialization_df.index:
            init_row = initialization_df.loc[idx]
            warm_start = model_config['pack'](
                *[init_row[param] for param in model_config['param_names']]
            )
        results.append(fit_single_site(row, model_config, warm_start=warm_start))

    return results_to_dataframe(results, site_data)


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


def run_model_calibration(merged_site_data, model_config, backfilled_sites):
    """Run central, q05, and q95 fits for one model."""
    label = model_config['label']
    print(f'Running optimization for {label} model...')
    result_df = fit_sites(
        merged_site_data,
        model_config=model_config,
        description=label,
    )
    summarize_fit(result_df, f'Central fit summary for {label}')

    results_by_suffix = {}
    for suffix in ('05', '95'):
        uncertainty_df = build_uncertainty_frame(backfilled_sites, f'turnover_q{suffix}')
        results_by_suffix[suffix] = fit_sites(
            uncertainty_df,
            model_config=model_config,
            description=f'{label} q{suffix}',
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

    output_path = path.join(OUTPUT_DIR, model_config['output_name'])
    merged_result_df.to_csv(output_path, index=False)
    print(f'Saved results to {output_path}')
    return merged_result_df


def main():
    """Run both rate-distribution calibrations."""
    makedirs(OUTPUT_DIR, exist_ok=True)
    merged_site_data = pd.read_csv(DATA_PATH)
    backfilled_sites = merged_site_data[
        merged_site_data['turnover_q05'].notna()
        & merged_site_data['turnover_q95'].notna()
    ]

    for model_config in MODEL_CONFIGS:
        run_model_calibration(
            merged_site_data=merged_site_data,
            model_config=model_config,
            backfilled_sites=backfilled_sites,
        )


if __name__ == '__main__':
    main()
