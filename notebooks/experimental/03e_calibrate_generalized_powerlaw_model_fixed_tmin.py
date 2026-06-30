import warnings
import numpy as np
import pandas as pd

from os import makedirs, path
from scipy.integrate import IntegrationWarning
from scipy.optimize import least_squares
from soil_diskin.constants import INTERP_R_14C, LAMBDA_14C
from soil_diskin.continuum_models import GeneralPowerLawDisKin
from tqdm import tqdm

"""
Script should be run from the project root directory.

Calibrates the generalized power law model while holding t_min fixed.
For each fixed t_min, the script fits sitewise t_max and beta values.
"""

DATA_PATH = 'results/all_sites_14C_turnover.csv'
OUTPUT_DIR = 'results/03_calibrate_models/'
T_MIN_VALUES = np.logspace(-1, 1, 10)
DEFAULT_INITIAL_GUESS = np.array([4.0, 0.8], dtype=float)
MIN_BETA = 1e-8
MAX_BETA = np.inf
MAX_LOG10_T_MAX = 8.0
QUAD_LIMIT = 2000
QUAD_EPSABS = 1e-6
LARGE_RESIDUAL = 1e6
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


def approximate_radiocarbon_ratio(model):
    """Fast radiocarbon approximation used inside the optimizer."""
    integrand_values = FAST_INTEGRATION_WEIGHTS * model.pA(FAST_INTEGRATION_AGES)
    return float(np.trapezoid(integrand_values, FAST_INTEGRATION_AGES))


def get_model_predictions(t_min, t_max, beta, use_fast_14c=False):
    """Return the model predictions for turnover and steady-state radiocarbon."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        model = GeneralPowerLawDisKin(t_min=t_min, t_max=t_max, beta=beta)
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


def residuals_function(params, t_min, site_data):
    """Return normalized residuals for turnover and radiocarbon at one site."""
    log10_t_max, beta = params
    t_max = 10 ** log10_t_max
    try:
        model, predicted_14c_ratio = get_model_predictions(
            t_min=t_min,
            t_max=t_max,
            beta=beta,
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


def format_t_min_for_filename(t_min):
    """Format t_min compactly for use in output filenames."""
    return f'{t_min:.2f}'.rstrip('0').rstrip('.')


def build_initial_guess(t_min, previous_result_df, site_index):
    """Warm start from the previous t_min run when available."""
    if previous_result_df is None or site_index not in previous_result_df.index:
        return DEFAULT_INITIAL_GUESS.copy()

    previous_row = previous_result_df.loc[site_index]
    t_max = max(float(previous_row['t_max']), t_min + 1e-8)
    beta = float(np.clip(previous_row['beta'], MIN_BETA, MAX_BETA))
    return np.array([np.log10(t_max), beta], dtype=float)


def results_to_dataframe(results, t_min, merged_site_data):
    """Convert optimization results to a DataFrame."""
    result_df = pd.DataFrame(results, index=merged_site_data.index)
    result_df['t_min'] = t_min
    result_df['relative_turnover_error'] = (
        (result_df['modeled_T'] - merged_site_data['turnover'])
        / merged_site_data['turnover'].abs().clip(lower=1e-6)
    )
    result_df['relative_14C_error'] = (
        (result_df['modeled_14C'] - merged_site_data['fm'])
        / merged_site_data['fm'].abs().clip(lower=1e-6)
    )
    result_df['site_turnover'] = merged_site_data['turnover']
    result_df['site_fm'] = merged_site_data['fm']
    return result_df


def fit_single_t_min(merged_site_data, t_min, previous_result_df=None):
    """Fit t_max and beta for every site at one fixed t_min."""
    results = []
    lower_bounds = np.array([np.log10(t_min + 1e-8), MIN_BETA], dtype=float)
    upper_bounds = np.array([MAX_LOG10_T_MAX, MAX_BETA], dtype=float)

    print(f'Running optimization for generalized power law with fixed t_min={t_min:.2f}...')
    for site_index, row in tqdm(
        merged_site_data.iterrows(),
        total=len(merged_site_data),
        desc=f't_min={t_min:.2f}',
    ):
        initial_guess = build_initial_guess(t_min, previous_result_df, site_index)
        initial_guess[0] = max(initial_guess[0], lower_bounds[0])

        res = least_squares(
            residuals_function,
            initial_guess,
            args=(t_min, row),
            bounds=(lower_bounds, upper_bounds),
        )

        t_max = 10 ** res.x[0]
        beta = float(res.x[1])
        objective_value = float(np.sum(res.fun ** 2))
        evaluation_mode = 'exact'

        try:
            model, predicted_14c_ratio = get_model_predictions(
                t_min=t_min,
                t_max=t_max,
                beta=beta,
                use_fast_14c=False,
            )
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            model, predicted_14c_ratio = get_model_predictions(
                t_min=t_min,
                t_max=t_max,
                beta=beta,
                use_fast_14c=True,
            )
            evaluation_mode = 'fast_approximation'

        if (
            not model.params_valid()
            or not np.isfinite(model.T)
            or not np.isfinite(predicted_14c_ratio)
        ):
            predicted_14c_ratio = np.nan
            objective_value = np.inf
            evaluation_mode = 'failed'
        elif evaluation_mode == 'exact':
            turnover_scale = max(abs(row['turnover']), 1e-6)
            fm_scale = max(abs(row['fm']), 1e-6)
            objective_value = float(
                ((model.T - row['turnover']) / turnover_scale) ** 2
                + ((predicted_14c_ratio - row['fm']) / fm_scale) ** 2
            )

        results.append(
            {
                't_max': t_max,
                'beta': beta,
                'objective_value': objective_value,
                'fit_success': res.success,
                'fit_status': res.status,
                'fit_message': res.message,
                'n_function_evaluations': res.nfev,
                'radiocarbon_evaluation_mode': evaluation_mode,
                'modeled_T': model.T,
                'modeled_14C': predicted_14c_ratio,
                'params_valid': model.params_valid(),
            }
        )

    return results_to_dataframe(results, t_min, merged_site_data)


def main():
    """Run the fixed-t_min sweep and save one result file per t_min."""
    makedirs(OUTPUT_DIR, exist_ok=True)
    merged_site_data = pd.read_csv(DATA_PATH)
    previous_result_df = None

    for t_min in T_MIN_VALUES:
        result_df = fit_single_t_min(
            merged_site_data=merged_site_data,
            t_min=float(t_min),
            previous_result_df=previous_result_df,
        )

        t_min_label = format_t_min_for_filename(t_min)
        output_name = (
            f'general_powerlaw_model_optimization_results_fixed_tmin_{t_min_label}.csv'
        )
        output_path = path.join(OUTPUT_DIR, output_name)
        result_df.to_csv(output_path, index=False)
        print(f'Saved results to {output_path}')

        previous_result_df = result_df


if __name__ == '__main__':
    main()
