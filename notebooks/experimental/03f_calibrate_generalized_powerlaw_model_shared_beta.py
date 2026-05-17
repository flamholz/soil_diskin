import numpy as np
import pandas as pd
from os import path
from scipy.optimize import least_squares

from soil_diskin.continuum_models import GeneralPowerLawDisKin

"""
Script should be run from the project root directory.

Calibrates the Generalized Power Law model with a single beta shared across
all sites while holding t_min fixed at 1.
"""

DATA_PATH = 'results/all_sites_14C_turnover.csv'
OUTPUT_DIR = 'results/03_calibrate_models/'
OUTPUT_NAME = 'general_powerlaw_model_optimization_results_shared_beta_fixed_tmin.csv'
INITIALIZATION_FILES = (
    'results/03_calibrate_models/general_powerlaw_model_optimization_results_fixed_tmin2.csv',
    'results/03_calibrate_models/general_powerlaw_model_optimization_results_fixed_tmin.csv',
)
DEFAULT_INITIAL_LOG10_T_MAX = 4.0
DEFAULT_INITIAL_BETA = 0.8
MAX_LOG10_T_MAX = 8.0
MIN_BETA = 1e-8
MAX_BETA = np.inf
T_MIN = 1.0


def get_model_predictions(t_min, t_max, beta):
    """Return the model predictions for turnover and steady-state radiocarbon."""
    model = GeneralPowerLawDisKin(t_min=t_min, t_max=t_max, beta=beta)
    predicted_14c_ratio = model.calc_radiocarbon_ratio_ss()[0]
    return model, predicted_14c_ratio


def build_initial_guess(n_sites, t_min):
    """Load a warm start from the sitewise fit if available, else use defaults."""
    min_t_max = t_min + 1e-8
    for candidate_path in INITIALIZATION_FILES:
        if not path.exists(candidate_path):
            continue

        init_df = pd.read_csv(candidate_path)
        if len(init_df) != n_sites or not {'t_max', 'beta'}.issubset(init_df.columns):
            continue

        initial_log10_t_max = np.log10(
            init_df['t_max'].fillna(10 ** DEFAULT_INITIAL_LOG10_T_MAX).clip(lower=min_t_max).to_numpy()
        )
        beta_values = init_df['beta'].dropna()
        initial_beta = DEFAULT_INITIAL_BETA if beta_values.empty else float(np.clip(beta_values.median(), MIN_BETA, MAX_BETA))
        initial_params = np.concatenate([initial_log10_t_max, [initial_beta]])
        return initial_params, candidate_path

    default_params = np.concatenate(
        [np.full(n_sites, DEFAULT_INITIAL_LOG10_T_MAX), [DEFAULT_INITIAL_BETA]]
    )
    return default_params, None


def joint_residuals_function(params, t_min, turnover_values, fm_values):
    """Return all site residuals for a shared beta and sitewise t_max values."""
    log10_t_max_values = params[:-1]
    beta = params[-1]
    residuals = np.empty(2 * len(turnover_values))

    for idx, (log10_t_max, turnover, fm) in enumerate(
        zip(log10_t_max_values, turnover_values, fm_values)
    ):
        t_max = 10 ** log10_t_max
        model, predicted_14c_ratio = get_model_predictions(t_min=t_min, t_max=t_max, beta=beta)
        turnover_scale = max(abs(turnover), 1e-6)
        fm_scale = max(abs(fm), 1e-6)

        residuals[2 * idx] = (model.T - turnover) / turnover_scale
        residuals[2 * idx + 1] = (predicted_14c_ratio - fm) / fm_scale

    return residuals


def results_to_dataframe(merged_site_data, t_min, fit_result, initialization_source):
    """Convert the shared-beta fit result into a per-site output table."""
    shared_beta = float(fit_result.x[-1])
    t_max_values = 10 ** fit_result.x[:-1]
    site_residuals = fit_result.fun.reshape(-1, 2)
    result_df = merged_site_data.copy()

    modeled_turnover = []
    modeled_14c = []
    params_valid = []
    for t_max in t_max_values:
        model, predicted_14c_ratio = get_model_predictions(t_min=t_min, t_max=t_max, beta=shared_beta)
        modeled_turnover.append(model.T)
        modeled_14c.append(predicted_14c_ratio)
        params_valid.append(model.params_valid())

    result_df['t_min'] = t_min
    result_df['t_max'] = t_max_values
    result_df['beta'] = shared_beta
    result_df['modeled_T'] = modeled_turnover
    result_df['modeled_14C'] = modeled_14c
    result_df['params_valid'] = params_valid
    result_df['relative_turnover_error'] = site_residuals[:, 0]
    result_df['relative_14C_error'] = site_residuals[:, 1]
    result_df['site_objective_value'] = np.sum(site_residuals ** 2, axis=1)
    result_df['global_objective_value'] = float(np.sum(fit_result.fun ** 2))
    result_df['fit_success'] = fit_result.success
    result_df['fit_status'] = fit_result.status
    result_df['fit_message'] = fit_result.message
    result_df['n_function_evaluations'] = fit_result.nfev
    result_df['initialization_source'] = initialization_source or 'default'

    return result_df


def main():
    merged_site_data = pd.read_csv(DATA_PATH)
    turnover_values = merged_site_data['turnover'].to_numpy()
    fm_values = merged_site_data['fm'].to_numpy()

    initial_guess, initialization_source = build_initial_guess(len(merged_site_data), T_MIN)
    lower_bounds = np.concatenate([
        np.full(len(merged_site_data), np.log10(T_MIN + 1e-8)),
        [MIN_BETA],
    ])
    upper_bounds = np.concatenate([
        np.full(len(merged_site_data), MAX_LOG10_T_MAX),
        [MAX_BETA],
    ])

    print(
        f'Running shared-beta optimization for generalized power law with fixed t_min={T_MIN} '
        f'across {len(merged_site_data)} sites...'
    )
    if initialization_source is None:
        print('Using default initialization.')
    else:
        print(f'Initializing from {initialization_source}.')

    fit_result = least_squares(
        joint_residuals_function,
        initial_guess,
        args=(T_MIN, turnover_values, fm_values),
        bounds=(lower_bounds, upper_bounds),
    )

    result_df = results_to_dataframe(
        merged_site_data=merged_site_data,
        t_min=T_MIN,
        fit_result=fit_result,
        initialization_source=initialization_source,
    )

    output_path = path.join(OUTPUT_DIR, OUTPUT_NAME)
    result_df.to_csv(output_path, index=False)

    print(f'Optimized shared beta: {fit_result.x[-1]:.6f}')
    print(f'Global objective value: {np.sum(fit_result.fun ** 2):.6f}')
    print(f'Max site objective value: {result_df["site_objective_value"].max():.6f}')
    print(f'Median site objective value: {result_df["site_objective_value"].median():.6f}')
    print(f'Saved results to {output_path}')


if __name__ == '__main__':
    main()
