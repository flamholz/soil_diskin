import argparse
from os import path
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from soil_diskin.constants import GAMMA
from soil_diskin.continuum_models import GeneralPowerLawDisKin
from tqdm import tqdm

"""
Script should be run from the project root directory.

Fits a shared beta across all sites while shrinking the site-level log(t_min)
distribution toward a common pooled mean. The procedure is an empirical-Bayes
MAP approximation:

1. For a fixed shrinkage precision, solve one joint least-squares problem with
   a shared beta and site-specific (t_min, t_max).
2. Update the shrinkage precision from the fitted variance of log(t_min).
3. Repeat until beta and the pooled variance stabilize.

This keeps the per-site model identical to the existing calibration code while
adding a hierarchical preference for the lowest between-site variance in t_min
that still reconstructs turnover and radiocarbon observations.
"""

RELATIVE_EPS = 1e-6
DEFAULT_BETA = 0.76
DEFAULT_SIGMA_FLOOR = 0.2
DEFAULT_OUTER_TOL = 1e-3
DEFAULT_MAX_OUTER_ITER = 8
DEFAULT_BETA_PRIOR_SCALE = 0.2
DEFAULT_DATA_WEIGHT = 100.0
LOG_PARAM_BOUNDS = (-16.0, 16.0)
LEAST_SQUARES_KWARGS = {
    'method': 'trf',
    'ftol': 1e-10,
    'xtol': 1e-10,
    'gtol': 1e-10,
}


def unpack_log_params(log_params):
    """Convert unconstrained log parameters into ordered model parameters."""
    log_t_min, log_t_gap = log_params
    t_min = np.exp(log_t_min)
    t_max = t_min + np.exp(log_t_gap)
    return t_min, t_max


def initial_log_guess(site_data):
    """Choose a scale-aware initial guess for the site being optimized."""
    t_min_guess = max(1e-6, float(site_data['turnover']) / 1000)
    t_gap_guess = max(float(site_data['turnover']), 1000)
    return np.log([t_min_guess, t_gap_guess])


def safe_model_predictions(log_params, beta):
    """Evaluate modeled turnover and radiocarbon with a large penalty on failures."""
    t_min, t_max = unpack_log_params(log_params)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', IntegrationWarning)
            model = GeneralPowerLawDisKin(t_min=t_min, t_max=t_max, beta=beta)
            modeled_14c = quad(
                model.radiocarbon_age_integrand,
                0,
                np.inf,
                limit=1500,
                epsabs=1e-3,
            )[0]
    except (FloatingPointError, IntegrationWarning, OverflowError, RuntimeError, ValueError):
        return np.nan, np.nan, False

    finite = np.isfinite(model.T) and np.isfinite(modeled_14c)
    return model.T, modeled_14c, finite and model.params_valid()


def objective_residuals(log_params, beta, observed_fm, observed_turnover):
    """Relative residual vector comparing model predictions and observations."""
    modeled_turnover, modeled_14c, valid = safe_model_predictions(log_params, beta)
    if not valid:
        return np.array([1e6, 1e6], dtype=float)

    relative_diff_14c = (modeled_14c - observed_fm) / (observed_fm + RELATIVE_EPS)
    relative_diff_turnover = (
        modeled_turnover - observed_turnover
    ) / (observed_turnover + RELATIVE_EPS)
    return np.array([relative_diff_14c, relative_diff_turnover], dtype=float)


def objective_function(log_params, beta, observed_fm, observed_turnover):
    """Return the sum of squared relative residuals for diagnostics/output."""
    residuals = objective_residuals(log_params, beta, observed_fm, observed_turnover)
    return float(np.dot(residuals, residuals))


def build_parameter_bounds(n_sites, beta_min, beta_max):
    """Bounds for the joint least-squares problem."""
    lower = np.full(2 + 2 * n_sites, LOG_PARAM_BOUNDS[0], dtype=float)
    upper = np.full(2 + 2 * n_sites, LOG_PARAM_BOUNDS[1], dtype=float)
    lower[0] = beta_min
    upper[0] = beta_max
    return lower, upper


def pack_state(beta, mu_log_t_min, site_log_params):
    """Pack global and site parameters into a flat optimization vector."""
    return np.concatenate(([beta, mu_log_t_min], np.asarray(site_log_params, dtype=float).ravel()))


def unpack_state(state, n_sites):
    """Unpack the flat optimization vector into global and site parameters."""
    beta = float(state[0])
    mu_log_t_min = float(state[1])
    site_log_params = np.asarray(state[2:], dtype=float).reshape(n_sites, 2)
    return beta, mu_log_t_min, site_log_params


def build_jacobian_sparsity(n_sites):
    """Jacobian sparsity for the joint residual function."""
    n_rows = 3 * n_sites + 2
    n_cols = 2 + 2 * n_sites
    sparsity = lil_matrix((n_rows, n_cols), dtype=int)

    for site_idx in range(n_sites):
        row_offset = 3 * site_idx
        col_offset = 2 + 2 * site_idx

        # Data residuals depend on shared beta and the local site parameters.
        sparsity[row_offset, 0] = 1
        sparsity[row_offset + 1, 0] = 1
        sparsity[row_offset, col_offset] = 1
        sparsity[row_offset + 1, col_offset] = 1
        sparsity[row_offset, col_offset + 1] = 1
        sparsity[row_offset + 1, col_offset + 1] = 1

        # Shrinkage residual depends on pooled mean and the local log(t_min).
        sparsity[row_offset + 2, 1] = 1
        sparsity[row_offset + 2, col_offset] = 1

    # Weak empirical priors on shared beta and pooled log(t_min).
    sparsity[3 * n_sites, 0] = 1
    sparsity[3 * n_sites + 1, 1] = 1

    return sparsity.tocsr()


def hierarchical_residuals(
    state,
    observed_fm,
    observed_turnover,
    shrinkage_precision,
    beta_prior_mean,
    beta_prior_scale,
    mu_prior_mean,
    mu_prior_scale,
    data_weight,
):
    """Residual vector for the joint hierarchical fit."""
    n_sites = observed_fm.shape[0]
    beta, mu_log_t_min, site_log_params = unpack_state(state, n_sites)
    residuals = np.empty(3 * n_sites + 2, dtype=float)
    data_scale = np.sqrt(data_weight)
    shrinkage_scale = np.sqrt(shrinkage_precision)

    for site_idx in range(n_sites):
        data_residuals = objective_residuals(
            site_log_params[site_idx],
            beta,
            observed_fm[site_idx],
            observed_turnover[site_idx],
        )
        row_offset = 3 * site_idx
        residuals[row_offset:row_offset + 2] = data_scale * data_residuals
        residuals[row_offset + 2] = shrinkage_scale * (
            site_log_params[site_idx, 0] - mu_log_t_min
        )

    residuals[3 * n_sites] = (beta - beta_prior_mean) / beta_prior_scale
    residuals[3 * n_sites + 1] = (mu_log_t_min - mu_prior_mean) / mu_prior_scale
    return residuals


def fit_sites_independently(merged_site_data, beta):
    """Independent per-site fit used to initialize the hierarchical optimization."""
    n_sites = len(merged_site_data)
    site_log_params = np.zeros((n_sites, 2), dtype=float)
    objective_values = np.zeros(n_sites, dtype=float)
    lower = np.full(2, LOG_PARAM_BOUNDS[0], dtype=float)
    upper = np.full(2, LOG_PARAM_BOUNDS[1], dtype=float)

    print(f'Running independent initialization at beta={beta:.4f}...')
    iterator = tqdm(merged_site_data.iterrows(), total=n_sites)
    for idx, (_, row) in enumerate(iterator):
        res = least_squares(
            objective_residuals,
            initial_log_guess(row),
            bounds=(lower, upper),
            args=(beta, float(row['fm']), float(row['turnover'])),
            **LEAST_SQUARES_KWARGS,
        )
        site_log_params[idx] = res.x
        objective_values[idx] = objective_function(
            res.x,
            beta,
            float(row['fm']),
            float(row['turnover']),
        )

    return site_log_params, objective_values


def variance_from_site_params(site_log_params, mu_log_t_min):
    """Variance of the pooled log(t_min) distribution."""
    centered = site_log_params[:, 0] - mu_log_t_min
    return float(np.mean(centered ** 2))


def pooled_log_t_min_mean(site_log_params):
    """Pooled mean of log(t_min)."""
    return float(np.mean(site_log_params[:, 0]))


def profiled_joint_score(data_objectives, variance_log_t_min, sigma_floor, data_weight):
    """Approximate profiled MAP objective for reporting and model comparison."""
    n_sites = data_objectives.shape[0]
    return float(
        data_weight * np.sum(data_objectives)
        + n_sites * np.log(variance_log_t_min + sigma_floor ** 2)
    )


def evaluate_site_fits(site_log_params, beta, observed_fm, observed_turnover):
    """Evaluate data-fit diagnostics for the current site parameter matrix."""
    n_sites = observed_fm.shape[0]
    modeled_turnover = np.zeros(n_sites, dtype=float)
    modeled_14c = np.zeros(n_sites, dtype=float)
    turnover_rel_error = np.zeros(n_sites, dtype=float)
    fm_rel_error = np.zeros(n_sites, dtype=float)
    params_valid = np.zeros(n_sites, dtype=bool)
    data_objective = np.zeros(n_sites, dtype=float)

    for site_idx in range(n_sites):
        current_turnover, current_14c, valid = safe_model_predictions(site_log_params[site_idx], beta)
        modeled_turnover[site_idx] = current_turnover
        modeled_14c[site_idx] = current_14c
        params_valid[site_idx] = valid
        if valid:
            fm_rel_error[site_idx] = (
                current_14c - observed_fm[site_idx]
            ) / (observed_fm[site_idx] + RELATIVE_EPS)
            turnover_rel_error[site_idx] = (
                current_turnover - observed_turnover[site_idx]
            ) / (observed_turnover[site_idx] + RELATIVE_EPS)
            data_objective[site_idx] = fm_rel_error[site_idx] ** 2 + turnover_rel_error[site_idx] ** 2
        else:
            fm_rel_error[site_idx] = np.nan
            turnover_rel_error[site_idx] = np.nan
            data_objective[site_idx] = np.inf

    return {
        'modeled_turnover': modeled_turnover,
        'modeled_14c': modeled_14c,
        'turnover_rel_error': turnover_rel_error,
        'fm_rel_error': fm_rel_error,
        'params_valid': params_valid,
        'data_objective': data_objective,
    }


def hierarchical_map_fit(
    merged_site_data,
    beta_init=DEFAULT_BETA,
    beta_min=0.05,
    beta_max=0.99,
    sigma_floor=DEFAULT_SIGMA_FLOOR,
    max_outer_iter=DEFAULT_MAX_OUTER_ITER,
    outer_tol=DEFAULT_OUTER_TOL,
    beta_prior_scale=DEFAULT_BETA_PRIOR_SCALE,
    data_weight=DEFAULT_DATA_WEIGHT,
):
    """Joint empirical-Bayes calibration with a shared beta across all sites."""
    observed_fm = merged_site_data['fm'].to_numpy(dtype=float)
    observed_turnover = merged_site_data['turnover'].to_numpy(dtype=float)
    n_sites = observed_fm.shape[0]

    independent_log_params, independent_objectives = fit_sites_independently(
        merged_site_data,
        beta=beta_init,
    )

    mu_log_t_min = pooled_log_t_min_mean(independent_log_params)
    variance_log_t_min = variance_from_site_params(independent_log_params, mu_log_t_min)
    shrinkage_precision = 1.0 / (variance_log_t_min + sigma_floor ** 2)
    mu_prior_mean = mu_log_t_min
    mu_prior_scale = max(np.sqrt(variance_log_t_min), sigma_floor)

    state = pack_state(beta_init, mu_log_t_min, independent_log_params)
    bounds = build_parameter_bounds(n_sites, beta_min=beta_min, beta_max=beta_max)
    jacobian_sparsity = build_jacobian_sparsity(n_sites)

    history_rows = []
    prev_beta = beta_init
    prev_variance = variance_log_t_min

    print('Running hierarchical empirical-Bayes optimization...')
    for outer_iter in range(max_outer_iter):
        res = least_squares(
            hierarchical_residuals,
            state,
            bounds=bounds,
            jac_sparsity=jacobian_sparsity,
            args=(
                observed_fm,
                observed_turnover,
                shrinkage_precision,
                beta_init,
                beta_prior_scale,
                mu_prior_mean,
                mu_prior_scale,
                data_weight,
            ),
            **LEAST_SQUARES_KWARGS,
        )
        state = res.x

        beta, _, site_log_params = unpack_state(state, n_sites)
        mu_log_t_min = pooled_log_t_min_mean(site_log_params)
        state[1] = mu_log_t_min

        diagnostics = evaluate_site_fits(
            site_log_params,
            beta,
            observed_fm,
            observed_turnover,
        )
        variance_log_t_min = variance_from_site_params(site_log_params, mu_log_t_min)
        shrinkage_precision = 1.0 / (variance_log_t_min + sigma_floor ** 2)
        joint_score = profiled_joint_score(
            diagnostics['data_objective'],
            variance_log_t_min,
            sigma_floor=sigma_floor,
            data_weight=data_weight,
        )

        history_rows.append({
            'outer_iter': outer_iter,
            'beta': beta,
            'mu_log_t_min': mu_log_t_min,
            'std_log_t_min': np.sqrt(variance_log_t_min),
            'variance_log_t_min': variance_log_t_min,
            'shrinkage_precision': shrinkage_precision,
            'total_data_objective': float(np.sum(diagnostics['data_objective'])),
            'max_data_objective': float(np.max(diagnostics['data_objective'])),
            'profiled_joint_score': joint_score,
            'nfev': res.nfev,
            'cost': res.cost,
            'success': res.success,
            'message': res.message,
        })

        beta_change = abs(beta - prev_beta)
        variance_change = abs(variance_log_t_min - prev_variance) / (prev_variance + 1e-12)
        print(
            f'iter={outer_iter:02d} beta={beta:.5f} '
            f'std(log_t_min)={np.sqrt(variance_log_t_min):.5f} '
            f'max_obj={np.max(diagnostics["data_objective"]):.3e}'
        )

        prev_beta = beta
        prev_variance = variance_log_t_min
        if beta_change < outer_tol and variance_change < outer_tol:
            break

    beta, mu_log_t_min, site_log_params = unpack_state(state, n_sites)
    final_diagnostics = evaluate_site_fits(
        site_log_params,
        beta,
        observed_fm,
        observed_turnover,
    )

    return {
        'beta': beta,
        'mu_log_t_min': mu_log_t_min,
        'sigma_log_t_min': np.sqrt(variance_from_site_params(site_log_params, mu_log_t_min)),
        'site_log_params': site_log_params,
        'diagnostics': final_diagnostics,
        'history': pd.DataFrame(history_rows),
        'independent_log_params': independent_log_params,
        'independent_objectives': independent_objectives,
        'beta_prior_scale': beta_prior_scale,
        'mu_prior_mean': mu_prior_mean,
        'mu_prior_scale': mu_prior_scale,
        'data_weight': data_weight,
    }


def results_to_dataframe(merged_site_data, fit_result):
    """Convert the hierarchical fit output to a site-level DataFrame."""
    site_log_params = fit_result['site_log_params']
    n_sites = site_log_params.shape[0]
    t_min = np.zeros(n_sites, dtype=float)
    t_max = np.zeros(n_sites, dtype=float)
    independent_t_min = np.zeros(n_sites, dtype=float)
    independent_t_max = np.zeros(n_sites, dtype=float)

    for site_idx in range(n_sites):
        t_min[site_idx], t_max[site_idx] = unpack_log_params(site_log_params[site_idx])
        independent_t_min[site_idx], independent_t_max[site_idx] = unpack_log_params(
            fit_result['independent_log_params'][site_idx]
        )

    result_df = merged_site_data.copy()
    result_df['beta'] = fit_result['beta']
    result_df['t_min'] = t_min
    result_df['t_max'] = t_max
    result_df['log_t_min'] = site_log_params[:, 0]
    result_df['log_t_gap'] = site_log_params[:, 1]
    result_df['modeled_T'] = fit_result['diagnostics']['modeled_turnover']
    result_df['modeled_14C'] = fit_result['diagnostics']['modeled_14c']
    result_df['turnover_rel_error'] = fit_result['diagnostics']['turnover_rel_error']
    result_df['fm_rel_error'] = fit_result['diagnostics']['fm_rel_error']
    result_df['data_objective'] = fit_result['diagnostics']['data_objective']
    result_df['params_valid'] = fit_result['diagnostics']['params_valid']
    result_df['pooled_log_t_min_mean'] = fit_result['mu_log_t_min']
    result_df['pooled_log_t_min_std'] = fit_result['sigma_log_t_min']
    result_df['independent_t_min'] = independent_t_min
    result_df['independent_t_max'] = independent_t_max
    result_df['independent_log_t_min'] = fit_result['independent_log_params'][:, 0]
    result_df['independent_data_objective'] = fit_result['independent_objectives']
    result_df['t_min_shrinkage_ratio'] = result_df['t_min'] / result_df['independent_t_min']
    return result_df


def summary_to_dataframe(merged_site_data, fit_result, sigma_floor):
    """One-row summary of the shared-beta hierarchical fit."""
    independent_log_t_min = fit_result['independent_log_params'][:, 0]
    pooled_log_t_min = fit_result['site_log_params'][:, 0]
    diagnostics = fit_result['diagnostics']

    independent_std = float(np.std(independent_log_t_min))
    pooled_std = float(np.std(pooled_log_t_min))
    pooled_var = float(np.mean((pooled_log_t_min - fit_result['mu_log_t_min']) ** 2))
    profiled_score = profiled_joint_score(
        diagnostics['data_objective'],
        pooled_var,
        sigma_floor=sigma_floor,
        data_weight=fit_result['data_weight'],
    )

    return pd.DataFrame([{
        'n_sites': len(merged_site_data),
        'shared_beta': fit_result['beta'],
        'pooled_log_t_min_mean': fit_result['mu_log_t_min'],
        'pooled_log_t_min_std': pooled_std,
        'independent_log_t_min_std': independent_std,
        'log_t_min_std_reduction': independent_std - pooled_std,
        'sigma_floor': sigma_floor,
        'beta_prior_scale': fit_result['beta_prior_scale'],
        'mu_prior_mean': fit_result['mu_prior_mean'],
        'mu_prior_scale': fit_result['mu_prior_scale'],
        'data_weight': fit_result['data_weight'],
        'total_data_objective': float(np.sum(diagnostics['data_objective'])),
        'max_data_objective': float(np.max(diagnostics['data_objective'])),
        'mean_abs_turnover_rel_error': float(np.mean(np.abs(diagnostics['turnover_rel_error']))),
        'mean_abs_fm_rel_error': float(np.mean(np.abs(diagnostics['fm_rel_error']))),
        'profiled_joint_score': profiled_score,
    }])


def parse_args():
    """Command-line configuration for the hierarchical calibration run."""
    parser = argparse.ArgumentParser(
        description='Shared-beta empirical-Bayes fit for the generalized power law model.'
    )
    parser.add_argument(
        '--input-path',
        default='results/all_sites_14C_turnover.csv',
        help='CSV with fm and turnover observations for each site.',
    )
    parser.add_argument(
        '--output-dir',
        default='results/03_calibrate_models',
        help='Directory for the site-level and summary outputs.',
    )
    parser.add_argument(
        '--output-prefix',
        default='general_powerlaw_model_hierarchical',
        help='Prefix used for all output filenames.',
    )
    parser.add_argument(
        '--beta-init',
        type=float,
        default=DEFAULT_BETA,
        help='Starting value for the shared beta parameter.',
    )
    parser.add_argument(
        '--beta-min',
        type=float,
        default=0.05,
        help='Lower bound for the shared beta parameter.',
    )
    parser.add_argument(
        '--beta-max',
        type=float,
        default=0.99,
        help='Upper bound for the shared beta parameter.',
    )
    parser.add_argument(
        '--sigma-floor',
        type=float,
        default=DEFAULT_SIGMA_FLOOR,
        help='Minimum pooled std(log(t_min)) used to avoid degenerate collapse.',
    )
    parser.add_argument(
        '--max-outer-iter',
        type=int,
        default=DEFAULT_MAX_OUTER_ITER,
        help='Maximum number of empirical-Bayes outer updates.',
    )
    parser.add_argument(
        '--outer-tol',
        type=float,
        default=DEFAULT_OUTER_TOL,
        help='Convergence tolerance for beta and pooled variance updates.',
    )
    parser.add_argument(
        '--beta-prior-scale',
        type=float,
        default=DEFAULT_BETA_PRIOR_SCALE,
        help='Weak prior scale around the initial beta used to avoid boundary collapse.',
    )
    parser.add_argument(
        '--data-weight',
        type=float,
        default=DEFAULT_DATA_WEIGHT,
        help='Relative weight on data reconstruction versus pooled shrinkage. Raise it to preserve fit more strongly; lower it to enforce more pooling.',
    )
    parser.add_argument(
        '--gamma-default',
        action='store_true',
        help='Initialize beta with exp(-GAMMA) instead of the explicit default.',
    )
    return parser.parse_args()


def main():
    """Run the shared-beta hierarchical fit and save site-level diagnostics."""
    args = parse_args()
    beta_init = np.exp(-GAMMA) if args.gamma_default else args.beta_init

    merged_site_data = pd.read_csv(args.input_path)
    fit_result = hierarchical_map_fit(
        merged_site_data,
        beta_init=beta_init,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        sigma_floor=args.sigma_floor,
        max_outer_iter=args.max_outer_iter,
        outer_tol=args.outer_tol,
        beta_prior_scale=args.beta_prior_scale,
        data_weight=args.data_weight,
    )

    site_results = results_to_dataframe(merged_site_data, fit_result)
    summary_df = summary_to_dataframe(merged_site_data, fit_result, sigma_floor=args.sigma_floor)
    history_df = fit_result['history']

    site_output_path = path.join(args.output_dir, f'{args.output_prefix}_site_results.csv')
    summary_output_path = path.join(args.output_dir, f'{args.output_prefix}_summary.csv')
    history_output_path = path.join(args.output_dir, f'{args.output_prefix}_history.csv')

    site_results.to_csv(site_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)
    history_df.to_csv(history_output_path, index=False)

    print('Finished shared-beta hierarchical calibration.')
    print(f'shared beta = {fit_result["beta"]:.5f}')
    print(f'pooled std(log_t_min) = {fit_result["sigma_log_t_min"]:.5f}')
    print(f'max per-site data objective = {summary_df.loc[0, "max_data_objective"]:.3e}')
    print(f'wrote site results to {site_output_path}')
    print(f'wrote summary to {summary_output_path}')
    print(f'wrote iteration history to {history_output_path}')


if __name__ == '__main__':
    main()