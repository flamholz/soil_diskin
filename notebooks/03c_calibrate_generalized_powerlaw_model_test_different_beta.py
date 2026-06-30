import pandas as pd
import numpy as np
from os import makedirs, path
from scipy.integrate import quad
from scipy.optimize import least_squares
from soil_diskin.continuum_models import GeneralPowerLawDisKin
from tqdm import tqdm

"""
Script should be run from the project root directory.

Calibrates the Generalized Power Law model from Rothman, PNAS 2025.
"""

INITIAL_BETA = 0.76
BETA_LOWER_BOUND = 1e-6
BETA_UPPER_BOUND = 1.0
T_MIN_VALUES = np.logspace(np.log10(0.1), np.log10(10), 10)
OPTIMIZER_KWARGS = {
    'method': 'trf',
    'ftol': 1e-12,
    'xtol': 1e-12,
    'gtol': 1e-12,
}


# Define the residuals and objective for optimization.
# We solve t_max as log(t_max - t_min) because it spans orders of magnitude,
# and direct optimization in raw units frequently stalls near the initial t_max.
def unpack_params(params, fixed_t_min):
    """Convert optimization parameters into model parameters."""
    beta, log_t_gap = params
    t_max = fixed_t_min + np.exp(log_t_gap)
    return beta, fixed_t_min, t_max


def objective_residuals(params, fixed_t_min, merged_site_data):
    """
    Residual vector comparing model predictions and observations.

    Parameters
    ----------
    params : list
        List of optimization parameters [beta, log_t_gap].
    fixed_t_min : float
        The fixed t_min parameter for the GeneralPowerLawDisKin model.
    merged_site_data : pd.DataFrame
        DataFrame containing the observed data for each site, including 'fm' (14C ratio) and 'turnover'.
    
    Returns
    -------
    np.ndarray
        Relative residuals for the 14C ratio and turnover.
    """
    # Unpack the ordered parameters
    beta, t_min, t_max = unpack_params(params, fixed_t_min)

    # Create an instance of the GeneralPowerLawDisKin model with the given parameters
    model = GeneralPowerLawDisKin(t_min=t_min, t_max=t_max, beta=beta)
    
    # Calculate the predicted 14C ratio and turnover
    predicted_14C_ratio = quad(model.radiocarbon_age_integrand, 0,
                               np.inf, limit=1500,epsabs=1e-3)[0]
    
    # Calculate the relative difference between the predicted and observed data
    relative_diff_14C = (
        predicted_14C_ratio - merged_site_data['fm']) / (merged_site_data['fm'] + 1e-6)
    relative_diff_turnover = (
        model.T - merged_site_data['turnover']) / (merged_site_data['turnover'] + 1e-6)

    return np.array([relative_diff_14C, relative_diff_turnover])


def objective_function(params, fixed_t_min, merged_site_data):
    """Return the sum of squared relative residuals for diagnostics/output."""
    residuals = objective_residuals(params, fixed_t_min, merged_site_data)
    return float(np.dot(residuals, residuals))


def initial_guess(site_data):
    """Choose a scale-aware initial guess for the site being optimized."""
    t_gap_guess = max(site_data['turnover'], 1000)
    return np.array([INITIAL_BETA, np.log(t_gap_guess)])


# Helper functions to calculate values on DataFrame rows
def calc_modeled_T(row):
    """Calculate the modeled turnover time T from a row of parameters."""
    model = GeneralPowerLawDisKin(row['t_min'], row['t_max'], row['beta'])
    return model.T


def calc_modeled_14C(row):
    """Numerically integrate to get the modeled 14C ratio from a row of parameters."""
    model = GeneralPowerLawDisKin(row['t_min'], row['t_max'], row['beta'])
    return quad(model.radiocarbon_age_integrand, 0, np.inf, limit=1500,epsabs=1e-3)[0]


def calc_params_valid(row):
    """Returns True if the model parameters are valid for this DataFrame row."""
    model = GeneralPowerLawDisKin(row['t_min'], row['t_max'], row['beta'])
    return model.params_valid()


def results_to_dataframe(results, fixed_t_min, merged_site_data):
    """
    Convert optimization results to a DataFrame.
    
    Parameters
    ----------
    results : list
        List of optimization results.
    fixed_t_min : float
        The fixed t_min parameter used in the model optimization.
    merged_site_data : pd.DataFrame
        DataFrame containing the observed data for each site.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the optimization results.
    """
    result_df = pd.DataFrame(results, columns=['params', 'objective_value'],
                             index=merged_site_data.index)

    # Unpack the parameters into separate columns
    index_names = 'beta,t_max'.split(',')
    result_df[index_names] = result_df['params'].apply(
        lambda x: pd.Series(x, index=index_names))
    result_df.drop(columns='params', inplace=True)

    result_df['t_min'] = fixed_t_min
    result_df = result_df[['t_min', 't_max', 'beta', 'objective_value']]
    result_df['modeled_T'] = result_df.apply(calc_modeled_T, axis=1)
    result_df['modeled_14C'] = result_df.apply(calc_modeled_14C, axis=1)
    result_df['params_valid'] = result_df.apply(calc_params_valid, axis=1)
    merged_result_df = pd.concat([result_df, merged_site_data[['fm', 'turnover']]], axis=1)

    return merged_result_df


def format_t_min_for_filename(t_min):
    """Format t_min as a readable filename token."""
    return f'{t_min:.3f}'.replace('.', 'p')


def fit_sites_for_t_min(fixed_t_min, merged_site_data):
    """Fit beta and t_max for each site at one fixed t_min value."""
    results = []
    bounds = ([BETA_LOWER_BOUND, -np.inf], [BETA_UPPER_BOUND, np.inf])

    print(f'Running optimization for generalized power law with t_min = {fixed_t_min:.3f}...')
    for i, row in tqdm(merged_site_data.iterrows(), total=len(merged_site_data)):
        res = least_squares(
            objective_residuals,
            initial_guess(row),
            bounds=bounds,
            args=(fixed_t_min, row),
            **OPTIMIZER_KWARGS,
        )

        beta, _, t_max = unpack_params(res.x, fixed_t_min)
        results.append([
            (beta, t_max),
            objective_function(res.x, fixed_t_min, row),
        ])

    return results_to_dataframe(results, fixed_t_min, merged_site_data)


# Load the data
merged_site_data = pd.read_csv('results/all_sites_14C_turnover.csv')

output_dir = 'results/03_calibrate_models/'
makedirs(output_dir, exist_ok=True)

for t_min in T_MIN_VALUES:
    merged_result_df = fit_sites_for_t_min(t_min, merged_site_data)
    print(
        f'the Maximum objective value for t_min = {t_min:.3f} '
        f'is {merged_result_df["objective_value"].max():.3f}'
    )

    t_min_token = format_t_min_for_filename(t_min)
    fname = f'general_powerlaw_model_optimization_results_tmin_{t_min_token}.csv'
    output_path = path.join(output_dir, fname)
    merged_result_df.to_csv(output_path, index=False)
