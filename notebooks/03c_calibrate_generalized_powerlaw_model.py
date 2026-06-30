import pandas as pd
import numpy as np
from os import path
from scipy.integrate import quad
from scipy.optimize import least_squares
from soil_diskin.constants import GAMMA
from soil_diskin.continuum_models import GeneralPowerLawDisKin
from tqdm import tqdm

"""
Script should be run from the project root directory.

Calibrates the Generalized Power Law model from Rothman, PNAS 2025.
"""

# Define the residuals and objective for optimization.
# We solve in log space because t_min and t_max span orders of magnitude, and
# direct optimization in raw units frequently stalls near the initial t_max.
def unpack_log_params(log_params):
    """Convert unconstrained log parameters into ordered model parameters."""
    log_t_min, log_t_gap = log_params
    t_min = np.exp(log_t_min)
    t_max = t_min + np.exp(log_t_gap)
    return t_min, t_max


def objective_residuals(log_params, beta, merged_site_data):
    """
    Residual vector comparing model predictions and observations.

    Note: 'beta' is the power law exponent parameter in the GeneralPowerLawDisKin model.
    We do not alter it during optimization, but pass it in for model instantiation.
    
    Parameters
    ----------
    log_params : list
        List of log parameters [log_t_min, log_t_gap].
    beta : float
        The beta parameter for the GeneralPowerLawDisKin model.
    merged_site_data : pd.DataFrame
        DataFrame containing the observed data for each site, including 'fm' (14C ratio) and 'turnover'.
    
    Returns
    -------
    np.ndarray
        Relative residuals for the 14C ratio and turnover.
    """
    # Unpack the ordered parameters
    t_min, t_max = unpack_log_params(log_params)

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


def objective_function(log_params, beta, merged_site_data):
    """Return the sum of squared relative residuals for diagnostics/output."""
    residuals = objective_residuals(log_params, beta, merged_site_data)
    return float(np.dot(residuals, residuals))


def initial_log_guess(site_data):
    """Choose a scale-aware initial guess for the site being optimized."""
    t_min_guess = max(1e-6, site_data['turnover'] / 1000)
    t_gap_guess = max(site_data['turnover'], 1000)
    return np.log([t_min_guess, t_gap_guess])


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


def results_to_dataframe(results, beta, merged_site_data):
    """
    Convert optimization results to a DataFrame.
    
    Parameters
    ----------
    results : list
        List of optimization results.
    beta : float
        The beta parameter used in the model optimization.
    merged_site_data : pd.DataFrame
        DataFrame containing the observed data for each site.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the optimization results.
    """
    result_df = pd.DataFrame(results, columns=['params','objective_value'],
                             index=merged_site_data.index)

    # Unpack the parameters into separate columns
    index_names = 't_min,t_max'.split(',')
    result_df[index_names] = result_df['params'].apply(lambda x: pd.Series(x, index=index_names))
    result_df.drop(columns ='params',inplace=True)

    result_df['beta'] = beta
    result_df['modeled_T'] = result_df.apply(calc_modeled_T, axis=1)
    result_df['modeled_14C'] = result_df.apply(calc_modeled_14C, axis=1)
    result_df['params_valid'] = result_df.apply(calc_params_valid, axis=1)
    merged_result_df = pd.concat([result_df, merged_site_data[['fm', 'turnover']]], axis=1)

    return merged_result_df


# Load the data
merged_site_data = pd.read_csv('results/all_sites_14C_turnover.csv')

# initial guess for the parameters
beta = np.exp(-GAMMA) # gamma is the Euler-Mascheroni constant

# optimize the parameters using a simple optimization method
results = []

# iterate over each row in the merged_site_data DataFrame
print('Running optimization for generalized power law with b = np.exp(-GAMMA)...')
for i, row in tqdm(merged_site_data.iterrows(), total=len(merged_site_data)):
    # Solve for ordered parameters in log space to avoid scale-related stalls.
    my_args = (beta, row)
    res = least_squares(
        objective_residuals,
        initial_log_guess(row),
        args=my_args,
        method='trf',
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    
    # Append the optimized parameters to the result list
    results.append([unpack_log_params(res.x), objective_function(res.x, beta, row)])

merged_result_df = results_to_dataframe(results, beta, merged_site_data)
print(f'the Maximum objective value is {merged_result_df["objective_value"].max():.3f}')


# Now run it all again, but with b = np.exp(-GAMMA) / 2
results_2 = []
beta = np.exp(-GAMMA) / 2

# iterate over each row in the merged_site_data DataFrame
print('Running optimization for generalized power law with b = np.exp(-GAMMA) / 2...')
for i, row in tqdm(merged_site_data.iterrows(), total=len(merged_site_data)):
    # Solve for ordered parameters in log space to avoid scale-related stalls.
    my_args = (beta, row)
    res = least_squares(
        objective_residuals,
        initial_log_guess(row),
        args=my_args,
        method='trf',
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    
    # Append the optimized parameters to the result list
    results_2.append([unpack_log_params(res.x), objective_function(res.x, beta, row)])

merged_result_df_2 = results_to_dataframe(results_2, beta, merged_site_data)
print(f'the Maximum objective value is {merged_result_df_2["objective_value"].max():.3f}')

# Save the two sets of results to different CSV files
output_dir = 'results/03_calibrate_models/'

fname1 = 'general_powerlaw_model_optimization_results.csv'
output_path = path.join(output_dir, fname1)
merged_result_df.to_csv(output_path, index=False)

fname2 = 'general_powerlaw_model_optimization_results_beta_half.csv'
output_path = path.join(output_dir, fname2)
merged_result_df_2.to_csv(output_path, index=False)
