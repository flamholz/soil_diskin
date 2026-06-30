import pandas as pd
import numpy as np
from os import path
from scipy.integrate import quad
from scipy.optimize import least_squares
from soil_diskin.continuum_models import PowerLawDisKin
from tqdm import tqdm

# Define the residuals and objective for optimization.
# We solve in log space because t_min and t_max span orders of magnitude, and
# direct optimization in raw units is less stable than an ordered parameterization.
def unpack_log_params(log_params):
    """Convert unconstrained log parameters into ordered model parameters."""
    log_t_min, log_t_gap = log_params
    t_min = np.exp(log_t_min)
    t_max = t_min + np.exp(log_t_gap)
    return t_min, t_max


def objective_residuals(log_params, merged_site_data):
    """
    Residual vector comparing model predictions and the observed data.
    
    Parameters
    ----------
    log_params : list
        List of log parameters [log_t_min, log_t_gap].
    merged_site_data : pd.DataFrame
        DataFrame containing the observed data for each site, including 'fm' (14C ratio) and 'turnover'.
    
    Returns
    -------
    np.ndarray
        Relative residuals for the 14C ratio and turnover.
    """
    # Unpack the ordered parameters
    t_min, t_max = unpack_log_params(log_params)

    # Create an instance of the PowerLawDisKin model with the given parameters
    model = PowerLawDisKin(t_min, t_max)
    
    # Calculate the predicted 14C ratio and turnover
    predicted_14C_ratio = quad(model.radiocarbon_age_integrand,
                               0, np.inf, limit=1500, epsabs=1e-3)[0]

    # Calculate the relative residuals between the predicted and observed data
    relative_diff_14C = (
        predicted_14C_ratio - merged_site_data['fm']) / (merged_site_data['fm'] + 1e-6)
    relative_diff_turnover = (
        model.T - merged_site_data['turnover']) / (merged_site_data['turnover'] + 1e-6)

    return np.array([relative_diff_14C, relative_diff_turnover])


def objective_function(log_params, merged_site_data):
    """Return the sum of squared relative residuals for diagnostics/output."""
    residuals = objective_residuals(log_params, merged_site_data)
    return float(np.dot(residuals, residuals))


def initial_log_guess(site_data):
    """Choose a scale-aware initial guess for the site being optimized."""
    t_min_guess = max(1e-6, site_data['turnover'] / 1000)
    t_gap_guess = max(site_data['turnover'], 1000)
    return np.log([t_min_guess, t_gap_guess])

#%% Load the data

merged_site_data = pd.read_csv('results/all_sites_14C_turnover.csv')

# optimize the parameters using a simple optimization method
result = []

# iterate over each row in the merged_site_data DataFrame
for i, row in tqdm(merged_site_data.iterrows(), total=len(merged_site_data)):
    # Solve for ordered parameters in log space to avoid scale-related stalls.
    res = least_squares(
        objective_residuals,
        initial_log_guess(row),
        args=(row,),
        method='trf',
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    
    # Append the optimized parameters to the result list
    result.append([unpack_log_params(res.x), objective_function(res.x, row)])

# Convert the result list to a DataFrame
result_df = pd.DataFrame(result,columns=['params','objective_value'],index=merged_site_data.index)

# Unpack the parameters into separate columns
result_df[['t_min','t_max']] = result_df['params'].apply(lambda x: pd.Series(x,index = ['t_min','t_max']))
result_df.drop(columns ='params',inplace=True)

result_df['modeled_tau'] = result_df.apply(lambda x: PowerLawDisKin(x['t_min'], x['t_max']).T, axis=1)
result_df['modeled_14C'] = result_df.apply(lambda x: quad(PowerLawDisKin(x['t_min'], x['t_max']).radiocarbon_age_integrand, 0, np.inf, limit=1500,epsabs=1e-3)[0], axis=1)
result_df['params_valid'] = result_df.apply(lambda x: PowerLawDisKin(x['t_min'], x['t_max']).params_valid(), axis=1)

merged_result_df = pd.concat([result_df, merged_site_data[['fm', 'turnover']]], axis=1)

print(f'the Maximum objective value is {result_df["objective_value"].max():.3f}')

# Save the result to a CSV file
output_dir = 'results/03_calibrate_models'
output_fname = 'powerlaw_model_optimization_results.csv'
output_path = path.join(output_dir, output_fname)
merged_result_df.to_csv(output_path, index=False)
