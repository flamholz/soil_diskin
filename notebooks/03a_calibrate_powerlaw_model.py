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


def fit_row(row):
    """Fit one site with ordered parameters in log space."""
    res = least_squares(
        objective_residuals,
        initial_log_guess(row),
        args=(row,),
        method='trf',
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    return [unpack_log_params(res.x), objective_function(res.x, row)]


def parse_results(result,index):
    """Convert optimization results to a DataFrame and compute additional metrics."""

    # Convert the result list to a DataFrame
    df = pd.DataFrame(result,columns=['params','objective_value'],index=index)

    # Unpack the parameters into separate columns
    df[['t_min','t_max']] = df['params'].apply(lambda x: pd.Series(x,index = ['t_min','t_max']))
    df.drop(columns ='params',inplace=True)

    model_x = lambda x: PowerLawDisKin(x['t_min'], x['t_max'])
    df['modeled_tau'] = df.apply(lambda x: model_x(x).T, axis=1)
    df['modeled_14C'] = df.apply(lambda x: quad(model_x(x).radiocarbon_age_integrand, 0, np.inf, limit=1500,epsabs=1e-3)[0], axis=1)
    df['params_valid'] = df.apply(lambda x: model_x(x).params_valid(), axis=1)

    return df

#%% Load the data

merged_site_data = pd.read_csv('results/all_sites_14C_turnover.csv')

# optimize the parameters using a simple optimization method
result = []

#%% Perform optimization for each site
# iterate over each row in the merged_site_data DataFrame
for i, row in tqdm(merged_site_data.iterrows(), total=len(merged_site_data)):
    result.append(fit_row(row))

# Convert the result list to a DataFrame
result_df = parse_results(result, merged_site_data.index)

#%% Perform optimization for the 5% and 95% uncertainty bounds
results_05 = []
results_95 = []
backfilled_sites = merged_site_data[merged_site_data['turnover_q05'].notna() & merged_site_data['turnover_q95'].notna()]
for i, row in tqdm(backfilled_sites.iterrows(), total=len(backfilled_sites)):
    row_05 = row.copy()
    row_05['turnover'] = row.loc['turnover_q05']
    row_95 = row.copy()
    row_95['turnover'] = row.loc['turnover_q95']
    results_05.append(fit_row(row_05))
    results_95.append(fit_row(row_95))

result_df_05 = parse_results(results_05, backfilled_sites.index)
result_df_95 = parse_results(results_95, backfilled_sites.index)

merged_result_df = pd.concat([result_df, merged_site_data[['fm', 'turnover']]], axis=1)
merged_result_df = pd.merge(merged_result_df, result_df_05.add_suffix('_05'), left_index=True, right_index=True, how='left')
merged_result_df = pd.merge(merged_result_df, result_df_95.add_suffix('_95'), left_index=True, right_index=True, how='left')

print(f'the Maximum objective value is {result_df["objective_value"].max():.3f}')

# Save the result to a CSV file
output_dir = 'results/03_calibrate_models'
output_fname = 'powerlaw_model_optimization_results.csv'
output_path = path.join(output_dir, output_fname)
merged_result_df.to_csv(output_path, index=False)
