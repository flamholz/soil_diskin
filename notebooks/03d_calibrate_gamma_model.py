import pandas as pd
import numpy as np
from os import path

from scipy.integrate import quad
from scipy.optimize import minimize
from soil_diskin.continuum_models import GammaDisKin
from tqdm import tqdm

# Define the objective function for optimization
# optimize the two parameters of the model to match the turnover and 14C data
def objective_function(params, merged_site_data):
    """
    Objective function to minimize the difference between the model predictions and the observed data.
    
    Parameters
    ----------
    params : list
        List of parameters [a, b] for the PowerLawDisKin model.
    merged_site_data : pd.DataFrame
        DataFrame containing the observed data for each site, including 'fm' (14C ratio) and 'turnover'.
    
    Returns
    -------
    float
        The sum of squared differences between the predicted and observed data.
    """
    
    # Unpack the parameters
    a, b = params

    # Create an instance of the GammaDisKin model with the given parameters
    model = GammaDisKin(a, b)
    
    # Calculate the predicted 14C ratio and turnover
    predicted_14C_ratio = quad(
        model.radiocarbon_age_integrand,
        0, np.inf, limit=1500,epsabs=1e-3)[0]
    
    # Calculate the difference between the predicted and observed data
    diff_14C = np.nansum((predicted_14C_ratio - merged_site_data['fm']))
    total_14C = np.nansum(merged_site_data['fm'] + 1e-6)
    relative_diff_14C = diff_14C / total_14C
    diff_turnover = np.nansum((model.T - merged_site_data['turnover']))
    total_turnover = np.nansum(merged_site_data['turnover'] + 1e-6)
    relative_diff_turnover = diff_turnover / total_turnover

    # Return the sum of squared relative differences
    return relative_diff_14C**2 + relative_diff_turnover**2


def parse_results(result,index):
    """Convert optimization results to a DataFrame and compute additional metrics."""

    # Convert the result list to a DataFrame
    df = pd.DataFrame(result,columns=['params','objective_value'],index=index)

    # Unpack the parameters into separate columns
    df[['a','b']] = df['params'].apply(lambda x: pd.Series(x,index = ['a','b']))
    df.drop(columns ='params',inplace=True)
    df['modeled_tau'] = df.apply(lambda x: GammaDisKin(x['a'], x['b']).T, axis=1)
    df['modeled_14C'] = df.apply(lambda x: quad(GammaDisKin(x['a'], x['b']).radiocarbon_age_integrand, 0, np.inf, limit=1500,epsabs=1e-3)[0], axis=1)
    df['params_valid'] = df.apply(lambda x: GammaDisKin(x['a'], x['b']).params_valid(), axis=1)
    return df


# Load the data
merged_site_data = pd.read_csv('results/all_sites_14C_turnover.csv')

# initial guess for the parameters
initial_guess = [1.2, 0.5]
# initial_guess = [0.5, 10000]

# optimize the parameters using a simple optimization method
result = []

# iterate over each row in the merged_site_data DataFrame
for i, row in tqdm(merged_site_data.iterrows(), total=len(merged_site_data)):
    
    # Minimize the objective function for each site
    res = minimize(objective_function, initial_guess, args=(row,), method='Nelder-Mead', bounds=((1.00001, None), (0, None)))
    
    # Append the optimized parameters to the result list
    result.append([res.x,res.fun])

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
    res = minimize(objective_function, initial_guess, args=(row_05,), method='Nelder-Mead')
    results_05.append([res.x,res.fun])
    res = minimize(objective_function, initial_guess, args=(row_95,), method='Nelder-Mead')
    results_95.append([res.x,res.fun])

result_df_05 = parse_results(results_05, backfilled_sites.index)
result_df_95 = parse_results(results_95, backfilled_sites.index)

merged_result_df = pd.concat([result_df, merged_site_data[['fm', 'turnover']]], axis=1)
merged_result_df = pd.merge(merged_result_df, result_df_05.add_suffix('_05'), left_index=True, right_index=True, how='left')
merged_result_df = pd.merge(merged_result_df, result_df_95.add_suffix('_95'), left_index=True, right_index=True, how='left')

print(f'the Maximum objective value is {result_df["objective_value"].max():.3f}')

# Save the result to a CSV file
output_dir = 'results/03_calibrate_models'
output_fname = 'gamma_model_optimization_results.csv'
output_path = path.join(output_dir, output_fname)
merged_result_df.to_csv(output_path, index=False)
