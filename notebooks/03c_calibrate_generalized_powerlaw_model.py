import pandas as pd
import numpy as np
from os import path
from scipy.integrate import quad
from scipy.optimize import minimize
from soil_diskin.constants import GAMMA
from soil_diskin.continuum_models import GeneralPowerLawDisKin
from tqdm import tqdm

"""
Script should be run from the project root directory.

Calibrates the Generalized Power Law model from Rothman, PNAS 2025.
"""

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
    t_min, t_max, beta = params

    # Create an instance of the GeneralPowerLawDisKin model with the given parameters
    model = GeneralPowerLawDisKin(t_min=t_min, t_max=t_max, beta=beta)
    
    # Calculate the predicted 14C ratio and turnover
    predicted_14C_ratio = quad(model.radiocarbon_age_integrand, 0,
                               np.inf, limit=1500,epsabs=1e-3)[0]
    
    # Calculate the relative difference between the predicted and observed data
    diff_14C = np.nansum((predicted_14C_ratio - merged_site_data['fm']))
    total_14C = np.nansum(merged_site_data['fm'] + 1e-6)
    relative_diff_14C = diff_14C / total_14C
    diff_turnover = np.nansum((model.T - merged_site_data['turnover']))
    total_turnover = np.nansum(merged_site_data['turnover'] + 1e-6)
    relative_diff_turnover = diff_turnover / total_turnover
    
    # Return the sum of squared differences
    return relative_diff_14C**2 + relative_diff_turnover**2


# Helper functions to calculate modeled T and 14C for rows
def calc_modeled_T(row):
    model = GeneralPowerLawDisKin(row['t_min'], row['t_max'], row['beta'])
    return model.T


def calc_modeled_14C(row):
    model = GeneralPowerLawDisKin(row['t_min'], row['t_max'], row['beta'])
    return quad(model.radiocarbon_age_integrand, 0, np.inf, limit=1500,epsabs=1e-3)[0]


def results_to_dataframe(results, merged_site_data):
    """
    Convert optimization results to a DataFrame.
    
    Parameters
    ----------
    results : list
        List of optimization results.
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
    index_names = 't_min,t_max,beta'.split(',')
    result_df[index_names] = result_df['params'].apply(lambda x: pd.Series(x, index=index_names))
    result_df.drop(columns ='params',inplace=True)

    result_df['modeled_T'] = result_df.apply(calc_modeled_T, axis=1)
    result_df['modeled_14C'] = result_df.apply(calc_modeled_14C, axis=1)
    merged_result_df = pd.concat([result_df, merged_site_data[['fm', 'turnover']]], axis=1)

    return merged_result_df


# Load the data
merged_site_data = pd.read_csv('results/all_sites_14C_turnover.csv')

# initial guess for the parameters
beta = np.exp(-GAMMA) # gamma is the Euler-Mascheroni constant
initial_guess = [0.0005, 1000, beta]

# optimize the parameters using a simple optimization method
results = []

# iterate over each row in the merged_site_data DataFrame
print('Running optimization for generalized power law with b = np.exp(-GAMMA)...')
for i, row in tqdm(merged_site_data.iterrows(), total=len(merged_site_data)):
    # Minimize the objective function for each site
    res = minimize(objective_function, initial_guess, args=(row,), method='Nelder-Mead')
    
    # Append the optimized parameters to the result list
    results.append([res.x, res.fun])

merged_result_df = results_to_dataframe(results, merged_site_data)
print(f'the Maximum objective value is {merged_result_df["objective_value"].max():.3f}')


# Now run it all again, but with b = np.exp(-GAMMA) / 2
results_2 = []

beta = np.exp(-GAMMA) / 2
initial_guess = [0.0005, 1000, beta]

# iterate over each row in the merged_site_data DataFrame
print('Running optimization for generalized power law with b = np.exp(-GAMMA) / 2...')
for i, row in tqdm(merged_site_data.iterrows(), total=len(merged_site_data)):
    
    # Minimize the objective function for each site
    res = minimize(objective_function, initial_guess, args=(row,), method='Nelder-Mead')
    
    # Append the optimized parameters to the result list
    results_2.append([res.x,res.fun])

merged_result_df_2 = results_to_dataframe(results_2, merged_site_data)
print(f'the Maximum objective value is {merged_result_df_2["objective_value"].max():.3f}')

# Save the two sets of results to different CSV files
output_dir = 'results/03_calibrate_models/'

fname1 = 'general_powerlaw_model_optimization_results.csv'
output_path = path.join(output_dir, fname1)
merged_result_df.to_csv(output_path, index=False)

fname2 = 'general_powerlaw_model_optimization_results_beta_half.csv'
output_path = path.join(output_dir, fname2)
merged_result_df_2.to_csv(output_path, index=False)
