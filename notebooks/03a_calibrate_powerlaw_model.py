import pandas as pd
import numpy as np
from notebooks.models import PowerLawDisKin
from os import path
from scipy.integrate import quad
from scipy.optimize import minimize
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

    # Create an instance of the PowerLawDisKin model with the given parameters
    model = PowerLawDisKin(a, b)
    
    # Calculate the predicted 14C ratio and turnover
    predicted_14C_ratio = quad(model.radiocarbon_age_integrand,
                               0, np.inf, limit=1500, epsabs=1e-3)[0]

    # Calculate the difference between the predicted and observed data
    diff_14C = np.nansum((predicted_14C_ratio - merged_site_data['fm'])**2)
    diff_turnover = np.nansum((model.T - merged_site_data['turnover'])**2)
    # For the above line, _all.py has 
    # diff_turnover = np.nansum((model.T - merged_site_data['turnover']*2)**2)
    # why the factor of 2???
    
    # Return the sum of squared differences
    return diff_14C + diff_turnover

#%% Load the data

merged_site_data = pd.read_csv('results/all_sites_14C_turnover.csv')

# initial guess for the parameters
initial_guess = [0.5, 10000]

# optimize the parameters using a simple optimization method
result = []

# iterate over each row in the merged_site_data DataFrame
for i, row in tqdm(merged_site_data.iterrows(), total=len(merged_site_data)):
    
    # Minimize the objective function for each site
    res = minimize(objective_function, initial_guess, args=(row,), method='Nelder-Mead')
    
    # Append the optimized parameters to the result list
    result.append([res.x,res.fun])

# Convert the result list to a DataFrame
result_df = pd.DataFrame(result,columns=['params','objective_value'],index=merged_site_data.index)

# Unpack the parameters into separate columns
result_df[['tau_0','tau_inf']] = result_df['params'].apply(lambda x: pd.Series(x,index = ['tau_0','tau_inf']))
result_df.drop(columns ='params',inplace=True)

result_df['modeled_tau'] = result_df.apply(lambda x: PowerLawDisKin(x['tau_0'], x['tau_inf']).T, axis=1)
result_df['modeled_14C'] = result_df.apply(lambda x: quad(PowerLawDisKin(x['tau_0'], x['tau_inf']).radiocarbon_age_integrand, 0, np.inf, limit=1500,epsabs=1e-3)[0], axis=1)

merged_result_df = pd.concat([result_df, merged_site_data[['fm', 'turnover']]], axis=1)

print(f'the Maximum objective value is {result_df["objective_value"].max():.3f}')

# Save the result to a CSV file
output_dir = 'results/03_calibrate_models'
output_fname = 'powerlaw_model_optimization_results.csv'
output_path = path.join(output_dir, output_fname)
merged_result_df.to_csv(output_path, index=False)
