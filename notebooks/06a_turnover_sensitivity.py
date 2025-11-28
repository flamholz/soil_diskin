# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

# %%
# Load libraries
import pandas as pd
import numpy as np
from soil_diskin.continuum_models import PowerLawDisKin, GammaDisKin
from os import path
from scipy.integrate import quad
from scipy.optimize import minimize
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

#TODO: refactor the objective function and prediction code with the code in step 03 for both the Powerlaw and Gamma models so that we don't have code duplication

# %%

# Define the objective function for optimization
# optimize the two parameters of the model to match the turnover and 14C data
def objective_function(params, model, merged_site_data, weights):
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

    # Create an instance of the PowerLawDisKin model with the given parameters
    model = model(*params)
    
    # Calculate the predicted 14C ratio and turnover
    predicted_14C_ratio = quad(model.radiocarbon_age_integrand,
                               0, np.inf, limit=1500, epsabs=1e-3)[0]

    # Calculate the difference between the predicted and observed data
    diff_14C = np.nansum((predicted_14C_ratio - merged_site_data['fm'])**2)
    diff_turnover = np.nansum((model.T - merged_site_data['turnover'])**2)
    
    # Return the sum of squared differences
    return weights[0] * diff_14C + weights[1] * diff_turnover

# %%
def generate_predictions(model, initial_guess, data):
    """
    Generate predictions for each site using the specified model and initial guess for parameters.

    Parameters
    ----------
    model : class
        The model class to use (e.g., PowerLawDisKin, GammaDisKin).
    initial_guess : list
        Initial guess for the model parameters.
    data : pd.DataFrame
        DataFrame containing the site data for which predictions are to be made.

    Returns
    -------
    list
        List of predictions for each site.
    """
    # optimize the parameters using a simple optimization method
        
        # Minimize the objective function for each site
    def generate_prediction(model, initial_guess, row):
        # First optimize the parameters for the given site
        
        # if isinstance(model, PowerLawDisKin):
        if model == PowerLawDisKin:
            res = minimize(objective_function, initial_guess, args=(model, row, [1, 1]), method='Nelder-Mead')
        elif model == GammaDisKin:
            res = minimize(objective_function, initial_guess, args=(model, row, [300, 1]), method='Nelder-Mead', bounds=((1.00001, None), (0, None)))
        else:
            print(model)
            raise ValueError("Model type not recognized for optimization.")
        # Generate the prediction using the optimized parameters
        prediction = model(*res.x).cdfA(row['Duration_labeling'])
        return prediction

    # Use parallel processing to generate predictions for all sites
    with parallel_backend('loky', n_jobs=-1):
        predictions = Parallel(verbose=1)(
                delayed(generate_prediction)(model, initial_guess, row) for i, row in data.iterrows()
                    )

    return predictions


# %%
# Load site data
site_data = pd.read_csv('results/processed_balesdent_2018.csv')
turnover_14C = pd.read_csv('results/all_sites_14C_turnover.csv')
merged_df = turnover_14C.join(site_data['Duration_labeling'])

# Define different turnover time scaling ratios to test sensitivity
ratios = [0.5, 1/1.5, 1, 1.5, 2]

# %%
# Generate predictions for each scaling for the Powerlaw model
pl_sensitivity_results = []
initial_guess = [0.5, 10000]
for ratio in ratios:
    df = merged_df.copy()
    df['turnover'] = df['turnover'] * ratio
    pl_sensitivity_results.append(generate_predictions(PowerLawDisKin, initial_guess, df))

# %%
# Generate predictions for each scaling for the Gamma model
gamma_sensitivity_results = []
initial_guess = [1.2, 0.5]
for ratio in ratios:
    df = merged_df.copy()
    df['turnover'] = df['turnover'] * ratio
    gamma_sensitivity_results.append(generate_predictions(GammaDisKin, initial_guess, df))

# %%
# Collect the results into DataFrames
pl_df = pd.concat([pd.DataFrame(x) for x in pl_sensitivity_results], axis=1, keys=ratios)
gamma_df = pd.concat([pd.DataFrame(x) for x in gamma_sensitivity_results], axis=1, keys=ratios)
pl_df.columns = pl_df.columns.droplevel(1)
gamma_df.columns = gamma_df.columns.droplevel(1)

# %%
# Save the results to CSV files
pl_df.to_csv('results/06_sensitivity_analysis/powerlaw_turnover_sensitivity_results.csv', index=False)
gamma_df.to_csv('results/06_sensitivity_analysis/gamma_turnover_sensitivity_results.csv', index=False)


