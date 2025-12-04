#%% 
import numpy as np
import pandas as pd

from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm

#%% Load the site data
all_sites = pd.read_csv('results/processed_balesdent_2018.csv')

#%% Load the predictions
print('Loading model predictions...')
powerlaw_predictions = pd.read_csv(f'results/04_model_predictions/power_law.csv',header=None, names=['prediction'])
gen_powerlaw_preds_beta = pd.read_csv(f'results/04_model_predictions/general_power_law.csv', header=None, names=['prediction'])
gen_powerlaw_preds_beta_half = pd.read_csv(f'results/04_model_predictions/general_power_law_beta_half.csv', header=None, names=['prediction'])
lognormal_predictions = pd.read_csv(f'results/04_model_predictions/lognormal.csv',header=None, names=['prediction'])
gamma_predictions = pd.read_csv(f'results/04_model_predictions/gamma.csv',header=None, names=['prediction'])    
CLM45_predictions = pd.read_csv(f'results/04_model_predictions/CLM45_fnew.csv', header=None, names=['prediction'])
JSBACH_predictions = pd.read_csv(f'results/04_model_predictions/JSBACH_fnew.csv', header=None, names=['prediction'])
RCM_predictions = pd.read_csv(f'results/04_model_predictions/RCM.csv')

# Subsample 90% of the sites 10^4 times to estimate the RMSE and KGE distributions.
n_iterations = 10000
n_sites = int(0.9 * len(all_sites))
results = { 'Power-law': {'RMSE': [], 'KGE': []},
            'Gen. Power-law (b=exp(-gamma))': {'RMSE': [], 'KGE': []},
            'Gen. Power-law (b=exp(-gamma)/2)': {'RMSE': [], 'KGE': []},
            'Lognormal': {'RMSE': [], 'KGE': []},
            'Gamma': {'RMSE': [], 'KGE': []},
            'CLM4.5': {'RMSE': [], 'KGE': []},
            'JSBACH': {'RMSE': [], 'KGE': []},
          }
# Add entries for the reduced complexity models
for i, col in enumerate(RCM_predictions.columns):
    results[col] = {'RMSE': [], 'KGE': []}

# Run the bootstrap resampling
print('Running bootstrap resampling to estimate RMSE and KGE distributions...')
for i in tqdm(range(n_iterations)):
    sample_indices = np.random.choice(all_sites.index, size=n_sites, replace=True)
    true_values = all_sites.loc[sample_indices, 'total_fnew'].values

    # Power-law
    preds = powerlaw_predictions.loc[sample_indices, 'prediction'].values
    evaluator = RegressionMetric(y_true=true_values, y_pred=preds)
    results['Power-law']['RMSE'].append(root_mean_squared_error(true_values, preds))
    results['Power-law']['KGE'].append(evaluator.kling_gupta_efficiency())

    # Generalized Power-law (b=exp(-gamma))
    preds = gen_powerlaw_preds_beta.loc[sample_indices, 'prediction'].values
    evaluator = RegressionMetric(y_true=true_values, y_pred=preds)
    results['Gen. Power-law (b=exp(-gamma))']['RMSE'].append(root_mean_squared_error(true_values, preds))
    results['Gen. Power-law (b=exp(-gamma))']['KGE'].append(evaluator.kling_gupta_efficiency())

    # Generalized Power-law (b=exp(-gamma)/2)
    preds = gen_powerlaw_preds_beta_half.loc[sample_indices, 'prediction'].values
    evaluator = RegressionMetric(y_true=true_values, y_pred=preds)
    results['Gen. Power-law (b=exp(-gamma)/2)']['RMSE'].append(root_mean_squared_error(true_values, preds))
    results['Gen. Power-law (b=exp(-gamma)/2)']['KGE'].append(evaluator.kling_gupta_efficiency())

    # Lognormal
    preds = lognormal_predictions.loc[sample_indices, 'prediction'].values
    evaluator = RegressionMetric(y_true=true_values, y_pred=preds)
    results['Lognormal']['RMSE'].append(root_mean_squared_error(true_values, preds))
    results['Lognormal']['KGE'].append(evaluator.kling_gupta_efficiency())

    # Gamma
    preds = gamma_predictions.loc[sample_indices, 'prediction'].values
    evaluator = RegressionMetric(y_true=true_values, y_pred=preds)
    results['Gamma']['RMSE'].append(root_mean_squared_error(true_values, preds))
    results['Gamma']['KGE'].append(evaluator.kling_gupta_efficiency())

    # CLM4.5
    preds = CLM45_predictions.loc[sample_indices, 'prediction'].values
    evaluator = RegressionMetric(y_true=true_values, y_pred=preds)
    results['CLM4.5']['RMSE'].append(root_mean_squared_error(true_values, preds))
    results['CLM4.5']['KGE'].append(evaluator.kling_gupta_efficiency())

    # JSBACH
    preds = JSBACH_predictions.loc[sample_indices, 'prediction'].values
    evaluator = RegressionMetric(y_true=true_values, y_pred=preds)
    results['JSBACH']['RMSE'].append(root_mean_squared_error(true_values, preds))
    results['JSBACH']['KGE'].append(evaluator.kling_gupta_efficiency())

    # RCM
    for i, col in enumerate(RCM_predictions.columns):
        preds = RCM_predictions.loc[sample_indices, col].values
        evaluator = RegressionMetric(y_true=true_values, y_pred=preds)
        results[col]['RMSE'].append(root_mean_squared_error(true_values, preds))
        results[col]['KGE'].append(evaluator.kling_gupta_efficiency())

# Make into a dataframe and save to CSV for later plotting. 
print('Saving results to a long-form CSV...')
results_df = []
for model_name, metrics in results.items():
    for metric_name, values in metrics.items():
        for value in values:
            results_df.append({'model': model_name, 'metric': metric_name, 'value': value})

results_df = pd.DataFrame(results_df)
results_df.to_csv('results/fig3_calcs.csv', index=False)
# %%
