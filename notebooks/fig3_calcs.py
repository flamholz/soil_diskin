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
powerlaw_predictions = pd.read_csv(f'results/04_model_predictions/power_law_model_predictions.csv')
gen_powerlaw_preds_beta = pd.read_csv(f'results/04_model_predictions/general_power_law_model_predictions.csv')
gen_powerlaw_preds_beta_half = pd.read_csv(f'results/04_model_predictions/general_power_law_model_predictions_beta_half.csv')
lognormal_predictions = pd.read_csv(f'results/04_model_predictions/lognormal_model_predictions.csv')
gamma_predictions = pd.read_csv(f'results/04_model_predictions/gamma_model_predictions.csv')    
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
# Sample n_site random sites for n_iterations
sample_indices = np.random.choice(all_sites.index, size=(n_sites, n_iterations), replace=True)
true_values = all_sites['total_fnew'].values[sample_indices]


def analyze_model(model_type, df, samples, true_vals):
    """Helper function to analyze a model's predictions."""
    print(f'Analyzing model: {model_type}')
    for i in range(samples.shape[1]):
        if model_type in ['CLM4.5', 'JSBACH']:
            preds = df['prediction'].values[samples[:, i]]
        elif model_type in RCM_predictions.columns:
            preds = df.values[samples[:, i]]
        else:
            preds = df['predicted_fnew'].values[samples[:, i]]
        evaluator = RegressionMetric(y_true=true_vals[:, i], y_pred=preds)
        results[model_type]['RMSE'].append(root_mean_squared_error(true_vals[:, i], preds))
        results[model_type]['KGE'].append(evaluator.kling_gupta_efficiency())

# Run the bootstrap resampling
print('Running bootstrap resampling to estimate RMSE and KGE distributions...')
model_types = ['Power-law',
                   'Gen. Power-law (b=exp(-gamma))',
                   'Gen. Power-law (b=exp(-gamma)/2)',
                   'Lognormal',
                   'Gamma',
                   'CLM4.5',
                   'JSBACH'] + list(RCM_predictions.columns)
prediction_dfs = [powerlaw_predictions,
                      gen_powerlaw_preds_beta,
                      gen_powerlaw_preds_beta_half,
                      lognormal_predictions,
                      gamma_predictions,
                      CLM45_predictions,
                      JSBACH_predictions] + [RCM_predictions[col] for col in RCM_predictions.columns]
for mt, df in zip(model_types, prediction_dfs):
    analyze_model(mt, df, sample_indices, true_values)

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
