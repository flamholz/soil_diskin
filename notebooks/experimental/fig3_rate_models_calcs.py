#%%
import numpy as np
import pandas as pd

from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error

"""
Bootstrap performance summaries for the rate-distribution continuum models.
"""

#%% Load data and predictions
all_sites = pd.read_csv('results/processed_balesdent_2018.csv')

prediction_files = {
    'Power-law': 'results/04_model_predictions/power_law_model_predictions.csv',
    'Lognormal': 'results/04_model_predictions/lognormal_model_predictions.csv',
    'Gamma': 'results/04_model_predictions/gamma_model_predictions.csv',
    'Gaussian rate': 'results/04_model_predictions/gaussian_rate_model_predictions.csv',
    'Log-uniform rate': 'results/04_model_predictions/loguniform_rate_model_predictions.csv',
}
prediction_dfs = {
    model_name: pd.read_csv(fname)
    for model_name, fname in prediction_files.items()
}

#%% Bootstrap RMSE and KGE
n_iterations = 10000
n_sites = int(0.9 * len(all_sites))
rng = np.random.default_rng(12345)
sample_indices = rng.choice(all_sites.index, size=(n_sites, n_iterations), replace=True)
true_values = all_sites['total_fnew'].values

results = {
    model_name: {'RMSE': [], 'KGE': []}
    for model_name in prediction_files
}


def analyze_model(model_name, predictions):
    """Analyze one model's bootstrap performance."""
    print(f'Analyzing model: {model_name}')
    predicted_values = predictions['predicted_fnew'].values
    for i in range(sample_indices.shape[1]):
        sample = sample_indices[:, i]
        y_true = true_values[sample]
        y_pred = predicted_values[sample]
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 2:
            results[model_name]['RMSE'].append(np.nan)
            results[model_name]['KGE'].append(np.nan)
            continue

        evaluator = RegressionMetric(y_true=y_true[mask], y_pred=y_pred[mask])
        results[model_name]['RMSE'].append(
            root_mean_squared_error(y_true[mask], y_pred[mask])
        )
        results[model_name]['KGE'].append(evaluator.kling_gupta_efficiency())


for model_name, predictions in prediction_dfs.items():
    analyze_model(model_name, predictions)

#%% Save long-form metrics
results_df = []
for model_name, metrics in results.items():
    for metric_name, values in metrics.items():
        for value in values:
            results_df.append({
                'model': model_name,
                'metric': metric_name,
                'value': value,
            })

results_df = pd.DataFrame(results_df)
results_df.to_csv('results/fig3_rate_models_calcs.csv', index=False)
# %%
