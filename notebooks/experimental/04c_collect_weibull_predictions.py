"""
04c_collect_weibull_predictions.py

Collect the Weibull/hockey-stick model F_new predictions for every Balesdent
(2018) site, mirroring notebooks/04_collect_continuum_model_predictions.py.

For each site the model is built from the calibrated (k, alpha) parameters and
the fraction of "new" carbon after the labeling duration is

    F_new = cdfA(Duration_labeling) = P(1/alpha, (k * Duration)^alpha)

i.e. the fraction of the steady-state carbon younger than the labeling period.
The 5%/95% columns propagate the turnover-uncertainty calibrations.

Run from the repository root:
    .venv/bin/python notebooks/experimental/04c_collect_weibull_predictions.py
"""
import sys
from os import path

import pandas as pd

sys.path.insert(0, path.join(path.dirname(__file__)))
from continuum_models_experimental import WeibullDisKin

# Load the site data
site_data = pd.read_csv('results/processed_balesdent_2018.csv')


def generate_predictions(model_class, params_df, param_names):
    """Generate F_new predictions for each site (see module docstring)."""
    result = site_data.copy()
    print(f"Generating {model_class.__name__} model predictions...")
    for i, row in params_df.iterrows():
        model_params = {name: row[name] for name in param_names}
        model = model_class(**model_params)
        if not model.params_valid():
            print(f"Invalid parameters for site {i}: {model_params}")
        result.loc[i, 'predicted_fnew'] = model.cdfA(
            site_data.loc[i, 'Duration_labeling'])
        if all(f"{name}_05" in row and f"{name}_95" in row
               for name in param_names):
            model_05_params = {name: row[f"{name}_05"] for name in param_names}
            model_95_params = {name: row[f"{name}_95"] for name in param_names}
            model_05 = model_class(**model_05_params)
            model_95 = model_class(**model_95_params)
            result.loc[i, 'predicted_fnew_05'] = model_05.cdfA(
                site_data.loc[i, 'Duration_labeling'])
            result.loc[i, 'predicted_fnew_95'] = model_95.cdfA(
                site_data.loc[i, 'Duration_labeling'])
    return result


#%% Weibull / hockey-stick model
fname = 'weibull_model_optimization_results.csv'
weibull_params = pd.read_csv(f'results/03_calibrate_models/{fname}')

result = generate_predictions(WeibullDisKin, weibull_params, ['k', 'alpha'])

output_path = 'results/04_model_predictions/weibull_model_predictions.csv'
result.to_csv(output_path, index=False)
print(f'wrote {output_path}')
