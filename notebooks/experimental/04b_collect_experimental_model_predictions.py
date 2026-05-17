#%%
import pandas as pd

from soil_diskin.continuum_models_experimental import GaussianDisKin, LogUniformRateDisKin

"""
Collects predictions for the experimental rate-distribution continuum models
(Gaussian rate, log-uniform rate) for all sites and saves them to CSV files.
"""

# Load the site data
site_data = pd.read_csv('results/processed_balesdent_2018.csv')


def generate_predictions(model_class, params_df, param_names):
    """
    Generate model predictions for each site based on the provided model class and parameters.

    Tolerates rows with missing or invalid parameters: skips them rather than
    raising, which is important for the experimental models where optimization
    can fail for some sites.

    Parameters
    ----------
    model_class : class
        The continuum model class to use for predictions.
    params_df : pd.DataFrame
        DataFrame containing the model parameters for each site.
    param_names : list
        List of parameter names corresponding to the model_class.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the model predictions for each site.
    """

    result = site_data.copy()
    print(f"Generating {model_class.__name__} model predictions...")
    for i, row in params_df.iterrows():
        model_params = {name: row[name] for name in param_names}
        if not all(pd.notna(value) for value in model_params.values()):
            print(f"Missing parameters for site {i}: {model_params}")
            continue
        try:
            model = model_class(**model_params)
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError) as exc:
            print(f"Could not instantiate {model_class.__name__} for site {i}: {model_params} ({exc})")
            continue
        if not model.params_valid():
            print(f"Invalid parameters for site {i}: {model_params}")
            continue
        result.loc[i, 'predicted_fnew'] = model.cdfA(site_data.loc[i, 'Duration_labeling'])
        has_uncertainty_params = all(
            f"{name}_05" in row and f"{name}_95" in row
            for name in param_names
        )
        if has_uncertainty_params:
            model_05_params = {name: row[f"{name}_05"] for name in param_names}
            model_95_params = {name: row[f"{name}_95"] for name in param_names}
            if (
                not all(pd.notna(value) for value in model_05_params.values())
                or not all(pd.notna(value) for value in model_95_params.values())
            ):
                continue
            model_05 = model_class(**model_05_params)
            model_95 = model_class(**model_95_params)
            if model_05.params_valid():
                result.loc[i, 'predicted_fnew_05'] = model_05.cdfA(site_data.loc[i, 'Duration_labeling'])
            if model_95.params_valid():
                result.loc[i, 'predicted_fnew_95'] = model_95.cdfA(site_data.loc[i, 'Duration_labeling'])
    return result


#%% Gaussian rate model

# load the Gaussian rate parameters
fname = 'gaussian_rate_model_optimization_results.csv'
gaussian_rate_params = pd.read_csv(f'results/03_calibrate_models/{fname}')

print("Generating Gaussian rate model predictions...")
result = generate_predictions(GaussianDisKin, gaussian_rate_params, ['mu', 'sigma'])
result.to_csv('results/04_model_predictions/gaussian_rate_model_predictions.csv', index=False)

#%% Log-uniform rate model

# load the log-uniform rate parameters
fname = 'loguniform_rate_model_optimization_results.csv'
loguniform_rate_params = pd.read_csv(f'results/03_calibrate_models/{fname}')

print("Generating log-uniform rate model predictions...")
result = generate_predictions(LogUniformRateDisKin, loguniform_rate_params, ['k_min', 'k_max'])
result.to_csv('results/04_model_predictions/loguniform_rate_model_predictions.csv', index=False)
