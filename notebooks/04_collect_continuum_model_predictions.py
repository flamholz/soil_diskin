#%%
from pathlib import Path

import numpy as np
import pandas as pd

from soil_diskin.continuum_models import PowerLawDisKin, GeneralPowerLawDisKin, WeibullDisKin, LognormalDisKin

"""
Collects the continuum model predictions for all sites and saves them to CSV files.
"""

def generate_predictions(model_class, params_df, param_names, site_data=None):
    """Predict f_new for bulk sites or layers at their labeling durations.

    Parameters
    ----------
    model_class : class
        Continuum model providing params_valid() and cdfA(duration).
    params_df : pd.DataFrame
        Calibrated parameters, aligned with site_data by row index. Optional
        columns ending in _05 or _95 contain stock uncertainty scenarios.
    param_names : list of str
        Model constructor arguments, e.g. ['mu', 'sigma'] for the lognormal model.
    site_data : pd.DataFrame, optional
        Observations and metadata, including Duration_labeling in years.
        Defaults to the prepared bulk table, loaded only when needed.

    Returns
    -------
    pd.DataFrame
        Copy of site_data with predicted_fnew and any available _05/_95 scenario
        predictions. Missing or invalid parameters and durations leave NaN;
        rows are retained. Scenarios are evaluated independently and need not
        give ordered lower and upper bounds on f_new.
    """
    if site_data is None:
        site_data = pd.read_csv('results/processed_balesdent_2018.csv')
    result = site_data.copy()
    durations = site_data['Duration_labeling']
    print(f"Generating {model_class.__name__} model predictions...")
    for suffix in ['', '_05', '_95']:
        columns = [name+suffix for name in param_names]
        if not all(column in params_df for column in columns):
            continue
        result['predicted_fnew'+suffix] = np.nan
        for i, row in params_df[columns].iterrows():
            if not np.isfinite(row).all() or not np.isfinite(durations.loc[i]) or durations.loc[i] < 0:
                continue
            model = model_class(**dict(zip(param_names, row)))
            if not model.params_valid():
                print(f"Invalid parameters for site {i}: {row.to_dict()}")
                continue
            result.loc[i, 'predicted_fnew'+suffix] = model.cdfA(durations.loc[i])
    return result

def main():
    site_data = pd.read_csv('results/processed_balesdent_2018.csv')
    #%% Power-law model

    # load the power-law parameters
    fname = 'powerlaw_model_optimization_results.csv'
    power_law_params = pd.read_csv(f'results/03_calibrate_models/{fname}')
    print("Generating power-law model predictions...")
    result = generate_predictions(PowerLawDisKin, power_law_params, ['t_min', 't_max'])
    result.to_csv('results/04_model_predictions/power_law_model_predictions.csv', index=False)

    #%% Generalized Power-law model with beta = np.exp(-GAMMA)

    # load the power-law parameters
    fname = 'general_powerlaw_model_optimization_results.csv'
    general_power_law_params = pd.read_csv(f'results/03_calibrate_models/{fname}')

    print("Generating generalized power-law model predictions...")
    result = generate_predictions(GeneralPowerLawDisKin, general_power_law_params, ['t_min', 't_max', 'beta'])
    result.to_csv('results/04_model_predictions/general_power_law_model_predictions.csv', index=False)

    #%% Generalized Power-law model with beta = np.exp(-GAMMA) / 2

    # load the power-law parameters
    fname = 'general_powerlaw_model_optimization_results_beta_half.csv'
    general_power_law_params = pd.read_csv(f'results/03_calibrate_models/{fname}')

    print("Generating generalized power-law model predictions...")
    result = generate_predictions(GeneralPowerLawDisKin, general_power_law_params, ['t_min', 't_max', 'beta'])
    result.to_csv('results/04_model_predictions/general_power_law_model_predictions_beta_half.csv', index=False)


    #%% Weibull / hockey-stick model
    fname = 'weibull_model_optimization_results.csv'
    weibull_params = pd.read_csv(f'results/03_calibrate_models/{fname}')

    result = generate_predictions(WeibullDisKin, weibull_params, ['k', 'alpha'])

    output_path = 'results/04_model_predictions/weibull_model_predictions.csv'
    result.to_csv(output_path, index=False)
    print(f'wrote {output_path}')


    #%% Lognormal model

    # Bulk and layers use the fitted mu/sigma from 03b and the same prediction function.
    for folder in ['', 'depth_resolved']:
        source = Path('results/03_calibrate_models')/folder/'03b_lognormal_predictions_calcurve_python.csv'
        if folder and not source.exists():
            continue  # Layer calibration is optional in the bulk pipeline.
        params = pd.read_csv(source, dtype={'profile_id': str})
        result = generate_predictions(LognormalDisKin, params, ['mu', 'sigma'], params if folder else site_data)
        output = Path('results/04_model_predictions')/folder/'lognormal_model_predictions.csv'
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)


if __name__ == '__main__':
    main()
