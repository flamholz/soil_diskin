"""Calibrate a log-uniform decay-rate model to turnover and radiocarbon."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from soil_diskin.continuum_models import LogUniformDisKin


def model_params_from_turnover(turnover, log_width):
    """Return model parameters with the requested turnover and log-rate width."""
    k_min = (1 - np.exp(-log_width)) / (turnover * log_width)
    return k_min, log_width


def objective_function(log_width, site):
    """Squared relative radiocarbon mismatch for one site."""
    model = LogUniformDisKin(*model_params_from_turnover(site['turnover'], log_width))
    modeled_14c = model.calc_radiocarbon_ratio_ss()[0]
    return ((modeled_14c - site['fm']) / (site['fm'] + 1e-6)) ** 2


def fit_site(site):
    """Fit log-rate width while matching observed turnover analytically."""
    fit = minimize_scalar(
        objective_function,
        args=(site,),
        method='bounded',
        bounds=(1e-3, 5000),
        options={'xatol': 1e-6},
    )
    k_min, log_width = model_params_from_turnover(site['turnover'], fit.x)
    model = LogUniformDisKin(k_min, log_width)
    return {
        'objective_value': fit.fun,
        'k_min': k_min,
        'log_width': log_width,
        'log_k_max': model.log_k_max,
        'modeled_tau': model.T,
        'modeled_age': model.A,
        'modeled_14C': model.calc_radiocarbon_ratio_ss()[0],
        'params_valid': model.params_valid(),
    }


def fit_sites(site_data):
    return pd.DataFrame(
        [fit_site(row) for _, row in tqdm(site_data.iterrows(), total=len(site_data))],
        index=site_data.index,
    )


def main():
    site_data = pd.read_csv('results/all_sites_14C_turnover.csv')
    result = fit_sites(site_data)

    backfilled = site_data[
        site_data['turnover_q05'].notna() & site_data['turnover_q95'].notna()
    ]
    for suffix in ('05', '95'):
        bound_data = backfilled.copy()
        bound_data['turnover'] = bound_data[f'turnover_q{suffix}']
        result = result.join(fit_sites(bound_data).add_suffix(f'_{suffix}'))

    result = result.join(site_data[['fm', 'turnover']])
    output = Path('results/03_calibrate_models/loguniform_model_optimization_results.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f'wrote {output}; maximum objective value: {result.objective_value.max():.3e}')


if __name__ == '__main__':
    main()
