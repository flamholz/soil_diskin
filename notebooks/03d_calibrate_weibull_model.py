"""
03h_calibrate_weibull_model.py

Calibrate the Feng (2009) "hockey stick" survival model

    s(tau) = exp( -(k * tau)^alpha )

to each Balesdent (2018) site, exactly mirroring the power-law pipeline in
notebooks/03a_calibrate_powerlaw_model.py.

For every site we have two steady-state observables:
  * turnover  -- the mean transit time T = C_stock / input
  * fm        -- the steady-state radiocarbon ratio (fraction modern)

The two free parameters (k, alpha) are tuned so the model reproduces both.
The relevant steady-state relationships for the Weibull/hockey-stick form
(derived in weibull_hockey_stick_pipeline.md) are

    T   = Gamma(1 + 1/alpha) / k
    A   = Gamma(1 + 2/alpha) / (2 k Gamma(1 + 1/alpha))
    A/T = Gamma(1 + 2/alpha) / (2 Gamma(1 + 1/alpha)^2)   (depends on alpha only)

so alpha controls the *shape* of the age distribution (Feng's stabilization
coefficient beta = A/T) and k sets the overall timescale. The radiocarbon
ratio is computed by numerically integrating the steady-state age
distribution against the historical atmospheric 14C curve (the inherited
AbstractDiskinModel.radiocarbon_age_integrand).

Run from the repository root:
    .venv/bin/python notebooks/experimental/03h_calibrate_weibull_model.py
"""
import sys
from os import path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gamma
from tqdm import tqdm

sys.path.insert(0, path.join(path.dirname(__file__)))
from soil_diskin.continuum_models import WeibullDisKin


def objective_function(params, merged_site_data):
    """Squared sum of the relative turnover and 14C mismatches for one site.

    Parameters
    ----------
    params : list
        [k, alpha] for the WeibullDisKin model.
    merged_site_data : pd.Series
        Observed data for a site, including 'fm' (14C ratio) and 'turnover'.

    Returns
    -------
    float
        relative_diff_14C**2 + relative_diff_turnover**2
    """
    k, alpha = params

    model = WeibullDisKin(k, alpha)
    if not model.params_valid():
        return 1e6

    # predicted steady-state radiocarbon ratio
    predicted_14C_ratio = quad(model.radiocarbon_age_integrand,
                               0, np.inf, limit=1500, epsabs=1e-3)[0]

    diff_14C = np.nansum((predicted_14C_ratio - merged_site_data['fm']))
    total_14C = np.nansum(merged_site_data['fm'] + 1e-6)
    relative_diff_14C = diff_14C / total_14C

    diff_turnover = np.nansum((model.T - merged_site_data['turnover']))
    total_turnover = np.nansum(merged_site_data['turnover'] + 1e-6)
    relative_diff_turnover = diff_turnover / total_turnover

    return relative_diff_14C**2 + relative_diff_turnover**2


def parse_results(result, index):
    """Convert optimization results to a DataFrame and compute model metrics."""
    df = pd.DataFrame(result, columns=['params', 'objective_value'], index=index)

    df[['k', 'alpha']] = df['params'].apply(
        lambda x: pd.Series(x, index=['k', 'alpha']))
    df.drop(columns='params', inplace=True)

    model_x = lambda x: WeibullDisKin(x['k'], x['alpha'])
    df['modeled_tau'] = df.apply(lambda x: model_x(x).T, axis=1)
    df['modeled_age'] = df.apply(lambda x: model_x(x).A, axis=1)
    df['modeled_14C'] = df.apply(
        lambda x: quad(model_x(x).radiocarbon_age_integrand,
                       0, np.inf, limit=1500, epsabs=1e-3)[0], axis=1)
    df['params_valid'] = df.apply(lambda x: model_x(x).params_valid(), axis=1)

    return df


def scan_initial_guess(row, alpha_grid):
    """Coarse 1-D search for a good starting (k, alpha).

    For each candidate alpha we pin k = Gamma(1 + 1/alpha) / turnover so the
    turnover is matched exactly, then keep the alpha whose steady-state 14C
    ratio is closest to the observation. This sidesteps the noisy
    finite-difference gradients that strand a plain L-BFGS-B start.
    """
    turnover = row['turnover']
    best = None
    for alpha in alpha_grid:
        k = gamma(1.0 + 1.0 / alpha) / turnover
        obj = objective_function([k, alpha], row)
        if best is None or obj < best[0]:
            best = (obj, [k, alpha])
    return best[1]


def fit_site(row, initial_guess, bounds, alpha_grid):
    """Fit one site: scan for a starting point, then polish in 2-D.

    We refine from both the per-site scan start and the shared default
    guess and keep whichever lands lower, so a single bad start cannot
    strand a site at its initial parameters.
    """
    candidates = []

    scan_guess = scan_initial_guess(row, alpha_grid)
    for guess in (scan_guess, initial_guess):
        res = minimize(objective_function, guess, args=(row,),
                       method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-12})
        candidates.append((res.fun, res.x))

    best = min(candidates, key=lambda c: c[0])
    return [best[1], best[0]]


#%% Load the data
merged_site_data = pd.read_csv('results/all_sites_14C_turnover.csv')

# initial guess [k (yr^-1), alpha (-)] and bounds keeping both positive
initial_guess = [0.1, 0.4]
bounds = [(1e-6, None), (1e-3, 5.0)]
# alpha grid for the per-site scan-based initialization
alpha_grid = np.linspace(0.08, 1.0, 47)

#%% Perform optimization for each site
result = []
for i, row in tqdm(merged_site_data.iterrows(), total=len(merged_site_data)):
    result.append(fit_site(row, initial_guess, bounds, alpha_grid))

result_df = parse_results(result, merged_site_data.index)

#%% Optimize the 5% and 95% turnover uncertainty bounds
results_05 = []
results_95 = []
backfilled_sites = merged_site_data[
    merged_site_data['turnover_q05'].notna()
    & merged_site_data['turnover_q95'].notna()]
for i, row in tqdm(backfilled_sites.iterrows(), total=len(backfilled_sites)):
    row_05 = row.copy()
    row_05['turnover'] = row.loc['turnover_q05']
    row_95 = row.copy()
    row_95['turnover'] = row.loc['turnover_q95']
    results_05.append(fit_site(row_05, initial_guess, bounds, alpha_grid))
    results_95.append(fit_site(row_95, initial_guess, bounds, alpha_grid))

result_df_05 = parse_results(results_05, backfilled_sites.index)
result_df_95 = parse_results(results_95, backfilled_sites.index)

merged_result_df = pd.concat(
    [result_df, merged_site_data[['fm', 'turnover']]], axis=1)
merged_result_df = pd.merge(merged_result_df, result_df_05.add_suffix('_05'),
                            left_index=True, right_index=True, how='left')
merged_result_df = pd.merge(merged_result_df, result_df_95.add_suffix('_95'),
                            left_index=True, right_index=True, how='left')

print(f'the Maximum objective value is {result_df["objective_value"].max():.3e}')

#%% Save the result to a CSV file
output_dir = 'results/03_calibrate_models'
output_fname = 'weibull_model_optimization_results.csv'
output_path = path.join(output_dir, output_fname)
merged_result_df.to_csv(output_path, index=False)
print(f'wrote {output_path}')
