"""Steady-state sensitivity analysis for the lognormal model.

Tests how the fraction of new carbon responds to a vegetation transition that
changes the steady state of the system, either through the carbon input rate or
through the parameters of the rate distribution. The size of the change is taken
from the percentiles of the ratio between the reference and the total C stock of
the sites of Balesdent et al. 2018.

Python port of `06b_lognormal_steady_state_sensitivity.jl`, which is now in
`notebooks/archive`. The Julia version solved the model by quadrature over an ODE
solution; here the same model is evaluated with the closed-form C(t) of
`soil_diskin.lognormal`, which reproduces the Julia output to ~1e-6 in F_new.

Outputs, each a time series per percentile of the change plus a `time` column:
  results/06_sensitivity_analysis/lognormal_input_data.csv   (change in inputs)
  results/06_sensitivity_analysis/lognormal_mu_data.csv      (change in mu)
  results/06_sensitivity_analysis/lognormal_sigma_data.csv   (change in sigma)
"""
# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

# %%
import numpy as np
import pandas as pd

from soil_diskin.lognormal import run_diskin_fast

TMAX = 100_000
TS_SIZE = 1000

# %%
# The change in the steady state is taken from the ratio between the C stock of the
# reference site and the C stock of the site that underwent the vegetation transition
raw_site_data = pd.read_excel('data/balesdent_2018/balesdent_2018_raw.xlsx',
                              sheet_name='Profiles', skiprows=7)
stock_ratios = (pd.to_numeric(raw_site_data['Cref_0-100estim'], errors='coerce')
                / pd.to_numeric(raw_site_data['Ctotal_0-100estim'], errors='coerce'))
J_ratio = np.percentile(stock_ratios.dropna(), [2.5, 25, 50, 75, 97.5])

# The unperturbed system is the mean turnover time and mean age across all sites
site_params = pd.read_csv('results/03_calibrate_models/03b_lognormal_predictions_calcurve_python.csv')
mean_age = site_params['pred'].mean()
mean_turnover = site_params['turnover'].mean()


# %%
def f_new(J1, J2, tau1, tau2, age1, age2):
    """Fraction of new carbon after a transition between two steady states.

    Parameters
    ----------
    J1, J2 : float
        Carbon input rate before and after the transition.
    tau1, tau2 : float
        Turnover time before and after the transition, in years.
    age1, age2 : float
        Mean age of the carbon before and after the transition, in years.

    Returns
    -------
    tuple of np.ndarray
        The time grid and the fraction of new carbon at each time point.
    """
    ts, labeled = run_diskin_fast(tau2, age2, 1.0, tmax=TMAX, ts_size=TS_SIZE)
    # The carbon remaining from before the transition is the steady state stock
    # of the old system minus the carbon that has been replaced since.
    unlabeled = tau1 - run_diskin_fast(tau1, age1, 1.0, tmax=TMAX, ts_size=TS_SIZE)[1]
    return ts, J2 * labeled / (J2 * labeled + J1 * unlabeled)


def f_new_mu_sigma(J1, J2, mu1, mu2, sigma1, sigma2):
    """`f_new` for systems defined by the parameters of the rate distribution."""
    tau1 = np.exp(-mu1 + 0.5 * sigma1**2)
    tau2 = np.exp(-mu2 + 0.5 * sigma2**2)
    return f_new(J1, J2, tau1, tau2, tau1 * np.exp(sigma1**2), tau2 * np.exp(sigma2**2))


# %%
# For each percentile of the change, perturb either the inputs, mu or sigma
input_data, mu_data, sigma_data = {}, {}, {}
old_mu = -np.log(np.sqrt(mean_turnover**3 / mean_age))
old_sigma = np.sqrt(np.log(mean_age / mean_turnover))

for ratio in J_ratio:
    new_mu = -np.log(ratio) + old_mu
    new_sigma = np.sqrt(2 * np.log(ratio) + old_sigma**2)

    col = f'{round(ratio, 2):g}'
    ts, input_data[col] = f_new(1, ratio, mean_turnover, mean_turnover, mean_age, mean_age)
    _, mu_data[col] = f_new_mu_sigma(1, 1, old_mu, new_mu, old_sigma, old_sigma)
    _, sigma_data[col] = f_new_mu_sigma(1, 1, old_mu, old_mu, old_sigma, new_sigma)

# %%
# Save the results to CSV files
for data, fname in [(input_data, 'lognormal_input_data.csv'),
                    (mu_data, 'lognormal_mu_data.csv'),
                    (sigma_data, 'lognormal_sigma_data.csv')]:
    df = pd.DataFrame(data)
    df['time'] = ts
    df.to_csv(f'results/06_sensitivity_analysis/{fname}', index=False)
