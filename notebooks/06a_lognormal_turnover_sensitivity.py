"""Turnover time sensitivity analysis for the lognormal model.

Scales the turnover time of every site by a set of ratios and repeats the
calibration and prediction of the lognormal model with the scaled turnover time,
which shows how sensitive the predicted F_new is to errors in the turnover time.

This is a full Python port of the pipeline that used to be split between
`06a_lognormal_turnover_sensitivity.wls` (age scans), this script (inversion) and
`06a_lognormal_turnover_sensitivity.jl` (CDFs); both are now in `notebooks/archive`.
Note that the Julia script solved the CDF with the *unscaled* turnover time, so
the scaling was only applied to the calibration; here the scaled turnover time is
used throughout, as it is for the power-law model in `06a_turnover_sensitivity.py`.

For each site and each ratio:
  1. Scan the predicted fm over the age grid at the scaled turnover time.
  2. Invert the scan and look up the site's measured fm to get the mean age.
  3. Evaluate the closed-form Diskin C(t) at the scaled turnover time and
     interpolate F_new at the site's labeling duration.

The same is done for the SoilGrids-backfilled sites using their q05/q95 turnover
estimates, which gives the error bars of those sites in figS4.

Outputs:
  results/06_sensitivity_analysis/lognormal_age_predictions.csv
  results/06_sensitivity_analysis/06a_lognormal_cdfs_{ratio}.csv
  results/06_sensitivity_analysis/06a_lognormal_fnew.csv
  results/06_sensitivity_analysis/06a_lognormal_fnew_q05.csv
  results/06_sensitivity_analysis/06a_lognormal_fnew_q95.csv
"""
# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

# %%
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

from soil_diskin.lognormal import run_diskin_fast, scan_ages
from soil_diskin.radiocarbon_utils import load_atm14c

# Age grid of the calibration curve: 101 log-spaced ages from 10^3 to 10^5.5.
AGELIST = np.logspace(3.0, 5.5, 101)
# Time grid of the CDF solution.
TMAX = 100_000
TS_SIZE = 1000
TS = np.logspace(-1.0, np.log10(TMAX), TS_SIZE)

# Ratios by which the turnover time of each site is scaled, and the labels used
# in the names of the per-ratio output files.
ratios = [0.5, 1/1.5, 1, 1.5, 2]
ratio_labels = ['0.50', '0.67', '1', '1.50', '2']

# %%
atm = load_atm14c('data/14C_atm_annot.csv')
site_data = pd.read_csv('results/processed_balesdent_2018.csv')
turnover_14C = pd.read_csv('results/all_sites_14C_turnover.csv')
backfilled_sites = turnover_14C[turnover_14C['turnover_q05'].notna()]


# %%
def predict_age(turnover, fm):
    """Calibrate the mean age of a site against its radiocarbon measurement.

    Parameters
    ----------
    turnover : float
        Turnover time of the site, in years.
    fm : float
        Measured radiocarbon ratio of the site.

    Returns
    -------
    float
        Mean age of the carbon in the soil, in years.
    """
    age_scan = scan_ages(atm, turnover, AGELIST)
    return float(interp1d(age_scan, AGELIST, fill_value='extrapolate')(fm))


def predict_cdf(turnover, age):
    """Evaluate the closed-form Diskin C(t) on the `TS` grid for a unit input."""
    return run_diskin_fast(turnover, age, 1.0, tmax=TMAX, ts_size=TS_SIZE)[1]


def predict_fnew(turnover, fm, duration):
    """Predict F_new of a site whose turnover time is `turnover`.

    Parameters
    ----------
    turnover : float
        Turnover time of the site, in years, after scaling by the ratio.
    fm : float
        Measured radiocarbon ratio of the site.
    duration : float
        Time since the vegetation transition, in years.

    Returns
    -------
    float
        Predicted fraction of new carbon.
    """
    cdf = predict_cdf(turnover, predict_age(turnover, fm))
    return float(interp1d(TS, cdf / turnover)(duration))


# %%
# Calibrate and predict for all sites at each ratio. The CDFs are written out per
# ratio in the same layout the Julia script used, because 06c reads the ratio=1 file.
ages = {}
fnew = {}
for ratio, label in zip(ratios, ratio_labels):
    print(f'Scanning ratio {label} for {len(turnover_14C)} sites...')
    turnovers = turnover_14C['turnover'] * ratio

    ages[label] = Parallel(n_jobs=-1, verbose=1)(
        delayed(predict_age)(turnover, fm)
        for turnover, fm in zip(turnovers, turnover_14C['fm']))

    cdfs = Parallel(n_jobs=-1, verbose=1)(
        delayed(predict_cdf)(turnover, age)
        for turnover, age in zip(turnovers, ages[label]))

    cdfs = pd.DataFrame(np.vstack(cdfs), columns=TS)
    cdfs.to_csv(f'results/06_sensitivity_analysis/06a_lognormal_cdfs_{label}.csv',
                index=False)

    fnew[ratio] = [float(interp1d(TS, cdf / turnover)(duration))
                   for cdf, turnover, duration
                   in zip(cdfs.values, turnovers, site_data['Duration_labeling'])]

pd.DataFrame(ages).to_csv('results/06_sensitivity_analysis/lognormal_age_predictions.csv',
                          index=False)
pd.DataFrame(fnew).to_csv('results/06_sensitivity_analysis/06a_lognormal_fnew.csv',
                          index=False)

# %%
# Repeat for the q05/q95 turnover estimates of the SoilGrids-backfilled sites.
# Rows of sites without a backfilled estimate are NaN.
for quantile in ['q05', 'q95']:
    print(f'Scanning {quantile} for {len(backfilled_sites)} backfilled sites...')
    predictions = Parallel(n_jobs=-1, verbose=1)(
        delayed(predict_fnew)(row[f'turnover_{quantile}'] * ratio, row['fm'],
                              site_data.loc[i, 'Duration_labeling'])
        for ratio in ratios for i, row in backfilled_sites.iterrows()
    )

    result = pd.DataFrame(np.array(predictions).reshape(len(ratios), -1).T,
                          index=backfilled_sites.index, columns=ratios)
    result = result.reindex(site_data.index)
    result.to_csv(f'results/06_sensitivity_analysis/06a_lognormal_fnew_{quantile}.csv',
                  index=False)
