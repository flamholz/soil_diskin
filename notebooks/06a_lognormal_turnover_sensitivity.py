"""Turnover time sensitivity analysis for the lognormal model.

Scales the turnover time of every site by a set of ratios and repeats the
calibration and prediction of the lognormal model with the scaled turnover time,
which shows how sensitive the predicted F_new is to errors in the turnover time.

This replaces a pipeline that used to be split between
`06a_lognormal_turnover_sensitivity.wls` (age scans), an interpolation-based
inversion and `06a_lognormal_turnover_sensitivity.jl` (CDFs); both are now in
`notebooks/archive`. Note that the Julia script solved the CDF with the
*unscaled* turnover time, so the scaling was only applied to the calibration;
here the scaled turnover time is used throughout, as it is for the power-law
model in `06a_turnover_sensitivity.py`.

For each ratio:
  1. Calibrate mu and sigma to each site's scaled turnover time and measured fm
     with the shared calibration (calibration.lognormal_spec), as script 03b
     does. At ratio 1 the fits are those of 03b.
  2. Evaluate each fitted model's age CDF on a time grid (written out), and at
     the site's labeling duration to get F_new.

The SoilGrids-backfilled sites are also fit with their scaled q05/q95 turnover
estimates, which gives their error bars in figS4. Sites whose fit fails get NaN.

Outputs (in --output-dir):
  lognormal_age_predictions.csv
  06a_lognormal_cdfs_{ratio}.csv
  06a_lognormal_fnew.csv
  06a_lognormal_fnew_q05.csv
  06a_lognormal_fnew_q95.csv

Run from the project root directory:
    python notebooks/06a_lognormal_turnover_sensitivity.py [--n-jobs -1]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from soil_diskin.calibration import calibrate, lognormal_spec
from soil_diskin.continuum_models import LognormalDisKin
from soil_diskin.radiocarbon_utils import load_atm14c

TURNOVER_PATH = 'results/all_sites_14C_turnover.csv'
SITES_PATH = 'results/processed_balesdent_2018.csv'
ATM_PATH = 'data/14C_atm_annot.csv'
OUTPUT_DIR = 'results/06_sensitivity_analysis'

# Time grid of the CDF solution.
TMAX = 100_000
TS_SIZE = 1000
TS = np.logspace(-1.0, np.log10(TMAX), TS_SIZE)
SCENARIOS = {'': 'turnover', '_05': 'turnover_q05', '_95': 'turnover_q95'}

# Ratios by which the turnover time of each site is scaled, and the labels used
# in the names of the per-ratio output files.
RATIOS = [0.5, 1/1.5, 1, 1.5, 2]
RATIO_LABELS = ['0.50', '0.67', '1', '1.50', '2']


def stock_younger_than_grid(mu, sigma):
    """Steady-state carbon younger than each time in `TS`, per unit input."""
    if not np.isfinite([mu, sigma]).all():
        return np.full(TS_SIZE, np.nan)
    return LognormalDisKin(mu, sigma).stock_younger_than(TS)


def predict_fnew(fits, suffix, durations):
    """F_new at each labeling duration from the site's fitted model (NaN without one)."""
    predictions = []
    for i, duration in durations.items():
        mu, sigma = fits.loc[i, 'mu'+suffix], fits.loc[i, 'sigma'+suffix]
        fitted = np.isfinite([mu, sigma]).all()
        predictions.append(LognormalDisKin(mu, sigma).cdfA(duration) if fitted else np.nan)
    return predictions


def main(turnover_path=TURNOVER_PATH, sites_path=SITES_PATH, atm_path=ATM_PATH,
         output_dir=OUTPUT_DIR, n_jobs=-1):
    atm = load_atm14c(atm_path)
    durations = pd.read_csv(sites_path)['Duration_labeling']
    turnover_14C = pd.read_csv(turnover_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calibrate and predict for all sites at each ratio. The CDFs are written out
    # per ratio in the same layout the Julia script used, because 06c reads the
    # ratio=1 file.
    ages = {}
    fnew = {suffix: {} for suffix in SCENARIOS}
    for ratio, label in zip(RATIOS, RATIO_LABELS):
        print(f'Calibrating ratio {label} for {len(turnover_14C)} sites...')
        scaled = turnover_14C[['fm_obs']].copy()
        for column in SCENARIOS.values():
            scaled[column] = turnover_14C[column] * ratio
        fits = calibrate(lognormal_spec(), scaled, atm=atm, n_jobs=n_jobs)
        for suffix in SCENARIOS:
            print(f'  {suffix or "central"}: {fits["calibration_status"+suffix].value_counts().to_dict()}')

        ages[label] = fits['model_A']
        cdfs = Parallel(n_jobs=n_jobs)(delayed(stock_younger_than_grid)(mu, sigma)
                                       for mu, sigma in zip(fits.mu, fits.sigma))
        pd.DataFrame(np.vstack(list(cdfs)), columns=TS).to_csv(
            output_dir/f'06a_lognormal_cdfs_{label}.csv', index=False)
        for suffix in SCENARIOS:
            fnew[suffix][ratio] = predict_fnew(fits, suffix, durations)

    pd.DataFrame(ages).to_csv(output_dir/'lognormal_age_predictions.csv', index=False)
    # Rows of sites without a backfilled estimate are NaN in the q05/q95 files.
    for suffix, name in [('', '06a_lognormal_fnew'), ('_05', '06a_lognormal_fnew_q05'),
                         ('_95', '06a_lognormal_fnew_q95')]:
        pd.DataFrame(fnew[suffix], index=durations.index).to_csv(output_dir/f'{name}.csv', index=False)
    print(f'wrote results to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=TURNOVER_PATH, help='Turnover and fm table from script 02.')
    parser.add_argument('--sites', default=SITES_PATH, help='Processed sites, for the labeling durations.')
    parser.add_argument('--atm', default=ATM_PATH, help='Atmospheric 14C record.')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Directory for the outputs.')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs over sites (-1 uses all cores).')
    args = parser.parse_args()
    main(args.input, args.sites, args.atm, args.output_dir, args.n_jobs)
