"""
Script should be run from the project root directory.

Calibrates the log-uniform decay-rate model (LogUniformDisKin) to each site's
turnover and radiocarbon with the shared calibration
(soil_diskin.calibration.loguniform_spec), fitting the central turnover and
the q05/q95 stock scenarios. Sites run in parallel.
"""
from os import path

import numpy as np
import pandas as pd

from soil_diskin.calibration import calibrate, loguniform_spec

DATA_PATH = 'results/all_sites_14C_turnover.csv'
OUTPUT_PATH = path.join('results/03_calibrate_models', 'loguniform_model_optimization_results.csv')

# Snakemake's thread allowance when run by Snakemake, otherwise all cores.
N_JOBS = snakemake.threads if 'snakemake' in globals() else -1  # noqa: F821


def main():
    site_data = pd.read_csv(DATA_PATH)
    fits = calibrate(loguniform_spec(), site_data, n_jobs=N_JOBS)
    # ln k_max, for plotting tau_min = 1/k_max without underflow (figS8).
    for suffix in ['', '_05', '_95']:
        fits['log_k_max'+suffix] = np.log(fits['k_min'+suffix]) + fits['log_width'+suffix]
    for suffix in ['', '_05', '_95']:
        print(suffix or 'central', fits['calibration_status'+suffix].value_counts().to_dict())
    print(f'maximum objective {fits.objective.max():.3e}')
    fits.join(site_data[['fm_obs', 'turnover']]).to_csv(OUTPUT_PATH, index=False)
    print(f'wrote {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
