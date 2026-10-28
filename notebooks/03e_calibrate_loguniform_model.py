"""
Script should be run from the project root directory.

Calibrates the log-uniform decay-rate model (LogUniformDisKin) to each site's
turnover and radiocarbon with the shared calibration
(soil_diskin.calibration.loguniform_spec), fitting the central turnover and
the q05/q95 stock scenarios. Sites run in parallel.
"""
import argparse
from os import path

import numpy as np
import pandas as pd

from soil_diskin.calibration import calibrate, loguniform_spec

DATA_PATH = 'results/all_sites_14C_turnover.csv'
OUTPUT_PATH = path.join('results/03_calibrate_models', 'loguniform_model_optimization_results.csv')


def main(input_path=DATA_PATH, output_path=OUTPUT_PATH, n_jobs=-1):
    site_data = pd.read_csv(input_path)
    fits = calibrate(loguniform_spec(), site_data, n_jobs=n_jobs)
    # ln k_max, for plotting tau_min = 1/k_max without underflow (figS8).
    for suffix in ['', '_05', '_95']:
        fits['log_k_max'+suffix] = np.log(fits['k_min'+suffix]) + fits['log_width'+suffix]
    for suffix in ['', '_05', '_95']:
        print(suffix or 'central', fits['calibration_status'+suffix].value_counts().to_dict())
    print(f'maximum objective {fits.objective.max():.3e}')
    fits.join(site_data[['fm_obs', 'turnover']]).to_csv(output_path, index=False)
    print(f'wrote {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=DATA_PATH, help='Turnover and fm table from script 02 (or 02b).')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Calibration results CSV.')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs over sites (-1 uses all cores).')
    args = parser.parse_args()
    main(args.input, args.output, args.n_jobs)
