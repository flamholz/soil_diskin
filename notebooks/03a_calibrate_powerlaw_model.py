"""
Script should be run from the project root directory.

Calibrates the power law model with alpha = 1 (PowerLawDisKin) to each site's
turnover and radiocarbon with the shared calibration
(soil_diskin.calibration.power_law_spec), fitting the central turnover and the
q05/q95 stock scenarios. Sites run in parallel.

Snakemake also runs this script on the permuted-fm null (--input
results/all_sites_14C_turnover_permuted.csv, see 02b_permute_fm.py); its
replicate, site_index and fm_obs_original columns are carried into the output.
For the ISRaD profiles (--input results/israd/israd_turnover.csv --year-column
sampling_year), each profile is fit with the atmosphere of its sampling year,
and the year column is carried into the output.
"""
import argparse
from os import path

import pandas as pd

from soil_diskin.calibration import calibrate, power_law_spec

DATA_PATH = 'results/all_sites_14C_turnover.csv'
OUTPUT_PATH = path.join('results/03_calibrate_models', 'powerlaw_model_optimization_results.csv')


def main(input_path=DATA_PATH, output_path=OUTPUT_PATH, n_jobs=-1, year_column=None):
    site_data = pd.read_csv(input_path)
    fits = calibrate(power_law_spec(), site_data, n_jobs=n_jobs, year_column=year_column)
    for suffix in ['', '_05', '_95']:
        print(suffix or 'central', fits['calibration_status'+suffix].value_counts().to_dict())
    print(f'maximum objective {fits.objective.max():.3e}')
    ids = [c for c in ['replicate', 'site_index'] if c in site_data]
    observations = [c for c in ['fm_obs', 'turnover', 'fm_obs_original', year_column] if c in site_data]
    site_data[ids].join(fits).join(site_data[observations]).to_csv(output_path, index=False)
    print(f'wrote {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=DATA_PATH, help='Turnover and fm table from script 02 (or 02b).')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Calibration results CSV.')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs over sites (-1 uses all cores).')
    parser.add_argument('--year-column', help='fit each row with the atmosphere of the year in this column '
                                              '(default: ages counted back from 2000 for every row)')
    args = parser.parse_args()
    main(args.input, args.output, args.n_jobs, args.year_column)
