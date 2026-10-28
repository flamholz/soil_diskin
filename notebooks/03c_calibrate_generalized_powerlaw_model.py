"""
Script should be run from the project root directory.

Calibrates the generalized power law model from Rothman, PNAS 2025.

Fits t_min and t_max per site at fixed alpha with the shared calibration
(soil_diskin.calibration.general_powerlaw_spec). The two named alpha values in
ALPHA_RUNS each get their own output file. A sweep over SWEEP_ALPHA_VALUES is
written to a single long-format file with one row per (alpha, site); each
site's fit starts from its fit at the previous alpha. Sites run in parallel.
"""
import argparse
from os import path

import numpy as np
import pandas as pd

from soil_diskin.calibration import calibrate, calibrate_sweep, general_powerlaw_spec
from soil_diskin.constants import GAMMA

DATA_PATH = 'results/all_sites_14C_turnover.csv'
OUTPUT_DIR = 'results/03_calibrate_models/'

ALPHA_RUNS = (
    {
        'alpha': float(np.exp(-GAMMA)),
        'label': 'np.exp(-GAMMA)',
        'output_name': 'general_powerlaw_model_optimization_results.csv',
    },
    {
        'alpha': float(np.exp(-GAMMA) / 2),
        'label': 'np.exp(-GAMMA) / 2',
        'output_name': 'general_powerlaw_model_optimization_results_alpha_half.csv',
    },
)
# Same grid as 03f_calibrate_soft_cutoff_powerlaw_model.py so the hard- and
# soft-cutoff sweeps can be compared directly, except alpha = 1.0:
# GeneralPowerLawDisKin requires alpha < 1, and alpha = 1 is PowerLawDisKin,
# already fit by 03a_calibrate_powerlaw_model.py. At alpha = 0.2 only ~81/99
# sites fit; fast sites push t_min to its 1e-8 yr lower bound.
# 0.55-0.75 add resolution where f_new prediction skill peaks (MSRE at 0.5-0.6, KGE at 0.7-0.8).
SWEEP_ALPHA_VALUES = np.sort(np.append(np.round(np.linspace(0.2, 0.9, 8), 1), [0.55, 0.65, 0.75]))
SWEEP_OUTPUT_NAME = 'general_powerlaw_model_alpha_sweep.csv'


def summarize(fits, label):
    """Print calibration status counts and the worst objective."""
    print(f'{label}: {fits.calibration_status.value_counts().to_dict()}; '
          f'maximum objective {fits.objective.max():.3e}')


def main(input_path=DATA_PATH, output_dir=OUTPUT_DIR, n_jobs=-1):
    site_data = pd.read_csv(input_path)

    for alpha_config in ALPHA_RUNS:
        print(f"Running optimization for generalized power law with alpha = {alpha_config['label']}...")
        fits = calibrate(general_powerlaw_spec(alpha_config['alpha']), site_data,
                         n_jobs=n_jobs, warm_start_scenarios=True)
        summarize(fits, f"alpha = {alpha_config['label']}")
        output_path = path.join(output_dir, alpha_config['output_name'])
        fits.join(site_data[['fm_obs', 'turnover']]).to_csv(output_path, index=False)
        print(f'Saved results to {output_path}')

    print(f'Running alpha sweep over {SWEEP_ALPHA_VALUES.tolist()}...')
    sweep = calibrate_sweep(general_powerlaw_spec, [float(alpha) for alpha in SWEEP_ALPHA_VALUES],
                            site_data, n_jobs=n_jobs)
    for alpha, fits in sweep.groupby('alpha'):
        summarize(fits, f'alpha = {alpha:g}')
    sweep = sweep.join(site_data[['fm_obs', 'turnover']], on='site_index')
    output_path = path.join(output_dir, SWEEP_OUTPUT_NAME)
    sweep.to_csv(output_path, index=False)
    print(f'Saved results to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=DATA_PATH, help='Turnover and fm table from script 02.')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Directory for the calibration results.')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs over sites (-1 uses all cores).')
    args = parser.parse_args()
    main(args.input, args.output_dir, args.n_jobs)
