"""Export calibrated lognormal and power-law results as one tidy CSV per model.

Each row is a processed Balesdent site. Columns are the site identifiers and
observations (source_row, latitude, longitude, turnover, fm_obs, A_r), the
calibrated parameters, and the model's steady-state turnover (model_T)
and mean age (model_A).

A_r is the apparent radiocarbon age, -ln(fm)/LAMBDA_14C, computed from the
observed fm_obs with the project's 14C decay constant. It is not the model's mean
age and is negative where fm_obs > 1 (post-bomb carbon).

source_row is the 0-based row of results/processed_balesdent_2018.csv. Rows are
site-level means, so it does not map back to a single row of the raw workbook.

Each model is also exported for the permuted-fm null (02b_permute_fm.py) as
{model}_calibration_results_permuted.csv, with the same columns plus a leading
replicate column and fm_obs_original. There fm_obs, and so A_r, is the permuted
value the model was calibrated to.

Run from the project root directory:
    python notebooks/04_export_calibration_results.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from soil_diskin.constants import LAMBDA_14C

SITES_PATH = Path('results/processed_balesdent_2018.csv')
CALIBRATION_DIR = Path('results/03_calibrate_models')
OUT_DIR = Path('results/04_calibration_exports')

# name: (calibration file, {output column: input column})
MODELS = {
    'powerlaw': ('powerlaw_model_optimization_results.csv',
                 {'t_min': 't_min', 't_max': 't_max',
                  'model_T': 'model_T', 'model_A': 'model_A'}),
    'lognormal': ('lognormal_model_optimization_results.csv',
                  {'mu': 'mu', 'sigma': 'sigma',
                   'model_T': 'model_T', 'model_A': 'model_A'}),
}
OBSERVATIONS = ['turnover', 'fm_obs']


def apparent_radiocarbon_age(fm):
    """Apparent radiocarbon age in years; NaN unless fm is positive and finite."""
    fm = pd.to_numeric(fm, errors='coerce').astype(float)
    return -np.log(fm.where(fm > 0)) / LAMBDA_14C


def export_model(sites, calibration_path, columns):
    """Combine site identifiers with one model's calibration results.

    Long-format calibrations (permuted-fm replicates) give each row's site in
    site_index; otherwise rows follow the processed sites in order.
    """
    calibration = pd.read_csv(calibration_path)
    if 'site_index' in calibration:
        rows = calibration['site_index'].to_numpy()
    else:
        # Calibration scripts preserve the order of the processed sites.
        if len(calibration) != len(sites):
            raise ValueError(f'{calibration_path} has {len(calibration)} rows; '
                             f'{SITES_PATH} has {len(sites)}')
        rows = sites.index.to_numpy()
    for name, source in [('Latitude', 'latitude'), ('Longitude', 'longitude')]:
        if name in calibration and not np.allclose(calibration[name], sites.loc[rows, name], equal_nan=True):
            raise ValueError(f'{calibration_path} {name} does not match {SITES_PATH} row order')
    out = pd.DataFrame({'source_row': rows, 'latitude': sites.loc[rows, 'Latitude'].to_numpy(),
                        'longitude': sites.loc[rows, 'Longitude'].to_numpy()})
    if 'replicate' in calibration:
        out.insert(0, 'replicate', calibration['replicate'])
    for name in OBSERVATIONS + (['fm_obs_original'] if 'fm_obs_original' in calibration else []):
        out[name] = calibration[name]
    out['A_r'] = apparent_radiocarbon_age(out.fm_obs)
    for name, source in columns.items():
        out[name] = calibration[source]
    return out


def main(sites_path=SITES_PATH, calibration_dir=CALIBRATION_DIR, out_dir=OUT_DIR):
    sites = pd.read_csv(sites_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, (filename, columns) in MODELS.items():
        for suffix in ['', '_permuted']:
            calibration_path = Path(calibration_dir)/filename.replace('.csv', f'{suffix}.csv')
            out = export_model(sites, calibration_path, columns)
            path = out_dir/f'{model}_calibration_results{suffix}.csv'
            out.to_csv(path, index=False)
            print(f'Wrote {path} ({len(out)} rows)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sites', type=Path, default=SITES_PATH)
    parser.add_argument('--calibration-dir', type=Path, default=CALIBRATION_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()
    main(args.sites, args.calibration_dir, args.output_dir)
