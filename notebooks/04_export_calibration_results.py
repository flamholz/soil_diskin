"""Export calibrated lognormal and power-law results as one tidy CSV per model and dataset.

Files are named {model}_calibration_results[_permuted]_{dataset}.csv, with
dataset balesdent or israd, all in results/04_calibration_exports.

Each row is a processed Balesdent site. Columns are the site identifiers and
observations (source_row, latitude, longitude, turnover, fm_obs, A_r), the
calibrated parameters, and the model's steady-state turnover (model_T)
and mean age (model_A).

A_r is the apparent radiocarbon age, -ln(fm)/LAMBDA_14C, computed from the
observed fm_obs with the project's 14C decay constant. It is not the model's mean
age and is negative where fm_obs > 1 (post-bomb carbon).

source_row is the 0-based row of results/processed_balesdent_2018.csv. Rows are
site-level means, so it does not map back to a single row of the raw workbook.

Each model is also exported for the permutation null (02b_permute_data.py) as
{model}_calibration_results_permuted_balesdent.csv, with the same columns plus a
leading replicate column and the shuffled column's original value: turnover is
shuffled across sites, so turnover is the shuffled value the model was calibrated
to and turnover_original the site's own, while fm_obs and A_r are the site's own.
(Were fm_obs shuffled instead, the table would carry fm_obs_original, and A_r
would be the apparent age of the shuffled fm.)

With --dataset israd, the ISRaD calibrations (results/03_calibrate_models/israd)
are exported the same way, as {model}_calibration_results[_permuted]_israd.csv. Each row is a
pseudo-bulk profile and sampling year, identified by entry_name, site_name,
pro_name and sampling_year as in ISRaD; source_row is the 0-based row of
results/israd/israd_turnover.csv. fm_obs refers to the sampling year, and so does
A_r, which is therefore not comparable across years. calibration_status is kept:
profiles sampled after the atmospheric record ends are 'no_atmosphere', with no
parameters. The permutation null is exported as for Balesdent, with
turnover_original.

Run from the project root directory:
    python notebooks/04_export_calibration_results.py [--dataset israd]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from soil_diskin.constants import LAMBDA_14C

SITES_PATH = Path('results/processed_balesdent_2018.csv')
CALIBRATION_DIR = Path('results/03_calibrate_models')
OUT_DIR = Path('results/04_calibration_exports')
ISRAD_CALIBRATION_DIR = CALIBRATION_DIR/'israd'
ISRAD_IDENTIFIERS = ['entry_name', 'site_name', 'pro_name', 'sampling_year', 'latitude', 'longitude']

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


def observation_columns(calibration):
    """OBSERVATIONS, each followed by its original value where the permutation null shuffled it."""
    return [c for name in OBSERVATIONS for c in [name, f'{name}_original'] if c in calibration]


def apparent_radiocarbon_age(fm):
    """Apparent radiocarbon age in years; NaN unless fm is positive and finite."""
    fm = pd.to_numeric(fm, errors='coerce').astype(float)
    return -np.log(fm.where(fm > 0)) / LAMBDA_14C


def export_model(sites, calibration_path, columns):
    """Combine site identifiers with one model's calibration results.

    Long-format calibrations (permutation replicates) give each row's site in
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
    for name in observation_columns(calibration):
        out[name] = calibration[name]
    out['A_r'] = apparent_radiocarbon_age(out.fm_obs)
    for name, source in columns.items():
        out[name] = calibration[source]
    return out


def export_israd_model(calibration_path, columns):
    """One model's ISRaD calibration with each profile's ISRaD identifiers.

    The ISRaD calibrations carry every column of their input table, so no join is
    needed. Permutation replicates give each row's profile in site_index.
    """
    calibration = pd.read_csv(calibration_path)
    out = pd.DataFrame(index=calibration.index)
    if 'replicate' in calibration:
        out['replicate'] = calibration['replicate']
        out['source_row'] = calibration['site_index']
    else:
        out['source_row'] = calibration.index
    for name in ISRAD_IDENTIFIERS + observation_columns(calibration):
        out[name] = calibration[name]
    out['A_r'] = apparent_radiocarbon_age(out.fm_obs)
    for name, source in columns.items():
        out[name] = calibration[source]
    out['calibration_status'] = calibration['calibration_status']
    return out


def main_israd(calibration_dir=ISRAD_CALIBRATION_DIR, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, (filename, columns) in MODELS.items():
        for suffix in ['', '_permuted']:
            out = export_israd_model(Path(calibration_dir)/filename.replace('.csv', f'{suffix}.csv'), columns)
            path = out_dir/f'{model}_calibration_results{suffix}_israd.csv'
            out.to_csv(path, index=False)
            print(f'Wrote {path} ({len(out)} rows)')


def main(sites_path=SITES_PATH, calibration_dir=CALIBRATION_DIR, out_dir=OUT_DIR):
    sites = pd.read_csv(sites_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, (filename, columns) in MODELS.items():
        for suffix in ['', '_permuted']:
            calibration_path = Path(calibration_dir)/filename.replace('.csv', f'{suffix}.csv')
            out = export_model(sites, calibration_path, columns)
            path = out_dir/f'{model}_calibration_results{suffix}_balesdent.csv'
            out.to_csv(path, index=False)
            print(f'Wrote {path} ({len(out)} rows)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['balesdent', 'israd'], default='balesdent')
    parser.add_argument('--sites', type=Path, default=SITES_PATH, help='Balesdent sites (not used for israd)')
    parser.add_argument('--calibration-dir', type=Path, help='default: by dataset')
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.dataset == 'israd':
        main_israd(args.calibration_dir or ISRAD_CALIBRATION_DIR, args.output_dir)
    else:
        main(args.sites, args.calibration_dir or CALIBRATION_DIR, args.output_dir)
