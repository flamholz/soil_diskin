"""Fit lognormal mu/sigma directly to turnover and radiocarbon, for bulk or layers.

Uses the shared calibration in soil_diskin.calibration with lognormal_spec():
three starting points, 1e-13 optimizer stopping tolerances, and the
higher-sigma solution when several fits are exact. Stock uncertainty scenarios
are fitted separately. Missing inputs retain their rows; converged compromises
are flagged as approximate_fit.

Writes the input table with mu/sigma, model_T/model_A/model_fm and fit
diagnostics for each scenario (suffixes '', '_05', '_95') to
lognormal_model_optimization_results.csv.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd

# Allow running this file directly from the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soil_diskin.calibration import calibrate, lognormal_spec
from soil_diskin.radiocarbon_utils import load_atm14c

SITES_PATH = Path('results/all_sites_14C_turnover.csv')
ATM_PATH = Path('data/14C_atm_annot.csv')
OUT_DIR = Path('results/03_calibrate_models')
OUTPUT_NAME = 'lognormal_model_optimization_results.csv'


def main(n_jobs: int = -1, sites_path: Path = SITES_PATH, out_dir: Path | None = None,
         output_name: str = OUTPUT_NAME) -> None:
    """Calibrate each usable observation and stock scenario, preserving input rows."""
    sites = pd.read_csv(sites_path, dtype={'profile_id': str})
    sites = sites.rename(columns={'implied_turnover_years': 'turnover'})
    numeric = sites.columns.intersection(['turnover', 'fm_obs', 'turnover_q05', 'turnover_q95'])
    sites[numeric] = sites[numeric].apply(pd.to_numeric, errors='coerce').astype(float)
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR/('depth_resolved' if 'z_top_cm' in sites else '')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Fitting {len(sites)} observations', flush=True)
    fits = calibrate(lognormal_spec(), sites, atm=load_atm14c(str(ATM_PATH)), n_jobs=n_jobs)
    for suffix in ['', '_05', '_95']:
        print(suffix or 'central', fits['calibration_status'+suffix].value_counts().to_dict(), flush=True)
    out = sites.join(fits)
    out_path = out_dir/output_name
    out.to_csv(out_path, index=False)
    print(f'Wrote {out_path} ({len(out)} rows)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', type=Path, default=SITES_PATH, help='Bulk or depth-resolved turnover CSV from script 02.')
    parser.add_argument('--output-dir', type=Path, help='Defaults to results/03_calibrate_models (depth_resolved subfolder for layers).')
    parser.add_argument('--output-name', default=OUTPUT_NAME,
                        help='Output file name, e.g. for the permuted-fm null.')
    parser.add_argument('--n-jobs', type=int, default=int(os.environ.get('N_JOBS', -1)),
                        help='Number of parallel jobs for joblib (-1 uses all cores).')
    args = parser.parse_args()
    main(n_jobs=args.n_jobs, sites_path=args.input, out_dir=args.output_dir, output_name=args.output_name)
