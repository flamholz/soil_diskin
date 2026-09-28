"""Fit lognormal mu/sigma directly to turnover and radiocarbon, for bulk or layers.

Fits from three starting points within this script, with 1e-13 optimizer
stopping tolerances and the higher-sigma solution when several fits are exact.
Stock uncertainty scenarios are fitted separately. Missing inputs retain their
rows; converged compromises are flagged as approximate_fit.

Exports mu/sigma and fitted mean ages (pred / pred_05 / pred_95). The existing
output filename is retained so downstream scripts keep finding the table.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares

# Allow running this file directly from the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soil_diskin.continuum_models import LognormalDisKinFast
from soil_diskin.lognormal import cached_radiocarbon
from soil_diskin.radiocarbon_utils import load_atm14c

SITES_PATH = Path('results/all_sites_14C_turnover.csv')
ATM_PATH = Path('data/14C_atm_annot.csv')
OUT_DIR = Path('results/03_calibrate_models')


def fit_observation(atm, fm_evaluator, turnover, fm):
    """Minimize squared relative turnover/Fm errors; prefer higher-sigma exact fits."""
    if not np.isfinite([turnover, fm]).all() or min(turnover, fm) <= 0:
        raise ValueError('positive finite turnover and fm required')
    lower, upper = [-15., .01], [10., 5.]  # Must match cached_radiocarbon's bounds.

    def residual(parameters):
        model = LognormalDisKinFast(*parameters, atm, fm_evaluator=fm_evaluator)
        fm_pred, _ = model.calc_radiocarbon_ratio_ss_fast()
        # Relative turnover error equals relative stock error at the observed input.
        return (np.array([model.T, fm_pred])-[turnover, fm])/np.array([turnover, fm])

    best = dict.fromkeys(['mu', 'sigma', 'pred', 'model_turnover_years', 'fm_pred',
                         'turnover_relative_residual', 'fm_residual'], np.nan)
    best.update(objective=np.inf, calibration_status='fit_failed')
    best_key = (1, np.inf)
    for sigma0 in [2.5, 1., 4.]:
        sigma0 = np.clip(sigma0, lower[1], upper[1])
        start = np.clip([sigma0**2/2-np.log(turnover), sigma0], lower, upper)
        try:
            fit = least_squares(residual, start, bounds=(lower, upper), x_scale='jac',
                                max_nfev=2000, ftol=1e-13, xtol=1e-13, gtol=1e-13)
            mu, sigma = fit.x
            model = LognormalDisKinFast(mu, sigma, atm, fm_evaluator=fm_evaluator)
            fm_pred, _ = model.calc_radiocarbon_ratio_ss_fast()
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        relative_error, fm_error = model.T/turnover-1, fm_pred-fm
        exact = fit.success and abs(relative_error) < 1e-10 and abs(fm_error) < 1e-10
        objective = float(fit.fun@fit.fun)
        key = (0, -sigma) if exact else (1, objective)
        if key >= best_key:
            continue
        best_key = key
        best = {'mu': mu if fit.success else np.nan,
                'sigma': sigma if fit.success else np.nan,
                'pred': model.A if fit.success else np.nan,
                'model_turnover_years': model.T, 'fm_pred': fm_pred,
                'turnover_relative_residual': relative_error, 'fm_residual': fm_error,
                'objective': objective,
                'calibration_status': ('calibrated' if exact else 'approximate_fit') if fit.success else 'fit_failed'}
    return best


def main(n_jobs: int = -1, sites_path: Path = SITES_PATH, out_dir: Path | None = None) -> None:
    """Calibrate each usable observation and stock scenario, preserving input rows."""
    sites = pd.read_csv(sites_path, dtype={'profile_id': str})
    sites = sites.rename(columns={'fm_obs': 'fm', 'implied_turnover_years': 'turnover'})
    numeric = sites.columns.intersection(['turnover', 'fm', 'turnover_q05', 'turnover_q95'])
    sites[numeric] = sites[numeric].apply(pd.to_numeric, errors='coerce').astype(float)
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR/('depth_resolved' if 'z_top_cm' in sites else '')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Preparing atmospheric quadrature; {len(sites)} observations', flush=True)
    atm = load_atm14c(str(ATM_PATH))
    fm_evaluator = cached_radiocarbon(atm)
    out = sites.copy()
    for column, suffix in [('turnover', ''), ('turnover_q05', '_05'), ('turnover_q95', '_95')]:
        for name in ['mu', 'sigma', 'pred', 'model_turnover_years', 'fm_pred',
                     'turnover_relative_residual', 'fm_residual', 'objective']:
            out[name+suffix] = np.nan
        out['calibration_status'+suffix] = 'invalid_input'
        if column not in sites:
            continue
        valid = np.isfinite(sites[[column, 'fm']]).all(axis=1) & sites[column].gt(0) & sites.fm.gt(0)
        observations = sites.loc[valid, [column, 'fm']]
        print(f'Fitting {column}: {len(observations)} rows', flush=True)
        if observations.empty:
            continue
        fits = Parallel(n_jobs=n_jobs)(delayed(fit_observation)(atm, fm_evaluator, tau, fm)
                                      for tau, fm in observations.itertuples(index=False, name=None))
        fitted = pd.DataFrame(fits, index=observations.index).add_suffix(suffix)
        out.loc[valid, fitted.columns] = fitted
        print(fitted['calibration_status'+suffix].value_counts().to_dict(), flush=True)
    out_path = out_dir/'03b_lognormal_predictions_calcurve_python.csv'
    out.to_csv(out_path, index=False)
    print(f'Wrote {out_path} ({len(out)} rows)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', type=Path, default=SITES_PATH, help='Bulk or depth-resolved turnover CSV from script 02.')
    parser.add_argument('--output-dir', type=Path, help='Defaults to results/03_calibrate_models (depth_resolved subfolder for layers).')
    parser.add_argument('--n-jobs', type=int, default=int(os.environ.get('N_JOBS', -1)),
                        help='Number of parallel jobs for joblib (-1 uses all cores).')
    args = parser.parse_args()
    main(n_jobs=args.n_jobs, sites_path=args.input, out_dir=args.output_dir)
