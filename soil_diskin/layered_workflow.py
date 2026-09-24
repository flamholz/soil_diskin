"""Run the no-transport pipeline: python -m soil_diskin.layered_workflow.

Read run_profiles from top to bottom: read prepared inputs, fit layers, predict, save.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .layered_data import PreparedProfiles, file_digest, load_profiles
from .layered_evaluation import plot_comparison as plot_comparison
from .layered_lognormal import FitResult, InputAllocation, layer_model, fit_layer
from .radiocarbon_utils import AtmC14, load_atm14c
from .run_output import require_empty_output, run_record


def source_hashes() -> dict[str, str]:
    """Fingerprint the shared scientific and reporting code used by all drivers."""
    names = ['layered_lognormal.py', 'layered_data.py', 'layered_workflow.py',
             'layered_evaluation.py', 'run_output.py', 'continuum_models.py',
             'lognormal.py', 'data_wrangling.py', 'radiocarbon_utils.py', 'constants.py']
    return {str(Path(__file__).with_name(name).resolve()): file_digest(Path(__file__).with_name(name))
            for name in names}


def run_profiles(prepared: PreparedProfiles, atmosphere: AtmC14, output_dir: str | Path, *,
                 times: tuple[float, ...] = (),
                 max_nfev: int = 500, log_rate_step: float = .05, verbose: bool = False) -> dict:
    """Fit prepared layer inputs → predict → save tables → evaluate f_new."""
    output = Path(output_dir)
    require_empty_output(output)
    if prepared.profiles.empty:
        raise ValueError('no usable layers to fit')
    profiles = prepared.profiles.sort_values(['profile_id', 'layer'])
    if profiles.duplicated(['profile_id', 'layer']).any() or not (profiles.layer.ge(0) & profiles.layer.mod(1).eq(0)).all():
        raise ValueError('expected distinct nonnegative integer layer indices for each profile')
    if profiles.groupby('profile_id').npp_kg_m2_yr.nunique(dropna=False).ne(1).any():
        raise ValueError('inconsistent site NPP')
    if not np.isfinite(times).all() or np.any(np.asarray(times) < 0):
        raise ValueError('times must be finite and nonnegative')
    if 'input_kg_m2_yr' not in profiles:
        raise ValueError('prepare layer NPP inputs before fitting, using 02 or allocate_inputs')
    model = layer_model(atmosphere, log_rate_step=log_rate_step)
    refined = layer_model(atmosphere, log_rate_step=log_rate_step/2)
    settings = profiles[['input_depth_cm', 'surface_fraction', 'soil_npp_fraction']].drop_duplicates()
    allocation = InputAllocation(*settings.iloc[0]).metadata if len(settings) == 1 and np.isfinite(settings).all().all() else {}
    if 'zmid_cm' in profiles and allocation:
        for key in ['layer_input_weights', 'layer_npp_fractions']:
            allocation['reference_10cm_'+key] = allocation.pop(key)
    metadata = {'model': 'independent lognormal layers; no transport', **allocation,
                'max_nfev_per_start': max_nfev, 'starting_sigmas': [2.5, 1., 4.],
                'mu_bounds': model.mu_bounds, 'sigma_bounds': model.sigma_bounds,
                'stock_relative_scale': .1, 'fm_scale': .02, 'log_rate_step': log_rate_step,
                'requested_times_years': list(times), 'evaluation_used_for_parameter_fitting': False,
                'model_selection_used_fnew': True, 'data': prepared.metadata,
                'source_code_sha256': source_hashes()}
    output.mkdir(parents=True, exist_ok=True)
    prepared.excluded.to_csv(output/'exclusions.csv', index=False)
    profiles.to_csv(output/'prepared_inputs.csv', index=False)
    if verbose:
        print(f'Fitting {len(profiles)} layers from {profiles.profile_id.nunique()} profiles', flush=True)
    rows = []
    with run_record(output/'run.json', metadata):
        try:
            for _, observed in profiles.iterrows():
                rows.extend(_fit_and_predict(observed, model, refined, times, max_nfev))
        finally:
            layers = _save_tables(rows, output, metadata)  # Keep completed layers on interruption.
        plot_comparison(layers, output)
    return metadata


def _fit_and_predict(observed, model, refined, times, max_nfev) -> list[dict]:
    """Keep all starts; observed f_new is copied to outputs but never passed to the fitter."""
    duration = observed.duration_years
    prediction_times = sorted(set(times) | ({float(duration)} if np.isfinite(duration) and duration >= 0 else set()))
    try:
        candidates = fit_layer(model, observed.stock_kg_m2, observed.fm_obs,
                               observed.input_kg_m2_yr, max_nfev=max_nfev)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        candidates = [FitResult(message=str(error))]
    rows = []
    for candidate in candidates:
        row = {**observed.to_dict(), **asdict(candidate), 'quadrature_ok': False, 'prediction_error': ''}
        new_carbon = np.full(len(prediction_times), np.nan)
        try:
            args = (candidate.mu, candidate.sigma, observed.input_kg_m2_yr, tuple(prediction_times))
            prediction, fine = model.predict(*args), refined.predict(*args)
            new_carbon = prediction.fnew
            fm_error = abs(prediction.fm-fine.fm)
            new_error = float(np.max(np.abs(new_carbon-fine.fnew), initial=0))
            row.update(quadrature_fm_error=fm_error, quadrature_fnew_error=new_error,
                       quadrature_ok=fm_error <= 2e-5 and new_error <= 1e-6)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            row['prediction_error'] = str(error)
        row['fnew_pred'] = new_carbon[prediction_times.index(duration)] if duration in prediction_times else np.nan
        rows.append({**row, 'time_years': prediction_times, 'fnew_at_times': new_carbon})
    return rows


def _save_tables(rows: list[dict], output: Path, metadata: dict) -> pd.DataFrame:
    """Derive every output from the fit records; expand time arrays only for predictions.csv."""
    if not rows:
        return pd.DataFrame()
    fits = pd.DataFrame(rows)
    fits['observed_turnover_years'] = fits.implied_turnover_years  # Existing CSV alias.
    fits.drop(columns=['time_years', 'fnew_at_times']).to_csv(output/'fits.csv', index=False)
    primary = fits[fits.candidate_id == 0].drop(columns=['time_years', 'fnew_at_times'])
    primary.to_csv(output/'layers.csv', index=False)
    metadata.update(n_layers=len(primary), n_profiles=primary.profile_id.nunique(),
                    converged_layers=int(primary.success.sum()),
                    numerically_checked_layers=int(primary.quadrature_ok.sum()))
    columns = ['profile_id', 'layer', 'z_top_cm', 'z_bottom_cm', 'input_depth_cm', 'surface_fraction',
               'soil_npp_fraction', 'candidate_id', 'success', 'near_best', 'quadrature_ok',
               'duration_years', 'fnew_obs', 'time_years', 'fnew_at_times']
    predictions = fits[columns].explode(['time_years', 'fnew_at_times']).dropna(subset=['time_years'])
    predictions = predictions.rename(columns={'fnew_at_times': 'fnew_pred'}).astype({'time_years': float, 'fnew_pred': float})
    predictions['at_label_duration'] = predictions.time_years == predictions.duration_years
    predictions['fnew_obs'] = predictions.fnew_obs.where(predictions.at_label_duration)
    predictions = predictions.drop(columns='duration_years')
    predictions.to_csv(output/'predictions.csv', index=False)
    near_best = predictions[predictions.near_best & predictions.quadrature_ok]
    near_best.groupby(['profile_id', 'layer', 'time_years']).fnew_pred.agg(
        fnew_min='min', fnew_max='max', start_count='size').to_csv(output/'prediction_spread.csv')
    return primary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-table', default='results/all_sites_14C_turnover_depth.csv')
    parser.add_argument('--output-dir', type=Path, default=Path('results/layered_no_transport'))
    parser.add_argument('--times', type=float, nargs='*', default=[], help='extra prediction times in years')
    parser.add_argument('--limit', type=int, help='first N eligible profiles; default: all')
    parser.add_argument('--allow-partial', action='store_true',
                        help='include usable layers from incomplete profiles; do not fill missing inputs')
    parser.add_argument('--max-nfev', type=int, default=500)
    parser.add_argument('--log-rate-step', type=float, default=.05)
    parser.add_argument('--atmosphere', default='data/14C_atm_annot.csv')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.max_nfev < 1:
        parser.error('--max-nfev must be positive')
    prepared = load_profiles(args.input_table, allow_partial=args.allow_partial)
    if args.limit:
        ids = prepared.profiles.profile_id.drop_duplicates().head(args.limit)
        prepared.profiles = prepared.profiles[prepared.profiles.profile_id.isin(ids)]
    prepared.metadata['atmosphere'] = {'path': str(Path(args.atmosphere).resolve()),
                                      'sha256': file_digest(args.atmosphere)}
    result = run_profiles(prepared, load_atm14c(args.atmosphere), args.output_dir,
                          times=tuple(args.times),
                          max_nfev=args.max_nfev, log_rate_step=args.log_rate_step, verbose=True)
    print(f"Saved {args.output_dir}: {result['converged_layers']}/{result['n_layers']} layer fits converged.")


if __name__ == '__main__':
    main()
