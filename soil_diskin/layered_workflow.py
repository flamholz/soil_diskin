"""Run the no-transport pipeline: python -m soil_diskin.layered_workflow.

Read run_profiles from top to bottom: allocate inputs, fit layers, predict, save.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .layered_data import PreparedProfiles, file_digest, load_profiles
from .layered_evaluation import plot_comparison as plot_comparison
from .layered_lognormal import Array, FitResult, InputAllocation, LayerLognormal, fit_layer
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
                 allocation: InputAllocation | None = None, input_depth: float | None = None,
                 surface_fraction: float | None = None, soil_npp_fraction: float | None = None,
                 times: tuple[float, ...] = (),
                 max_nfev: int = 500, log_rate_step: float = .05,
                 verbose: bool = False) -> dict:
    """Fit every supplied layer; input allocation is fixed and f_new is evaluation only."""
    output = Path(output_dir)
    require_empty_output(output)
    if not len(prepared.profiles):
        raise ValueError('no usable layers to fit')
    if not np.isfinite(times).all() or np.any(np.asarray(times) < 0):
        raise ValueError('times must be finite and nonnegative')
    if allocation is None:  # Preserve existing keyword calls; all internal work uses one allocation.
        allocation = InputAllocation(30. if input_depth is None else input_depth,
                                     0. if surface_fraction is None else surface_fraction,
                                     1. if soil_npp_fraction is None else soil_npp_fraction)
    elif any(value is not None for value in (input_depth, surface_fraction, soil_npp_fraction)):
        raise ValueError('supply allocation or the legacy input keywords, not both')
    # 1. Allocate the soil share of NPP over all ten depths, including unobserved layers.
    model = LayerLognormal(atmosphere, log_rate_step=log_rate_step)
    refined = LayerLognormal(atmosphere, log_rate_step=log_rate_step/2)
    output.mkdir(parents=True, exist_ok=True)
    prepared.excluded.to_csv(output/'exclusions.csv', index=False)
    metadata = {'model': 'independent lognormal layers; no transport', **allocation.metadata,
                'max_nfev_per_start': max_nfev, 'starting_sigmas': [2.5, 1., 4.],
                'mu_bounds': model.mu_bounds, 'sigma_bounds': model.sigma_bounds,
                'stock_relative_scale': .1, 'fm_scale': .02, 'log_rate_step': log_rate_step,
                'requested_times_years': list(times), 'evaluation_used_for_parameter_fitting': False,
                'model_selection_used_fnew': True, 'data': prepared.metadata,
                'source_code_sha256': source_hashes()}
    fit_rows, prediction_rows = [], []
    with run_record(output/'run.json', metadata):
        try:
            for profile_id, profile in prepared.profiles.groupby('profile_id', sort=False):
                if profile.layer.duplicated().any() or not profile.layer.isin(range(10)).all():
                    raise ValueError(f'{profile_id}: expected distinct layer indices in 0..9')
                if profile.npp_kg_m2_yr.nunique(dropna=False) != 1:
                    raise ValueError(f'{profile_id}: inconsistent site NPP')
                if verbose:
                    print(f'Fitting {profile_id}', flush=True)
                for _, observed in profile.sort_values('layer').iterrows():
                    layer_input = allocation.layer_inputs(observed.npp_kg_m2_yr)[int(observed.layer)]
                    layer_record = {**observed.to_dict(), **allocation.parameters,
                                    'input_kg_m2_yr': layer_input,
                                    'implied_turnover_years': observed.stock_kg_m2/layer_input,
                                    'observed_turnover_years': observed.stock_kg_m2/layer_input}  # Legacy CSV alias.
                    duration = observed.duration_years
                    prediction_times = sorted(set(times) | ({float(duration)} if np.isfinite(duration)
                                                           and duration >= 0 else set()))
                    # 2. Each layer fits only its own stock and radiocarbon observations.
                    try:
                        candidates = fit_layer(model, observed.stock_kg_m2, observed.fm_obs, layer_input,
                                               max_nfev=max_nfev)
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                        candidates = [FitResult(message=str(error))]
                    for candidate in candidates:
                        fitted = {**layer_record, **asdict(candidate), 'quadrature_ok': False, 'prediction_error': ''}
                        new_carbon: Array = np.full(len(prediction_times), np.nan)
                        # 3. Predict new carbon after fitting; check a twice-finer integration grid.
                        try:
                            args = (candidate.mu, candidate.sigma, layer_input, tuple(prediction_times))
                            prediction, fine = model.predict(*args), refined.predict(*args)
                            new_carbon = prediction.fnew
                            fm_error = abs(prediction.fm-fine.fm)
                            new_error = float(np.max(np.abs(new_carbon-fine.fnew), initial=0))
                            fitted.update(quadrature_fm_error=fm_error, quadrature_fnew_error=new_error,
                                          quadrature_ok=fm_error <= 2e-5 and new_error <= 1e-6)
                        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                            fitted['prediction_error'] = str(error)
                        fitted['fnew_pred'] = (new_carbon[prediction_times.index(duration)]
                                               if duration in prediction_times else np.nan)
                        fit_rows.append(fitted)
                        for elapsed, value in zip(prediction_times, new_carbon):
                            prediction_rows.append({'profile_id': profile_id, 'layer': observed.layer,
                                'z_top_cm': observed.z_top_cm, 'z_bottom_cm': observed.z_bottom_cm,
                                **allocation.parameters, 'candidate_id': candidate.candidate_id,
                                'success': candidate.success, 'near_best': candidate.near_best,
                                'quadrature_ok': fitted['quadrature_ok'], 'time_years': elapsed,
                                'fnew_pred': value, 'at_label_duration': elapsed == duration,
                                'fnew_obs': observed.fnew_obs if elapsed == duration else np.nan})
        finally:
            # 4. Save the primary result, all starts, predictions, and settings (partial on interruption).
            fits = pd.DataFrame(fit_rows)
            if len(fits):
                fits.to_csv(output/'fits.csv', index=False)
                layers = fits[fits.candidate_id == 0]
                layers.to_csv(output/'layers.csv', index=False)
                metadata.update(n_layers=len(layers), n_profiles=layers.profile_id.nunique(),
                                converged_layers=int(layers.success.sum()),
                                numerically_checked_layers=int(layers.quadrature_ok.sum()))
            predictions = pd.DataFrame(prediction_rows)
            if len(predictions):
                predictions.to_csv(output/'predictions.csv', index=False)
                near_best = predictions[predictions.near_best & predictions.quadrature_ok]
                near_best.groupby(['profile_id', 'layer', 'time_years']).fnew_pred.agg(
                    fnew_min='min', fnew_max='max', start_count='size').to_csv(output/'prediction_spread.csv')
        # 5. Compare predicted and observed f_new only after fitting is finished.
        plot_comparison(predictions, output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-depth', type=float, default=30., help='shared NPP e-folding depth in cm')
    parser.add_argument('--surface-fraction', type=float, default=0.,
                        help='direct share of soil input in top 10 cm; remainder follows exponential over 0–100 cm')
    parser.add_argument('--soil-npp-fraction', type=float, default=1.,
                        help='fraction of site NPP entering the modeled soil column, in (0, 1]')
    parser.add_argument('--output-dir', type=Path, default=Path('results/layered_no_transport'))
    parser.add_argument('--times', type=float, nargs='*', default=[], help='extra prediction times in years')
    parser.add_argument('--limit', type=int, help='first N eligible profiles; default: all')
    parser.add_argument('--allow-partial', action='store_true',
                        help='include usable layers from incomplete profiles; do not fill missing inputs')
    parser.add_argument('--max-nfev', type=int, default=500)
    parser.add_argument('--log-rate-step', type=float, default=.05)
    parser.add_argument('--balesdent', default='data/balesdent_2018/balesdent_2018_raw.xlsx')
    parser.add_argument('--shi', default='data/shi_2020/global_delta_14C.nc')
    parser.add_argument('--npp', default='results/all_sites_14C_turnover.csv')
    parser.add_argument('--atmosphere', default='data/14C_atm_annot.csv')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.max_nfev < 1:
        parser.error('--max-nfev must be positive')
    prepared = load_profiles(args.balesdent, args.shi, args.npp, allow_partial=args.allow_partial)
    if args.limit:
        ids = prepared.profiles.profile_id.drop_duplicates().head(args.limit)
        prepared.profiles = prepared.profiles[prepared.profiles.profile_id.isin(ids)]
    prepared.metadata['atmosphere'] = {'path': str(Path(args.atmosphere).resolve()),
                                      'sha256': file_digest(args.atmosphere)}
    result = run_profiles(prepared, load_atm14c(args.atmosphere), args.output_dir,
                          allocation=InputAllocation(args.input_depth, args.surface_fraction, args.soil_npp_fraction), times=tuple(args.times),
                          max_nfev=args.max_nfev, log_rate_step=args.log_rate_step, verbose=True)
    print(f"Saved {args.output_dir}: {result['converged_layers']}/{result['n_layers']} layer fits converged.")


if __name__ == '__main__':
    main()
