"""Run conditional fits and sensitivity scans: python -m soil_diskin.layered_workflow."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .layered_data import PreparedProfiles, file_digest, load_profiles
from .layered_fitting import FitSettings, fit_profile
from .layered_lognormal import LayeredLognormal, MU_BOUNDS, SIGMA_BOUNDS, Prediction
from .radiocarbon_utils import AtmC14, load_atm14c


def _append(path: Path, rows: list[dict] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(path, index=False, mode='a', header=not path.exists())


def _save_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run_profiles(prepared: PreparedProfiles, atmosphere: AtmC14,
                 hyperparameters: list[tuple[float, float, float]], output_dir: str | Path, *,
                 settings: FitSettings | None = None, times: tuple[float, ...] = (),
                 log_rate_step: float = 0.05,
                 mu_bounds: tuple[float, float] = MU_BOUNDS,
                 sigma_bounds: tuple[float, float] = SIGMA_BOUNDS,
                 verbose: bool = False) -> dict:
    """Write linked CSVs, checking each candidate on a twice-finer quadrature.

    Each shared triple reuses one forward model across profiles. Partial CSVs
    survive an interrupted run; use a new directory to avoid mixing runs.
    A completed run can include failed profile fits: inspect fits/attempts.
    """
    settings = settings or FitSettings()
    if not hyperparameters:
        raise ValueError('supply at least one explicit (D, v, h) triple')
    for triple in hyperparameters:
        if (len(triple) != 3 or not np.isfinite(triple).all()
                or triple[0] < 0 or triple[1] < 0 or triple[2] <= 0):
            raise ValueError('each triple requires finite D >= 0, v >= 0, h > 0')
    if not np.isfinite(times).all() or np.any(np.asarray(times) < 0):
        raise ValueError('prediction times must be finite and nonnegative')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('output directory must be new or empty; existing runs are preserved')
    output.mkdir(parents=True, exist_ok=True)
    prepared.profiles.to_csv(output/'profiles.csv', index=False)
    prepared.excluded.to_csv(output/'exclusions.csv', index=False)
    started = time.perf_counter()
    metadata = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'settings': asdict(settings), 'hyperparameters_D_v_h': hyperparameters,
        'requested_times_years': list(times), 'mu_bounds': mu_bounds, 'sigma_bounds': sigma_bounds,
        'log_rate_step_requested': log_rate_step, 'quadrature_refinement_factor': 2,
        'quadrature_scaled_tolerance': 1e-3, 'quadrature_fnew_absolute_tolerance': 1e-6,
        'evaluation_used_for_fitting': False, 'data': prepared.metadata,
        'profile_ids': prepared.profiles.profile_id.drop_duplicates().tolist(),
        'profile_triples_with_candidates': 0, 'profile_triples_without_candidates': 0,
        'primary_converged': 0, 'primary_quadrature_ok': 0, 'primary_prediction_success': 0,
        'source_code_sha256': {name: file_digest(Path(__file__).with_name(name)) for name in
                               ['layered_lognormal.py', 'layered_fitting.py',
                                'layered_data.py', 'layered_workflow.py']}}
    _save_metadata(output/'run.json', metadata)
    try:
        for triple_index, (diffusion, velocity, depth) in enumerate(hyperparameters):
            common = {'hyper_id': triple_index, 'D_cm2_yr': diffusion,
                      'v_cm_yr': velocity, 'h_cm': depth}
            try:
                model = LayeredLognormal(diffusion, velocity, depth, atmosphere,
                                        log_rate_step=log_rate_step, mu_bounds=mu_bounds,
                                        sigma_bounds=sigma_bounds)
                refined = LayeredLognormal(diffusion, velocity, depth, atmosphere,
                                          log_rate_step=log_rate_step/2, mu_bounds=mu_bounds,
                                          sigma_bounds=sigma_bounds)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                _append(output/'attempts.csv', [{**common, 'profile_id': '', 'start_index': -1,
                        'success': False, 'objective': np.nan, 'nfev': 0,
                        'message': f'forward model initialization failed: {error}'}])
                metadata['profile_triples_without_candidates'] += prepared.profiles.profile_id.nunique()
                continue
            for profile_id, frame in prepared.profiles.groupby('profile_id', sort=False):
                frame = frame.sort_values('layer').reset_index(drop=True)
                identity = {**common, 'profile_id': profile_id}
                if verbose:
                    print(f"Fitting triple {triple_index}, profile {profile_id}", flush=True)
                try:
                    if not np.array_equal(frame.layer.to_numpy(), np.arange(10)):
                        raise ValueError('a profile must contain exactly layers 0..9')
                    for column in ('npp_kg_m2_yr', 'duration_years'):
                        if frame[column].nunique(dropna=False) != 1:
                            raise ValueError(f'inconsistent {column} within profile')
                    npp = float(frame.npp_kg_m2_yr.iloc[0])
                    stocks, fm = frame.stock_kg_m2.to_numpy(float), frame.fm_obs.to_numpy(float)
                    fit = fit_profile(model, stocks, fm, npp, settings=settings)
                    # Fixed columns keep append-mode CSVs consistent after failed starts.
                    attempts = [{**identity, 'start_index': a['start_index'], 'success': a['success'],
                                 'objective': a.get('objective', np.nan), 'nfev': a.get('nfev', 0),
                                 'message': a['message']} for a in fit.attempts]
                    _append(output/'attempts.csv', attempts)
                    if not fit.candidates:
                        metadata['profile_triples_without_candidates'] += 1
                        continue
                    metadata['profile_triples_with_candidates'] += 1
                    duration = float(frame.duration_years.iloc[0])
                    prediction_times = sorted(set(times) | ({duration} if np.isfinite(duration)
                                                           and duration >= 0 else set()))
                    spread_predictions = []
                    for candidate_id, candidate in enumerate(fit.candidates):
                        key = {**identity, 'candidate_id': candidate_id}
                        prediction_error = ''
                        try:
                            pred = model.predict(candidate.mu, candidate.sigma, npp, tuple(prediction_times))
                            fine = refined.predict(candidate.mu, candidate.sigma, npp, tuple(prediction_times))
                            stock_error = float(np.max(np.abs(pred.stocks-fine.stocks)
                                                       / (settings.stock_relative_scale*stocks)))
                            fm_error = float(np.max(np.abs(pred.fm-fine.fm)/settings.fm_scale))
                            new_error = float(np.max(np.abs(pred.fnew-fine.fnew), initial=0))
                            quadrature_ok = max(stock_error, fm_error) <= 1e-3 and new_error <= 1e-6
                        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                            # Preserve a fitted candidate even if its requested predictions fail.
                            prediction_error = str(error)
                            pred = Prediction(candidate.prediction.stocks, candidate.prediction.fm,
                                              np.asarray(prediction_times),
                                              np.full((len(prediction_times), 10), np.nan))
                            stock_error = fm_error = new_error = np.nan
                            quadrature_ok = False
                        _append(output/'fits.csv', [{**key, 'is_primary': candidate_id == 0,
                            'success': candidate.success, 'objective': candidate.objective,
                            'near_best': candidate.near_best, 'start_index': candidate.start_index,
                            'nfev': candidate.nfev, 'optimality': candidate.optimality,
                            'jacobian_rank': candidate.jacobian_rank,
                            'singular_values': json.dumps(candidate.singular_values.tolist()),
                            'parameters_at_bounds': int(candidate.at_bounds.sum()),
                            'quadrature_stock_scaled_error': stock_error,
                            'quadrature_fm_scaled_error': fm_error,
                            'quadrature_fnew_absolute_error': new_error,
                            'quadrature_ok': quadrature_ok, 'prediction_success': not prediction_error,
                            'prediction_error': prediction_error, 'message': candidate.message}])
                        if candidate_id == 0:
                            metadata['primary_converged'] += int(candidate.success)
                            metadata['primary_quadrature_ok'] += int(quadrature_ok)
                            metadata['primary_prediction_success'] += int(not prediction_error)
                        parameters = frame.copy().assign(**key, mu=candidate.mu, sigma=candidate.sigma,
                            stock_pred_kg_m2=pred.stocks, fm_pred=pred.fm,
                            stock_scaled_residual=candidate.residuals[:10],
                            fm_scaled_residual=candidate.residuals[10:],
                            mu_at_bound=candidate.at_bounds[:10], sigma_at_bound=candidate.at_bounds[10:])
                        _append(output/'parameters.csv', parameters)
                        for time_index, elapsed in enumerate(prediction_times):
                            _append(output/'predictions.csv', [
                                {**key, 'layer': layer, 'z_top_cm': float(frame.z_top_cm.iloc[layer]),
                                 'z_bottom_cm': float(frame.z_bottom_cm.iloc[layer]),
                                 'time_years': elapsed, 'fnew_pred': float(pred.fnew[time_index, layer]),
                                 'fnew_obs': float(frame.fnew_obs.iloc[layer]) if elapsed == duration else np.nan,
                                 'at_label_duration': elapsed == duration}
                                for layer in range(10)])
                        if candidate.near_best and quadrature_ok:
                            spread_predictions.append(pred.fnew)
                    if spread_predictions and prediction_times:
                        stacked = np.stack(spread_predictions)
                        low, high = stacked.min(axis=0), stacked.max(axis=0)
                        _append(output/'prediction_spread.csv', [
                            {**identity, 'time_years': elapsed, 'layer': layer,
                             'candidate_count': len(stacked), 'fnew_min': low[t, layer],
                             'fnew_max': high[t, layer]}
                            for t, elapsed in enumerate(prediction_times) for layer in range(10)])
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    metadata['profile_triples_without_candidates'] += 1
                    _append(output/'attempts.csv', [{**identity, 'start_index': -1, 'success': False,
                        'objective': np.nan, 'nfev': 0, 'message': f'profile processing failed: {error}'}])
                finally:
                    _save_metadata(output/'run.json', metadata)
        metadata['status'] = 'complete'
    except BaseException:
        metadata['status'] = 'interrupted_or_failed'
        raise
    finally:
        metadata['elapsed_seconds'] = time.perf_counter()-started
        _save_metadata(output/'run.json', metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hyper', nargs=3, type=float, action='append', required=True,
                        metavar=('D', 'V', 'H'), help='repeat for a sensitivity scan; cm²/yr, cm/yr, cm')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--balesdent', type=Path, default=Path('data/balesdent_2018/balesdent_2018_raw.xlsx'))
    parser.add_argument('--shi', type=Path, default=Path('data/shi_2020/global_delta_14C.nc'))
    parser.add_argument('--npp', type=Path, default=Path('results/all_sites_14C_turnover.csv'))
    parser.add_argument('--atmosphere', type=Path, default=Path('data/14C_atm_annot.csv'))
    parser.add_argument('--profile', action='append', help='exact Internal_profile_ID; repeat to select profiles')
    parser.add_argument('--limit', type=int, help='first N eligible profiles, for a smoke run')
    parser.add_argument('--times', type=float, nargs='*', default=[])
    parser.add_argument('--starts', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-nfev', type=int, default=500)
    parser.add_argument('--stock-relative-scale', type=float, default=0.10)
    parser.add_argument('--fm-scale', type=float, default=0.02)
    parser.add_argument('--near-relative', type=float, default=0.01)
    parser.add_argument('--near-absolute', type=float, default=1e-6)
    parser.add_argument('--distinct-distance', type=float, default=1e-3)
    parser.add_argument('--log-rate-step', type=float, default=0.05)
    parser.add_argument('--mu-bounds', type=float, nargs=2, default=MU_BOUNDS)
    parser.add_argument('--sigma-bounds', type=float, nargs=2, default=SIGMA_BOUNDS)
    args = parser.parse_args()
    settings = FitSettings(args.starts, args.seed, args.max_nfev, args.stock_relative_scale,
                           args.fm_scale, args.near_relative, args.near_absolute, args.distinct_distance)
    prepared = load_profiles(args.balesdent, args.shi, args.npp)
    selected = prepared.profiles.profile_id.drop_duplicates().tolist()
    if args.profile:
        missing = set(args.profile)-set(selected)
        if missing:
            parser.error(f'unknown or ineligible profiles: {sorted(missing)}')
        selected = [p for p in selected if p in args.profile]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error('--limit must be positive')
        selected = selected[:args.limit]
    prepared.profiles = prepared.profiles[prepared.profiles.profile_id.isin(selected)]
    prepared.metadata['atmosphere'] = {'path': str(args.atmosphere.resolve()),
                                      'sha256': file_digest(args.atmosphere)}
    result = run_profiles(prepared, load_atm14c(str(args.atmosphere)),
                          [tuple(t) for t in args.hyper], args.output_dir,
                          settings=settings, times=tuple(args.times),
                          log_rate_step=args.log_rate_step,
                          mu_bounds=tuple(args.mu_bounds), sigma_bounds=tuple(args.sigma_bounds),
                          verbose=True)
    print(f"Wrote {args.output_dir}: {result['primary_converged']} converged primary fits; "
          f"{result['profile_triples_without_candidates']} failed profile/triple runs.")


if __name__ == '__main__':
    main()
