"""Run the no-transport pipeline: python -m soil_diskin.layered_workflow.

Read run_profiles from top to bottom: allocate inputs, fit layers, predict, save.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .layered_data import PreparedProfiles, file_digest, load_profiles
from .layered_lognormal import Array, LayerLognormal, fit_layer, input_weights
from .radiocarbon_utils import AtmC14, load_atm14c


def plot_comparison(predictions: pd.DataFrame, output: Path) -> None:
    """Evaluate primary predictions at observed labeling times; equal layer weights."""
    import matplotlib.pyplot as plt
    from permetrics.regression import RegressionMetric

    pairs = predictions[(predictions.candidate_id == 0) & predictions.at_label_duration
                        & predictions.quadrature_ok].dropna(subset=['fnew_obs', 'fnew_pred'])
    if pairs.empty:
        return
    observed, predicted = pairs.fnew_obs.to_numpy(), pairs.fnew_pred.to_numpy()
    rmse = float(np.sqrt(np.mean((predicted-observed)**2)))
    kge = np.nan
    if len(pairs) > 1 and min(observed.std(), predicted.std(), observed.mean(), predicted.mean()) > 1e-14:
        kge = float(RegressionMetric(y_true=observed, y_pred=predicted)
                    .kling_gupta_efficiency(force_finite=False))
    pd.DataFrame([{'n_profiles': pairs.profile_id.nunique(), 'n_layer_pairs': len(pairs),
                   'rmse': rmse, 'kge_2012': kge,
                   'unconverged_layer_pairs': int((~pairs.success).sum())}]).to_csv(output/'metrics.csv', index=False)
    fig, ax = plt.subplots(figsize=(5.6, 5.6), layout='constrained')
    for success, color, marker, label in [(True, '#2373ac', 'o', 'Converged'),
                                         (False, '#ce6428', 'x', 'Unconverged')]:
        frame = pairs[pairs.success == success]
        if len(frame):
            ax.scatter(frame.fnew_obs, frame.fnew_pred, s=22, c=color, marker=marker,
                       alpha=.6, label=label)
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), aspect='equal',
           xlabel='Observed new-carbon fraction', ylabel='Predicted new-carbon fraction',
           title=f'No transport · h = {pairs.input_depth_cm.iloc[0]:g} cm\n'
                 f'{pairs.profile_id.nunique()} profiles · {len(pairs)} layer pairs')
    ax.text(.04, .96, f'RMSE = {rmse:.4f}\nKGE (2012) = {kge:.3f}', transform=ax.transAxes,
            va='top', bbox={'facecolor': 'white', 'edgecolor': 'lightgray', 'alpha': .9})
    ax.legend(loc='lower right')
    ax.grid(alpha=.15)
    for extension in ['png', 'pdf']:
        fig.savefig(output/f'fnew_scatter.{extension}', dpi=200)
    plt.close(fig)


def run_profiles(prepared: PreparedProfiles, atmosphere: AtmC14, output_dir: str | Path, *,
                 input_depth: float = 30., times: tuple[float, ...] = (),
                 max_nfev: int = 500, log_rate_step: float = .05,
                 verbose: bool = False) -> dict:
    """Fit every complete profile; h is supplied and f_new is evaluation data only."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a new or empty output directory; existing runs are preserved')
    if not len(prepared.profiles):
        raise ValueError('no complete profiles to fit')
    if not np.isfinite(times).all() or np.any(np.asarray(times) < 0):
        raise ValueError('times must be finite and nonnegative')
    # 1. Allocate NPP and prepare the same independent-layer model for every site.
    weights = input_weights(input_depth)
    model = LayerLognormal(atmosphere, log_rate_step=log_rate_step)
    refined = LayerLognormal(atmosphere, log_rate_step=log_rate_step/2)
    output.mkdir(parents=True, exist_ok=True)
    prepared.excluded.to_csv(output/'exclusions.csv', index=False)
    metadata = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'model': 'independent lognormal layers; no transport', 'input_depth_cm': input_depth,
                'max_nfev_per_start': max_nfev, 'starting_sigmas': [2.5, 1., 4.],
                'mu_bounds': model.mu_bounds, 'sigma_bounds': model.sigma_bounds,
                'stock_relative_scale': .1, 'fm_scale': .02, 'log_rate_step': log_rate_step,
                'requested_times_years': list(times), 'evaluation_used_for_parameter_fitting': False,
                'model_selection_used_fnew': True, 'data': prepared.metadata,
                'source_code_sha256': {name: file_digest(Path(__file__).with_name(name)) for name in
                                      ['layered_lognormal.py', 'layered_data.py', 'layered_workflow.py']}}
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    fit_rows, prediction_rows = [], []
    try:
        for profile_id, profile in prepared.profiles.groupby('profile_id', sort=False):
            if not np.array_equal(np.sort(profile.layer), np.arange(10)):
                raise ValueError(f'{profile_id}: expected exactly layers 0..9')
            if profile.npp_kg_m2_yr.nunique(dropna=False) != 1:
                raise ValueError(f'{profile_id}: inconsistent site NPP')
            if verbose:
                print(f'Fitting {profile_id}', flush=True)
            for _, observed in profile.sort_values('layer').iterrows():
                rate = observed.npp_kg_m2_yr*weights[int(observed.layer)]
                base = {**observed.to_dict(), 'input_depth_cm': input_depth,
                        'input_kg_m2_yr': rate, 'observed_turnover_years': observed.stock_kg_m2/rate}
                duration = observed.duration_years
                prediction_times = sorted(set(times) | ({float(duration)} if np.isfinite(duration)
                                                       and duration >= 0 else set()))
                # 2. Each layer fits only its own stock and radiocarbon observations.
                try:
                    candidates = fit_layer(model, observed.stock_kg_m2, observed.fm_obs, rate,
                                           max_nfev=max_nfev)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    candidates = [{'candidate_id': 0, 'mu': np.nan, 'sigma': np.nan,
                                   'success': False, 'near_best': False, 'message': str(error)}]
                for candidate in candidates:
                    fitted = {**base, **candidate, 'quadrature_ok': False, 'prediction_error': ''}
                    values: Array = np.full(len(prediction_times), np.nan)
                    # 3. Predict new carbon after fitting; check a twice-finer integration grid.
                    try:
                        args = (candidate['mu'], candidate['sigma'], rate, tuple(prediction_times))
                        prediction, fine = model.predict(*args), refined.predict(*args)
                        values = prediction.fnew
                        fm_error = abs(prediction.fm-fine.fm)
                        new_error = float(np.max(np.abs(values-fine.fnew), initial=0))
                        fitted.update(quadrature_fm_error=fm_error, quadrature_fnew_error=new_error,
                                      quadrature_ok=fm_error <= 2e-5 and new_error <= 1e-6)
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                        fitted['prediction_error'] = str(error)
                    fitted['fnew_pred'] = (values[prediction_times.index(duration)]
                                           if duration in prediction_times else np.nan)
                    fit_rows.append(fitted)
                    for elapsed, value in zip(prediction_times, values):
                        prediction_rows.append({'profile_id': profile_id, 'layer': observed.layer,
                            'z_top_cm': observed.z_top_cm, 'z_bottom_cm': observed.z_bottom_cm,
                            'input_depth_cm': input_depth, 'candidate_id': candidate['candidate_id'],
                            'success': candidate['success'], 'near_best': candidate['near_best'],
                            'quadrature_ok': fitted['quadrature_ok'], 'time_years': elapsed,
                            'fnew_pred': value, 'at_label_duration': elapsed == duration,
                            'fnew_obs': observed.fnew_obs if elapsed == duration else np.nan})
        metadata['status'] = 'fits_complete'
    finally:
        # 4. Save the primary result, all starts, predictions, and settings (partial on interruption).
        if metadata['status'] != 'fits_complete':
            metadata['status'] = 'interrupted_or_failed'
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
            near = predictions[predictions.near_best & predictions.quadrature_ok]
            near.groupby(['profile_id', 'layer', 'time_years']).fnew_pred.agg(
                fnew_min='min', fnew_max='max', start_count='size').to_csv(output/'prediction_spread.csv')
        (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    # 5. Compare predicted and observed f_new only after fitting is finished.
    try:
        if len(predictions):
            plot_comparison(predictions, output)
        metadata['status'] = 'complete'
    finally:
        if metadata['status'] != 'complete':
            metadata['status'] = 'interrupted_or_failed'
        (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-depth', type=float, default=30., help='shared NPP e-folding depth in cm')
    parser.add_argument('--output-dir', type=Path, default=Path('results/layered_no_transport'))
    parser.add_argument('--times', type=float, nargs='*', default=[], help='extra prediction times in years')
    parser.add_argument('--limit', type=int, help='first N eligible profiles; default: all')
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
    prepared = load_profiles(args.balesdent, args.shi, args.npp)
    if args.limit:
        ids = prepared.profiles.profile_id.drop_duplicates().head(args.limit)
        prepared.profiles = prepared.profiles[prepared.profiles.profile_id.isin(ids)]
    prepared.metadata['atmosphere'] = {'path': str(Path(args.atmosphere).resolve()),
                                      'sha256': file_digest(args.atmosphere)}
    result = run_profiles(prepared, load_atm14c(args.atmosphere), args.output_dir,
                          input_depth=args.input_depth, times=tuple(args.times),
                          max_nfev=args.max_nfev, log_rate_step=args.log_rate_step, verbose=True)
    print(f"Saved {args.output_dir}: {result['converged_layers']}/{result['n_layers']} layer fits converged.")


if __name__ == '__main__':
    main()
