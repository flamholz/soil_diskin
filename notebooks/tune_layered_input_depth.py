"""Tune shared input depth on validation locations, then evaluate a fixed test set.

Run: uv run python notebooks/tune_layered_input_depth.py
The local mu/sigma fits still use only each profile's stocks, radiocarbon, and NPP.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from soil_diskin.layered_evaluation import score as score

from soil_diskin.layered_lognormal import InputAllocation
from soil_diskin.layered_data import PreparedProfiles, file_digest, load_profiles
from soil_diskin.run_output import require_empty_output, run_record
from soil_diskin.layered_workflow import source_hashes, plot_comparison, run_profiles
from soil_diskin.radiocarbon_utils import load_atm14c

H_VALUES = [5., 10., 15., 20., 30., 40., 60., 80., 120., 200.]
BASELINE_H = 30.


def make_split(profiles: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """60/20/20 split of sorted unique locations; never split a location's profiles."""
    identities = profiles[['profile_id', 'latitude', 'longitude']].drop_duplicates()
    if identities.profile_id.duplicated().any():
        raise ValueError('each profile must have exactly one location')
    locations = identities[['latitude', 'longitude']].drop_duplicates().sort_values(
        ['latitude', 'longitude']).reset_index(drop=True)
    if len(locations) < 5:
        raise ValueError('at least five locations required for three nonempty splits')
    order = np.random.default_rng(seed).permutation(len(locations))
    n_train, n_val = int(.6*len(locations)), int(.2*len(locations))
    locations['split'] = 'test'
    locations.loc[order[:n_train], 'split'] = 'train'
    locations.loc[order[n_train:n_train+n_val], 'split'] = 'validation'
    return identities.merge(locations, on=['latitude', 'longitude'], validate='many_to_one')


def choose_h(scores: pd.DataFrame, metric: str = 'rmse') -> float:
    if metric not in ('rmse', 'kge_2012'):
        raise ValueError('selection metric must be rmse or kge_2012')
    valid = scores[(scores.split == 'validation') & scores.eligible & np.isfinite(scores[metric])]
    if valid.empty:
        raise ValueError('no candidate has complete converged, numerically checked validation predictions')
    return float(valid.sort_values([metric, 'h_cm'], ascending=[metric == 'rmse', True]).h_cm.iloc[0])


def plot_search(scores: pd.DataFrame, selected_h: float, output: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), layout='constrained')
    for ax, metric, label in zip(axes, ['rmse', 'kge_2012'], ['RMSE (fraction units)', 'KGE (2012)']):
        for split, color in [('train', '#2476b8'), ('validation', '#d66b22')]:
            frame = scores[scores.split == split].sort_values('h_cm')
            ax.plot(frame.h_cm, frame[metric], 'o-', color=color, label=split.capitalize(), ms=4)
            invalid = frame[~frame.eligible]
            ax.scatter(invalid.h_cm, invalid[metric], marker='x', color='black', s=65, zorder=4)
        ax.axvline(selected_h, color='#63666a', ls='--', lw=1)
        ax.set(xscale='log', xlabel='Input e-folding depth h (cm)', ylabel=label)
        ax.set_xticks(H_VALUES, labels=[f'{h:g}' for h in H_VALUES], rotation=45)
        ax.grid(alpha=.18)
        ax.legend()
    fig.suptitle(f'Input-depth tuning: selected h = {selected_h:g} cm\nTest locations excluded from selection')
    for extension in ['png', 'pdf']:
        fig.savefig(output/f'validation_curve.{extension}', dpi=200)
    plt.close(fig)


def run_experiment(output: Path, *, metric: str = 'rmse', seed: int = 42, max_nfev: int = 1000) -> None:
    require_empty_output(output)
    prepared = load_profiles()
    atmosphere = load_atm14c()
    prepared.metadata['atmosphere'] = {'path': str(Path('data/14C_atm_annot.csv').resolve()),
                                      'sha256': file_digest('data/14C_atm_annot.csv')}
    split = make_split(prepared.profiles, seed)
    profiles = prepared.profiles.merge(split[['profile_id', 'split']], on='profile_id', validate='many_to_one')
    if (not np.isfinite(profiles.fnew_obs).all() or not np.isfinite(profiles.duration_years).all()
            or not profiles.duration_years.gt(0).all()):
        raise ValueError('this experiment requires complete f_new observations and positive labeling durations')
    output.mkdir(parents=True, exist_ok=True)
    split.to_csv(output/'splits.csv', index=False)
    # 1. Freeze the split, grid, metric, solver settings, and test comparison before fitting.
    protocol = {
        'seed': seed, 'location_fractions': [.6, .2, .2], 'h_grid_cm': H_VALUES,
        'selection_metric': metric, 'tie_break': 'smaller h', 'baseline_h_cm': BASELINE_H,
        'max_nfev_per_start': max_nfev, 'failure_policy': 'any failed validation layer makes h ineligible',
        'weighting': 'equal weight per layer; ten layers per profile',
        'test_policy': 'evaluate selected h and predeclared h=30 baseline once, after selection',
        'retrospective_split': True, 'earlier_full_data_used_to_select_no_transport': True,
        'calibration_inputs_at_every_site': 'local stock, radiocarbon, NPP; never local f_new',
        'training_role': 'local fits and diagnostic scores; no shared mu/sigma regression is learned',
        'data': prepared.metadata, 'split_sha256': file_digest(output/'splits.csv'),
        'source_sha256': {**source_hashes(), str(Path(__file__)): file_digest(__file__)}}
    protocol_path = output/'protocol.json'
    development = profiles[profiles.split != 'test'].copy()
    scores = []
    with run_record(protocol_path, protocol):
        # 2. Refit local parameters for each h using train/validation calibration inputs.
        for h in H_VALUES:
            print(f'Development h={h:g} cm', flush=True)
            destination = output/'development'/f'h_{h:g}'
            run_profiles(PreparedProfiles(development, prepared.excluded, prepared.metadata),
                         atmosphere, destination, allocation=InputAllocation(h), max_nfev=max_nfev)
            layers = pd.read_csv(destination/'layers.csv')
            expected = development[['profile_id', 'layer', 'split']]
            layers = expected.merge(layers.drop(columns='split'), on=['profile_id', 'layer'],
                                    how='left', validate='one_to_one')
            for name, frame in layers.groupby('split'):
                scores.append({'h_cm': h, 'split': name, **score(frame)})
            pd.DataFrame(scores).to_csv(output/'validation_scores.csv', index=False)
            print(pd.DataFrame(scores[-2:])[['split','rmse','kge_2012','eligible']].to_string(index=False), flush=True)
        # 3. Select ONLY on validation; write the decision before any test f_new evaluation.
        score_table = pd.DataFrame(scores)
        selected_h = choose_h(score_table, metric)
        decision = {'selected_h_cm': selected_h, 'metric': metric,
                    'selected_utc': datetime.now(timezone.utc).isoformat(),
                    'validation_score': float(score_table[(score_table.h_cm == selected_h)
                                                         & (score_table.split == 'validation')][metric].iloc[0])}
        (output/'selection.json').write_text(json.dumps(decision, indent=2)+'\n')
        print(f'LOCKED: h={selected_h:g} selected by validation {metric}', flush=True)
        plot_search(score_table, selected_h, output)
        # 4. Calibrate test-site parameters with f_new masked. Unmask only for evaluation.
        test = profiles[profiles.split == 'test'].copy()
        test_scores = []
        for h in dict.fromkeys([selected_h, BASELINE_H]):
            destination = output/'test'/f'h_{h:g}'
            print(f'Final test evaluation h={h:g} cm', flush=True)
            run_profiles(PreparedProfiles(test.assign(fnew_obs=np.nan), prepared.excluded, prepared.metadata),
                         atmosphere, destination, allocation=InputAllocation(h), max_nfev=max_nfev)
            labels = test[['profile_id', 'layer', 'fnew_obs']]
            layers = pd.read_csv(destination/'layers.csv').drop(columns='fnew_obs').merge(
                labels, on=['profile_id', 'layer'], validate='one_to_one')
            layers.to_csv(destination/'evaluated_layers.csv', index=False)
            evaluation = destination/'evaluation'
            evaluation.mkdir()
            plot_comparison(layers, evaluation)
            test_scores.append({'h_cm': h, 'selected': h == selected_h, 'baseline': h == BASELINE_H,
                                'split': 'test', **score(layers)})
            pd.DataFrame(test_scores).to_csv(output/'test_scores.csv', index=False)
        print(pd.DataFrame(test_scores).to_string(index=False), flush=True)
        protocol.update(selected_h_cm=selected_h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/layered_h_tuning_seed42'))
    parser.add_argument('--metric', choices=['rmse', 'kge_2012'], default='rmse')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-nfev', type=int, default=1000)
    args = parser.parse_args()
    run_experiment(args.output_dir, metric=args.metric, seed=args.seed, max_nfev=args.max_nfev)
