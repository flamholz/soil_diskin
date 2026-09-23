"""Compare fixed Jackson et al. (1996) root-depth coefficients with h=10 cm.

Run from the repo root: uv run python -m notebooks.compare_jackson_inputs
The paper's beta parameter is exactly equivalent to h=-1/log(beta), so the
existing model is reused. No coefficients or vegetation choices fit f_new.
Add --surface-fraction 0.5 to compare mixtures with a direct top-layer input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.tune_layered_input_depth import score
from soil_diskin.layered_data import PreparedProfiles, file_digest, load_profiles
from soil_diskin.layered_lognormal import input_weights
from soil_diskin.layered_workflow import run_profiles
from soil_diskin.radiocarbon_utils import load_atm14c

PAPER = 'https://doi.org/10.1007/BF00333714'
# Global fit and Fig. 5 functional groups; crops are from Table 1.
JACKSON_BETA = {'global': .966, 'crop': .961, 'grass': .952, 'tree': .970, 'shrub': .978}
LABELS = {'exponential_h10': 'Exponential: h = 10 cm',
          'jackson_global': 'Jackson: global beta = 0.966',
          'jackson_vegetation': 'Jackson: vegetation-dependent beta'}


def jackson_assignments(raw: pd.DataFrame) -> pd.DataFrame:
    """Coarse metadata-based mapping, with global fallback for ambiguous stands.

    Forests in this dataset are temperate/tropical, matching the paper's tree
    group. This is not a map of all eleven biomes or an inferred root mixture.
    """
    result = raw[['Internal_profile_ID', 'Land_Use', 'Vegetation']].rename(columns={
        'Internal_profile_ID': 'profile_id', 'Land_Use': 'land_use', 'Vegetation': 'vegetation'}).copy()
    if result.profile_id.isna().any() or result.profile_id.duplicated().any():
        raise ValueError('unique nonmissing profile identities required')
    land = result.land_use.astype('string').str.strip().str.upper().fillna('')
    vegetation = result.vegetation.astype('string').str.lower().fillna('')
    group = land.map({'CROP': 'crop', 'GRASSLAND': 'grass', 'FOREST': 'tree'}).fillna('global')
    group.loc[land.ne('CROP') & vegetation.str.contains('sylvopastoral|savanna|trifolium')] = 'global'
    group.loc[land.ne('CROP') & vegetation.str.contains(r'\bshrub\b')] = 'shrub'
    result['jackson_group'] = group
    result['global_fallback'] = group.eq('global')
    result['jackson_beta'] = group.map(JACKSON_BETA)
    result['input_depth_cm'] = -1/np.log(result.jackson_beta)
    return result


def plot_results(layers: dict[str, pd.DataFrame], metrics: pd.DataFrame, output: Path,
                 surface_fraction: float = 0.) -> None:
    import matplotlib.pyplot as plt

    rows = 2 if len(layers) > 3 else 1
    fig, axes = plt.subplots(rows, 3, figsize=(13, 4.5*rows+.6), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    for ax in axes[len(layers):]:
        ax.set_visible(False)
    for ax, (name, frame), color in zip(axes, layers.items(), ['#2476b8', '#d66b22', '#298568', '#9a5aaf', '#ad8730']):
        values = metrics.set_index('scheme').loc[name]
        title = LABELS[name.removesuffix('_surface')]
        if name.endswith('_surface'):
            title = title.replace('Jackson:', f'{surface_fraction:.0%} surface + {1-surface_fraction:.0%} Jackson:\n')
        ax.scatter(frame.fnew_obs, frame.fnew_pred, s=15, alpha=.45, color=color, edgecolor='none')
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), aspect='equal',
               xlabel='Observed new-carbon fraction', title=title)
        ax.text(.04, .96, f'RMSE = {values.rmse:.4f}\nKGE (2012) = {values.kge_2012:.3f}',
                transform=ax.transAxes, va='top', bbox={'facecolor': 'white', 'edgecolor': 'lightgray'})
        ax.grid(alpha=.15)
    for ax in axes[::3]:
        ax.set_ylabel('Predicted new-carbon fraction')
    frame = next(iter(layers.values()))
    fig.suptitle(f'Same {frame.profile_id.nunique()} profiles and {len(frame)} layer observations', y=.98)
    fig.text(.5, .035, 'Fixed published coefficients; root biomass used as a proxy for input depth. '
             'All NPP allocated within 0–100 cm.', ha='center', fontsize=9, color='#50565c')
    fig.subplots_adjust(left=.06, right=.985, top=.89 if rows > 1 else .82,
                        bottom=.10 if rows > 1 else .18, wspace=.13, hspace=.35)
    for extension in ['png', 'pdf']:
        fig.savefig(output/f'comparison.{extension}', dpi=200)
    plt.close(fig)

    fractions = [0., surface_fraction] if surface_fraction else [0.]
    fig, axes = plt.subplots(1, len(fractions), figsize=(6.7*len(fractions), 5.2), layout='constrained')
    weight_rows = []
    depths = [('h10_baseline', 10.)] + [(group, float(-1/np.log(beta))) for group, beta in JACKSON_BETA.items()]
    for ax, fraction in zip(np.atleast_1d(axes), fractions):
        for group, depth in depths:
            direct = 0. if group == 'h10_baseline' else fraction
            weights = input_weights(depth, surface_fraction=direct)
            label = 'Exponential h=10 cm' if group == 'h10_baseline' else f'{group.capitalize()}: beta={JACKSON_BETA[group]:.3f}'
            ax.plot(weights, np.arange(5., 100., 10.), 'o-', label=label, ms=4)
            for layer, weight in enumerate(weights):
                weight_rows.append({'group': group, 'input_depth_cm': depth, 'surface_fraction': direct, 'layer': layer,
                                    'z_top_cm': layer*10, 'z_bottom_cm': (layer+1)*10, 'npp_fraction': weight})
        title = f'{fraction:.0%} surface + {1-fraction:.0%} Jackson' if fraction else 'Jackson only'
        ax.set(xlabel='Fraction of site NPP per 10 cm layer', ylabel='Depth (cm)',
               title=title+' (0–100 cm)', ylim=(100, 0), xlim=(0, 1))
        ax.grid(alpha=.15)
        ax.legend()
    for extension in ['png', 'pdf']:
        fig.savefig(output/f'input_profiles.{extension}', dpi=200)
    plt.close(fig)
    pd.DataFrame(weight_rows).drop_duplicates().to_csv(output/'input_weights.csv', index=False)


def run_comparison(output: Path, max_nfev: int = 1000, surface_fraction: float = 0.) -> None:
    input_weights(30., surface_fraction=surface_fraction)  # Validate before loading data or creating outputs.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a new or empty output directory')
    # 1. Use the same eligible layers for every allocation, including partial profiles.
    prepared = load_profiles(allow_partial=True)
    atmosphere = load_atm14c()
    prepared.metadata['atmosphere'] = {'path': str(Path('data/14C_atm_annot.csv').resolve()),
                                      'sha256': file_digest('data/14C_atm_annot.csv')}
    raw = pd.read_excel(prepared.metadata['sources']['balesdent']['path'], sheet_name='Profiles', skiprows=7)
    assignments = jackson_assignments(raw)
    assignments = assignments[assignments.profile_id.isin(prepared.profiles.profile_id)].copy()
    if not np.isfinite(prepared.profiles.fnew_obs).all() or not prepared.profiles.duration_years.gt(0).all():
        raise ValueError('comparison requires observed f_new and positive labeling durations for every layer')
    output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output/'vegetation_assignments.csv', index=False)
    # 2. Freeze the published coefficients and our mapping before fitting/evaluation.
    protocol = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'paper': PAPER, 'jackson_beta': JACKSON_BETA, 'baseline_h_cm': 10.,
        'conversion': 'h_cm = -1 / log(beta); beta exponent uses depth in cm',
        'input_assumption': 'root biomass profile represents the Jackson component; any direct surface fraction is added separately',
        'surface_fraction': surface_fraction,
        'surface_assumption': 'mixtures add direct input to 0–10 cm; remaining Jackson share spans 0–100 cm including top layer',
        'normalization': '0–100 cm; never renormalize to available layers',
        'mapping': 'crop/grass/tree by land use; explicit shrub override; global for sylvopastoral, savanna, Trifolium, unknown',
        'fnew_used_for_coefficients_or_mapping': False, 'evaluation': 'pooled descriptive comparison, not independent test',
        'max_nfev_per_start': max_nfev, 'data': prepared.metadata,
        'assignments_sha256': file_digest(output/'vegetation_assignments.csv'),
        'source_sha256': {str(path): file_digest(path) for path in [Path(__file__),
            Path('notebooks/tune_layered_input_depth.py'),
            Path('soil_diskin/layered_data.py'), Path('soil_diskin/layered_lognormal.py'), Path('soil_diskin/layered_workflow.py')]}}
    protocol_path = output/'protocol.json'
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    scenarios = {
        'exponential_h10': assignments.assign(input_depth_cm=10., jackson_group='baseline',
                                                jackson_beta=np.nan, global_fallback=False),
        'jackson_global': assignments.assign(input_depth_cm=-1/np.log(JACKSON_BETA['global']),
                                              jackson_group='global', jackson_beta=JACKSON_BETA['global'],
                                              global_fallback=False),
        'jackson_vegetation': assignments}
    scenarios = {name: allocation.assign(surface_fraction=0.) for name, allocation in scenarios.items()}
    if surface_fraction:
        for name in ['jackson_global', 'jackson_vegetation']:
            scenarios[name+'_surface'] = scenarios[name].assign(surface_fraction=surface_fraction)
    all_layers, summaries = {}, []
    try:
        # 3. Reuse the unchanged fitter for each coefficient group; refit local mu/sigma.
        for name, allocation in scenarios.items():
            frames = []
            data = prepared.profiles.merge(allocation, on='profile_id', validate='many_to_one')
            for group, subset in data.groupby('jackson_group', sort=False):
                depth = float(subset.input_depth_cm.iloc[0])
                direct = float(subset.surface_fraction.iloc[0])
                destination = output/name/str(group)
                print(f'{name}: {group}, h={depth:.6f} cm, surface={direct:.0%}, {len(subset)} layers', flush=True)
                metadata = {**prepared.metadata, 'input_scheme': name, 'jackson_group': group}
                run_profiles(PreparedProfiles(subset, prepared.excluded, metadata), atmosphere,
                             destination, input_depth=depth, surface_fraction=direct, max_nfev=max_nfev)
                frames.append(pd.read_csv(destination/'layers.csv', float_precision='round_trip'))
            fitted = pd.concat(frames, ignore_index=True).sort_values(['profile_id', 'layer'])
            expected = prepared.profiles[['profile_id', 'layer']].sort_values(['profile_id', 'layer'])
            pd.testing.assert_frame_equal(fitted[['profile_id', 'layer']].reset_index(drop=True), expected.reset_index(drop=True))
            fitted.to_csv(output/name/'layers.csv', index=False)
            summaries.append({'scheme': name, **score(fitted)})
            all_layers[name] = fitted
            pd.DataFrame(summaries).to_csv(output/'metrics.csv', index=False)
        # 4. Report every scheme and calibration diagnostics; do not select on these outcomes.
        table = pd.DataFrame(summaries)
        plot_results(all_layers, table, output, surface_fraction)
        print(table.to_string(index=False), flush=True)
        protocol['status'] = 'complete'
    finally:
        if protocol['status'] != 'complete':
            protocol['status'] = 'interrupted_or_failed'
        protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/layered_jackson_1996'))
    parser.add_argument('--max-nfev', type=int, default=1000)
    parser.add_argument('--surface-fraction', type=float, default=0.,
                        help='also compare this direct top-layer NPP fraction plus Jackson roots over 0–100 cm')
    args = parser.parse_args()
    if args.max_nfev < 1:
        parser.error('--max-nfev must be positive')
    run_comparison(args.output_dir, args.max_nfev, args.surface_fraction)
