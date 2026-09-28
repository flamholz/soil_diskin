"""Figure S7: compare five layered input allocations at 100% and 50% soil NPP.

Run: python -m notebooks.figS7 --n-jobs 4
Read the saved Fig. 4B table, refit alternatives in memory, and save one figure.
Mean and SoilGrids stock scenarios use the same fitter and predictions as Fig. 4.
"""
import argparse
from importlib import import_module
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notebooks.fig4 import ROOT, plot_panel
from notebooks.viz import color_palette
from soil_diskin.continuum_models import LognormalDisKin
from soil_diskin.lognormal import cached_radiocarbon
from soil_diskin.radiocarbon_utils import load_atm14c

fit_observation = import_module('notebooks.03b_lognormal_calibration').fit_observation
generate_predictions = import_module('notebooks.04_collect_continuum_model_predictions').generate_predictions
InputAllocation = import_module('notebooks.02_get_turnover_14C').InputAllocation
INPUT = ROOT/'results/04_model_predictions/depth_resolved/lognormal_model_predictions.csv'
OUTPUT = ROOT/'figures/figS7.png'
BASELINE_SCHEME = 'jackson_global_surface'
# Jackson (1996), doi:10.1007/BF00333714: Fig. 5 groups; crops from Table 1.
JACKSON_BETA = {'global': .966, 'crop': .961, 'grass': .952, 'tree': .970, 'shrub': .978}
SCHEMES = {
    'jackson_global_surface': '50% surface +\n50% Jackson global',
    'exponential_h10': 'Exponential h = 10 cm',
    'jackson_global': 'Jackson global',
    'jackson_vegetation': 'Jackson vegetation',
    'jackson_vegetation_surface': '50% surface +\n50% Jackson vegetation',
}


def layer_inputs(sites, scheme, fraction):
    """Integrate NPP over actual intervals, normalized over the whole 0–100 cm."""
    depth = pd.Series(10., index=sites.index)
    if scheme != 'exponential_h10':
        group = pd.Series('global', index=sites.index)
        if 'vegetation' in scheme:
            land = sites.land_use.astype('string').str.strip().str.upper().fillna('')
            vegetation = sites.vegetation.astype('string').str.lower().fillna('')
            group = land.map({'CROP': 'crop', 'GRASSLAND': 'grass', 'FOREST': 'tree'}).fillna('global')
            group.loc[land.ne('CROP') & vegetation.str.contains('sylvopastoral|savanna|trifolium')] = 'global'
            group.loc[land.ne('CROP') & vegetation.str.contains(r'\bshrub\b')] = 'shrub'
        depth = -1/np.log(group.map(JACKSON_BETA))
    inputs = pd.Series(np.nan, index=sites.index)
    for h, indices in depth.groupby(depth).groups.items():
        rows = sites.loc[indices]
        allocation = InputAllocation(h, .5 if scheme.endswith('_surface') else 0., fraction)
        inputs.loc[indices] = allocation.layer_input(rows.npp_kg_m2_yr, rows.z_top_cm, rows.z_bottom_cm)
    return inputs


def refit(sites, inputs, atmosphere, fm_evaluator, n_jobs):
    """Refit mean/q05/q95 stocks independently; observed f_new never enters fitting."""
    params = sites[['fnew_obs', 'stock_source', 'Duration_labeling']].copy()
    for stock_suffix, suffix in [('', ''), ('_q05', '_05'), ('_q95', '_95')]:
        columns = ['mu'+suffix, 'sigma'+suffix]
        params[columns] = np.nan
        turnover = sites.get('stock_kg_m2'+stock_suffix, np.nan)/inputs
        valid = np.isfinite(turnover) & turnover.gt(0) & np.isfinite(sites.fm) & sites.fm.gt(0)
        if valid.any():
            fits = Parallel(n_jobs=n_jobs)(delayed(fit_observation)(atmosphere, fm_evaluator, tau, fm)
                                          for tau, fm in zip(turnover[valid], sites.fm[valid]))
            params.loc[valid, columns] = pd.DataFrame(fits)[['mu', 'sigma']].to_numpy()
    return generate_predictions(LognormalDisKin, params, ['mu', 'sigma'], params)


def plot_comparison(tables, exclude_layered_soilgrids=False):
    baseline = tables[(1., BASELINE_SCHEME)]
    reference = np.isfinite(baseline[['fnew_obs', 'predicted_fnew']]).all(axis=1)
    if exclude_layered_soilgrids:
        reference &= baseline.stock_source.ne('SoilGrids backfill')
    common = reference.copy()
    for table in tables.values():
        common &= np.isfinite(table.predicted_fnew)
    if common.sum() < 2:
        raise ValueError('Need at least two common finite observation/prediction pairs')
    print(f'Comparing {common.sum()} of {reference.sum()} Fig. 4B pairs in every panel.', flush=True)
    with plt.style.context(ROOT/'notebooks/style.mpl'):
        plt.rcParams.update({'axes.titlesize': 10, 'axes.labelsize': 9,
                             'xtick.labelsize': 8, 'ytick.labelsize': 8})
        fig, axes = plt.subplots(2, 5, figsize=(15, 6.7), sharex=True, sharey=True)
        for row, fraction in enumerate([1., .5]):
            for col, (scheme, title) in enumerate(SCHEMES.items()):
                if fraction == 1. and scheme == BASELINE_SCHEME:
                    title += '\nFig. 4B baseline'
                ax = axes[row, col]
                plot_panel(ax, tables[(fraction, scheme)].loc[common], 'fnew_obs', 'stock_source',
                           title, color_palette()['blue'])
                if col == 0:
                    ax.set_ylabel(f'{fraction:.0%} of NPP enters soil\n'+r'predicted F$_{new}$')
        fig.subplots_adjust(left=.07, right=.99, bottom=.15, top=.9, wspace=.13, hspace=.47)
    return fig


def main(input=INPUT, output=OUTPUT, n_jobs=4, exclude_layered_soilgrids=False):
    sites = pd.read_csv(input, dtype={'profile_id': str}).sort_values(['profile_id', 'layer']).reset_index(drop=True)
    np.testing.assert_allclose(sites.input_kg_m2_yr, layer_inputs(sites, BASELINE_SCHEME, 1.), equal_nan=True)
    atmosphere = load_atm14c(ROOT/'data/14C_atm_annot.csv')
    fm_evaluator = cached_radiocarbon(atmosphere)
    tables = {(1., BASELINE_SCHEME): sites}
    for fraction in [1., .5]:
        for scheme in SCHEMES:
            if (fraction, scheme) not in tables:
                print(f'{fraction:.0%} soil NPP: {scheme}', flush=True)
                tables[(fraction, scheme)] = refit(sites, layer_inputs(sites, scheme, fraction),
                                                   atmosphere, fm_evaluator, n_jobs)
    fig = plot_comparison(tables, exclude_layered_soilgrids)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT, help='Saved Fig. 4B predictions, including their input data.')
    parser.add_argument('--output', type=Path, default=OUTPUT, help='Figure filename; its extension sets the format.')
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--exclude-layered-soilgrids', action='store_true')
    main(**vars(parser.parse_args()))
