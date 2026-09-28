"""Two-panel Figure 4: python -m notebooks.fig4 (after scripts 03b/04)."""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from permetrics.regression import RegressionMetric

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Support direct execution.

from notebooks.viz import color_palette
from soil_diskin.utils import file_digest

ROOT = Path(__file__).resolve().parents[1]


def plot_panel(ax, data, observed_column, source_column, title, color):
    """Score finite observation/prediction pairs; propagate stock scenarios vertically."""
    pairs = data[np.isfinite(data[[observed_column, 'predicted_fnew']]).all(axis=1)]
    observed, predicted = pairs[[observed_column, 'predicted_fnew']].to_numpy().T
    filled = pairs[source_column].eq('SoilGrids backfill')
    scenarios = pairs[['predicted_fnew_05', 'predicted_fnew_95']]
    available = np.isfinite(scenarios).sum(axis=1)
    bars = filled & available.gt(0)
    # Stock quantiles need not map to ordered f_new quantiles. Enclose the central
    # prediction and every available scenario, without reflecting either endpoint.
    envelope = pd.concat([pairs.predicted_fnew, scenarios.where(np.isfinite(scenarios))], axis=1)
    lower, upper = envelope.min(axis=1), envelope.max(axis=1)
    ax.errorbar(pairs.loc[bars, observed_column], pairs.loc[bars, 'predicted_fnew'],
                yerr=np.vstack([(pairs.predicted_fnew-lower)[bars], (upper-pairs.predicted_fnew)[bars]]),
                fmt='none', ecolor='black', elinewidth=.6, capsize=2, zorder=1)
    ax.scatter(observed, predicted, color=color, edgecolor='black', linewidth=.4,
               s=18, alpha=.8, zorder=2)
    ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=1, zorder=0)
    metrics = {'N': len(pairs), 'RMSE': float(np.sqrt(np.mean((predicted-observed)**2))),
               'KGE': float(RegressionMetric(observed, predicted).kling_gupta_efficiency(force_finite=False)),
               'soilgrids_N': int(filled.sum()), 'soilgrids_errorbars_N': int(bars.sum()),
               'soilgrids_both_scenarios_N': int((filled & available.eq(2)).sum())}
    palette = color_palette()
    ax.text(.05, .95, f'N = {metrics["N"]}\nRMSE = {metrics["RMSE"]:.2g}\nKGE = {metrics["KGE"]:.2g}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox={'boxstyle': 'round', 'facecolor': palette['light_yellow'],
                  'edgecolor': palette['dark_grey'], 'alpha': .9})
    ax.set(title=title, xlabel=r'observed F$_{new}$ ($\delta^{13}$C based)',
           xlim=(-.03, 1.03), ylim=(-.03, 1.03), aspect='equal',
           xticks=[0, .5, 1], yticks=[0, .5, 1])
    return metrics


def main(exclude_layered_soilgrids=False, output_stem='fig4'):
    sources = [ROOT/'results/04_model_predictions/lognormal_model_predictions.csv',
               ROOT/'results/04_model_predictions/depth_resolved/lognormal_model_predictions.csv']
    tables = [pd.read_csv(source) for source in sources]
    if exclude_layered_soilgrids:
        tables[1] = tables[1].loc[tables[1].stock_source.ne('SoilGrids backfill')]
    plt.style.use(ROOT/'notebooks/style.mpl')
    plt.rcParams.update({'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.24, 3.7), sharex=True, sharey=True)
    palette = color_palette()
    layer_title = 'Layered model: Balesdent stocks' if exclude_layered_soilgrids else 'depth-resolved lognormal model'
    settings = [('total_fnew', 'C_data_source', 'bulk soil lognormal model', palette['dark_blue']),
                ('fnew_obs', 'stock_source', layer_title, palette['blue'])]
    metrics = []
    for label, ax, table, setting, source in zip('AB', axes, tables, settings, sources):
        metrics.append({'panel': setting[2], 'source': str(source.relative_to(ROOT)),
                        'source_sha256': file_digest(source), **plot_panel(ax, table, *setting)})
        ax.text(-.14, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight='bold')
    axes[0].set_ylabel(r'predicted F$_{new}$')
    fig.subplots_adjust(left=.09, right=.985, bottom=.21, top=.88, wspace=.12)
    suffix = '_no_layered_soilgrids' if exclude_layered_soilgrids else ''
    output = ROOT/f'figures/{output_stem}{suffix}'
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(output.with_suffix('.'+extension), dpi=300, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(output.with_suffix('.csv'), index=False)
    print(pd.DataFrame(metrics).drop(columns=['source', 'source_sha256']).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exclude-layered-soilgrids', action='store_true',
                        help='Exclude SoilGrids-filled layers and save a separate figure; keep bulk unchanged.')
    main(**vars(parser.parse_args()))
