"""Figure 3: observed vs. predicted F_new for the lognormal model.

Panel A: bulk-soil lognormal model.  Panel B: depth-resolved lognormal model.
Points are sites (A) or layers (B); the dashed line is 1:1. Error bars appear
only where carbon stocks were backfilled from SoilGrids, and span the
predictions made with 5th/95th-percentile stock scenarios.

Inputs (from 03b_lognormal_calibration.py, then
04_collect_continuum_model_predictions.py):
    results/04_model_predictions/lognormal_model_predictions.csv
    results/04_model_predictions/depth_resolved/lognormal_model_predictions.csv

Outputs: figures/fig3[_no_layered_soilgrids].png and a .csv of
per-panel metrics (N, RMSE, KGE, SoilGrids counts, input SHA-256).

Usage: python -m notebooks.fig3 [--exclude-layered-soilgrids]
"""
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


def plot_panel(ax, data, observed_column, source_column, title, color, fontsize=8):
    """Plot one observed-vs-predicted F_new panel and return its fit metrics.

    Only rows where both `observed_column` and `predicted_fnew` are finite are
    plotted and scored. Rows whose `source_column` is 'SoilGrids backfill' get
    a vertical error bar spanning `predicted_fnew` and whichever of
    `predicted_fnew_05`/`predicted_fnew_95` are finite.

    Args:
        ax: matplotlib Axes to draw on.
        data: DataFrame with `observed_column`, `source_column`,
            `predicted_fnew`, `predicted_fnew_05`, `predicted_fnew_95`.
        observed_column: column holding observed F_new (δ13C based).
        source_column: column naming the carbon-stock source for each row.
        title: panel title.
        color: marker face color.
        fontsize: font size of the metrics box.

    Returns:
        dict with N, RMSE, KGE (unclipped, may be -inf/nan), and counts of
        SoilGrids-backfilled rows: total, with error bars, and with both
        stock scenarios available.
    """
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
    # Metrics use all finite pairs, including SoilGrids-backfilled rows.
    metrics = {'N': len(pairs), 'RMSE': float(np.sqrt(np.mean((predicted-observed)**2))),
               'KGE': float(RegressionMetric(observed, predicted).kling_gupta_efficiency(force_finite=False)),
               'soilgrids_N': int(filled.sum()), 'soilgrids_errorbars_N': int(bars.sum()),
               'soilgrids_both_scenarios_N': int((filled & available.eq(2)).sum())}
    palette = color_palette()
    ax.text(.05, .95, f'N = {metrics["N"]}\nRMSE = {metrics["RMSE"]:.2g}\nKGE = {metrics["KGE"]:.2g}',
            transform=ax.transAxes, va='top', fontsize=fontsize,
            bbox={'boxstyle': 'round', 'facecolor': palette['light_yellow'],
                  'edgecolor': palette['dark_grey'], 'alpha': .9})
    ax.set(title=title, xlabel=r'observed F$_{new}$ ($\delta^{13}$C based)',
           xlim=(-.03, 1.03), ylim=(-.03, 1.03), aspect='equal',
           xticks=[0, .5, 1], yticks=[0, .5, 1])
    return metrics


def main(exclude_layered_soilgrids=False, output_stem='fig3'):
    """Build Figure 3 and write the image files and the metrics CSV to figures/.

    Args:
        exclude_layered_soilgrids: drop SoilGrids-backfilled layers from
            panel B only (panel A is unchanged), retitle panel B, and add the
            suffix '_no_layered_soilgrids' to the output names.
        output_stem: base filename for outputs under figures/.
    """
    sources = [ROOT/'results/04_model_predictions/lognormal_model_predictions.csv',
               ROOT/'results/04_model_predictions/depth_resolved/lognormal_model_predictions.csv']
    tables = [pd.read_csv(source) for source in sources]
    if exclude_layered_soilgrids:
        tables[1] = tables[1].loc[tables[1].stock_source.ne('SoilGrids backfill')]
    plt.style.use(ROOT/'notebooks/style.mpl')
    fig, axes = plt.subplots(1, 2, figsize=(4.26, 2.4), sharex=True, sharey=True)
    palette = color_palette()
    layer_title = 'Layered model: Balesdent stocks' if exclude_layered_soilgrids else 'depth-resolved lognormal model'
    settings = [('total_fnew', 'C_data_source', 'bulk soil lognormal model', palette['dark_blue']),
                ('fnew_obs', 'stock_source', layer_title, palette['blue'])]
    metrics = []
    for ax, table, setting, source in zip(axes, tables, settings, sources):
        metrics.append({'panel': setting[2], 'source': str(source.relative_to(ROOT)),
                        'source_sha256': file_digest(source), **plot_panel(ax, table, *setting, fontsize=5)})
    axes[0].set_ylabel(r'predicted F$_{new}$')
    # Panel letters (7 pt, as in fig1). A clears the y label; B has no tick labels.
    axes[0].text(-.21, 1.04, 'A', transform=axes[0].transAxes, fontsize=7)
    axes[1].text(-.12, 1.04, 'B', transform=axes[1].transAxes, fontsize=7)
    #fig.subplots_adjust(left=.1, right=.99, bottom=.14, top=.92, wspace=.12)
    plt.tight_layout() 

    # Save the figure. 
    suffix = '_no_layered_soilgrids' if exclude_layered_soilgrids else ''
    output = ROOT/f'figures/{output_stem}{suffix}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)  # No tight bbox: keep the exact width.
    plt.close(fig)
    
    # source_sha256 ties the figure to the exact prediction files it was made from.
    pd.DataFrame(metrics).to_csv(output.with_suffix('.csv'), index=False)
    print(pd.DataFrame(metrics).drop(columns=['source', 'source_sha256']).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exclude-layered-soilgrids', action='store_true',
                        help='Exclude SoilGrids-filled layers and save a separate figure; keep bulk unchanged.')
    main(**vars(parser.parse_args()))
