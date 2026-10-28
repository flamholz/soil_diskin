"""Recalibrate the lognormal model on simulated sites and plot sigma against mu.

Reads the output of simulate_lognormal_input_variability.py and treats each
simulated site as an observation, as script 02 does for real sites:
    turnover = simulated stock / mean simulated NPP
    fm       = simulated Fm in 2000
Each observation is fitted with soil_diskin.calibration.fit_observation and
lognormal_spec(), as script 03b does, so starts, bounds and tolerances match the
real calibration.

Outputs:
    <input stem>_calibration.csv: simulated sites with fitted mu, sigma and fit status.
    <input stem>_calibration.png: calibrated sigma vs. mu, with the true values
        used in the simulation for reference.
    <input stem>_calibration_observed_range.png: calibrated sigma vs. mu for simulated
        sites whose true mu and sigma lie within the ranges calibrated for the
        Balesdent sites (03b), which are shown in grey.

Usage:
    python notebooks/experimental/calibrate_lognormal_input_variability.py \
        [-i results/experimental/lognormal_input_variability.csv] [--n-jobs -1]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # Support direct execution.

from soil_diskin.calibration import fit_observation, lognormal_spec
from soil_diskin.radiocarbon_utils import load_atm14c

INPUT = ROOT/'results/experimental/lognormal_input_variability.csv'
CALIBRATION = ROOT/'results/03_calibrate_models/lognormal_model_optimization_results.csv'


def calibrate(sims, n_jobs=-1):
    """Fit mu, sigma to each simulated site's turnover and Fm."""
    atm = load_atm14c(ROOT/'data/14C_atm_annot.csv')
    atm.log_rate_response()  # Tabulate once here rather than in every worker.
    spec = lognormal_spec()
    observed = sims.assign(turnover=sims.stock/(sims.NPP_mean_simulated/1000))
    fits = Parallel(n_jobs=n_jobs)(
        delayed(fit_observation)(spec, tau, fm, atm)
        for tau, fm in observed[['turnover', 'fm']].itertuples(index=False, name=None))
    fitted = pd.DataFrame(fits, index=observed.index)
    return observed.rename(columns={'mu': 'mu_true', 'sigma': 'sigma_true'}).join(fitted)


def plot_iso_turnover(ax, norm, cmap='viridis'):
    """Dashed curves of constant T = exp(sigma^2/2 - mu), i.e. mu = sigma^2/2 - ln T.

    Levels are decades, or 1-3-10 steps when norm spans at most two decades; each
    curve takes the color of its T under norm and is labeled with T (yr) where it leaves the axes.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    lo, hi = np.log10(norm.vmin), np.log10(norm.vmax)
    multiples = [1, 3] if hi - lo <= 2 else [1]
    levels = [m*10.**e for e in range(int(np.floor(lo)), int(np.ceil(hi)) + 1) for m in multiples]
    sigma = np.linspace(*ylim, 200)
    for T in levels:
        if not norm.vmin <= T <= norm.vmax:
            continue
        color = plt.get_cmap(cmap)(norm(T))
        mu = sigma**2/2 - np.log(T)
        ax.plot(mu, sigma, color=color, linestyle='--', linewidth=.6, zorder=0)
        if xlim[0] <= mu[-1] <= xlim[1]:  # Leaves through the top edge.
            ax.text(mu[-1], ylim[1], f'{T:g} ', color=color, fontsize=5, ha='right', va='top')
        elif mu[-1] > xlim[1]:  # Leaves through the right edge, at sigma^2/2 = mu_max + ln T.
            ax.text(xlim[1], np.sqrt(2*(xlim[1] + np.log(T))), f'{T:g} ', color=color,
                    fontsize=5, ha='right', va='bottom')
    ax.set(xlim=xlim, ylim=ylim)


def plot_observations(ax, groups, colorbar_label=None):
    """Scatter T vs. 14C/12C with marginal histograms on top and right, as in sns.jointplot.

    Args:
        groups: (T, R, scatter_kwargs, marginal_color) per group of points.
        colorbar_label: if given, add a colorbar for the last group's colors, right of the margin.
    """
    divider = make_axes_locatable(ax)
    top = divider.append_axes('top', size='22%', pad=.05, sharex=ax)
    right = divider.append_axes('right', size='22%', pad=.05, sharey=ax)
    ax.set_xscale('log')
    for T, R, kwargs, color in groups:
        points = ax.scatter(T, R, **kwargs)
        sns.histplot(x=T, ax=top, color=color, stat='density', log_scale=True, alpha=.5, linewidth=.3)
        sns.histplot(y=R, ax=right, color=color, stat='density', alpha=.5, linewidth=.3)
    for margin, keep in [(top, 'bottom'), (right, 'left')]:
        margin.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        margin.set(xlabel='', ylabel='')
        for side, spine in margin.spines.items():
            spine.set_visible(side == keep)
    ax.set(xlabel='turnover T (yr)', ylabel=r'$^{14}$C/$^{12}$C')
    if colorbar_label:
        ax.figure.colorbar(points, cax=divider.append_axes('right', size='5%', pad=.3),
                           label=colorbar_label)


def plot_parameters(ax, result, norm):
    """Calibrated sigma vs. mu colored by turnover, lines to the true values, and constant-T curves."""
    ax.plot(np.vstack([result.mu_true, result.mu]), np.vstack([result.sigma_true, result.sigma]),
            color='lightgrey', linewidth=.5, zorder=0)
    ax.scatter(result.mu_true, result.sigma_true, facecolor='none', edgecolor='grey',
               linewidth=.5, s=12, label='true (simulated)')
    points = ax.scatter(result.mu, result.sigma, c=result.turnover, cmap='viridis',
                        norm=norm, edgecolor='black', linewidth=.3, s=12, label='calibrated')
    plot_iso_turnover(ax, norm)
    ax.set(xlabel=r'$\mu$', ylabel=r'$\sigma$')
    return points


def plot(result):
    """Left: calibrated sigma vs. mu colored by turnover, with lines to the true values used in
    the simulation. Right: the observed T and 14C/12C that were fitted, colored by the true
    sigma used in the simulation. Only exact fits are shown; approximate or failed fits are
    left out (main prints them)."""
    plt.style.use(ROOT/'notebooks/style.mpl')
    fig, (ax, ax_obs) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    n_total = len(result)
    result = result[result.calibration_status.eq('calibrated')]
    norm = LogNorm(result.turnover.min(), result.turnover.max())
    points = plot_parameters(ax, result, norm)
    fig.colorbar(points, ax=ax, label='turnover T (yr)')
    plot_observations(ax_obs, [(result.turnover, result.fm,
                                dict(c=result.sigma_true, cmap='plasma', s=12,
                                     edgecolor='black', linewidth=.3), 'C0')],
                      colorbar_label=r'true $\sigma$ (simulated)')
    ax.set(title=f'cv = {result.cv.iloc[0]:g}, N = {len(result)} of {n_total} calibrated')
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


def in_observed_range(result, reference):
    """Rows whose true mu and sigma both lie within the calibrated ranges of reference."""
    inside = pd.Series(True, index=result.index)
    for name in ['mu', 'sigma']:
        inside &= result[name+'_true'].between(reference[name].min(), reference[name].max())
    return inside


def plot_observed_range(result, reference):
    """Like the left panel of `plot`, for exact fits whose true mu and sigma lie within the
    calibrated ranges of reference (dotted box); reference sites in grey."""
    plt.style.use(ROOT/'notebooks/style.mpl')
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    sims = result[result.calibration_status.eq('calibrated') & in_observed_range(result, reference)]
    norm = LogNorm(sims.turnover.min(), sims.turnover.max())
    ax.scatter(reference.mu, reference.sigma, s=8, color='lightgrey', zorder=0,
               label='Balesdent sites (03b)')
    (mu_lo, mu_hi), (sigma_lo, sigma_hi) = [(reference[n].min(), reference[n].max()) for n in ['mu', 'sigma']]
    ax.add_patch(plt.Rectangle((mu_lo, sigma_lo), mu_hi - mu_lo, sigma_hi - sigma_lo, fill=False,
                               edgecolor='grey', linestyle=':', linewidth=.6, zorder=0))
    points = plot_parameters(ax, sims, norm)
    fig.colorbar(points, ax=ax, label='turnover T (yr)')
    ax.set(title=rf'true $\mu$, $\sigma$ within calibrated ranges: N = {len(sims)} of {len(result)}')
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


def main(input=INPUT, n_jobs=-1):
    input = Path(input)
    result = calibrate(pd.read_csv(input), n_jobs)
    # Build names directly; with_suffix would cut stems that contain dots, e.g. cv0.05.
    csv, png = (input.with_name(f'{input.stem}_calibration.{ext}') for ext in ['csv', 'png'])
    result.to_csv(csv, index=False)
    fig = plot(result)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    # Also plot the calibrated sigma vs. mu for simulated sites whose true mu and sigma lie within
    # the ranges calibrated for the Balesdent sites (03b), which are shown in grey.
    # This is a more direct comparison to the real calibration, which was done on those sites.
    reference = pd.read_csv(CALIBRATION)
    reference = reference[reference.calibration_status.eq('calibrated')]
    png_range = input.with_name(f'{input.stem}_calibration_observed_range.png')
    fig = plot_observed_range(result, reference)
    fig.savefig(png_range, dpi=300)
    plt.close(fig)

    print(f'Saved {csv}, {png} and {png_range}')
    print(f'{in_observed_range(result, reference).sum()} of {len(result)} simulated sites have true '
          'mu and sigma within the calibrated ranges')
    print(result.calibration_status.value_counts().to_dict())
    failed = result[result.calibration_status.ne('calibrated')]
    if len(failed):
        print('Not plotted (approximate or failed fits):')
        print(failed[['turnover', 'fm', 'mu_true', 'sigma_true', 'calibration_status']]
              .rename(columns={'turnover': 'T', 'fm': '14C/12C'}).to_string(float_format='%.4g'))
    for name in ['mu', 'sigma']:
        error = result[name] - result[name+'_true']
        print(f'{name} - true: mean {error.mean():.3g}, SD {error.std():.3g}, max |.| {error.abs().max():.3g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', type=Path, default=INPUT,
                        help='output of simulate_lognormal_input_variability.py')
    parser.add_argument('--n-jobs', type=int, default=-1, help='parallel jobs for joblib (-1 uses all cores)')
    main(**vars(parser.parse_args()))
