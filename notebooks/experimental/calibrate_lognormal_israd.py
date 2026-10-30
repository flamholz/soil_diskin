"""Calibrate the lognormal model to pseudo-bulk 0-100 cm ISRaD profiles.

Steps:
  1. Read ISRaD layers (soil_diskin.israd, as 01_preprocess_israd.py does) with
     measured radiocarbon and a carbon stock, either the
     reported lyr_soc (g C/cm²) or C % x bulk density x thickness, which is how
     lyr_soc is computed in ISRaD (no coarse-fragment correction). C % is
     lyr_c_org, else lyr_c_tot; bulk density is lyr_bd_samp, else lyr_bd_tot.
     Layers using either fallback are counted per profile (n_fallback_layers).
  2. Pseudo-bulk each profile and sampling year over 0-100 cm of mineral soil:
     total stock, and stock-weighted radiocarbon. Profiles must be covered
     completely by such layers, without overlaps; organic horizons (negative
     depths) are excluded and layers crossing 100 cm are prorated. Profiles
     with stocks above --max-stock are dropped with a warning; e.g. Finstad_2020
     reports lyr_soc ~100x what its C % implies, likely Mg C/ha.
  3. Warn about profiles within --overlap-km of a Balesdent et al. 2018 site.
  4. Fetch NPP from MODIS MOD17A3HGF via Earth Engine at 10 km (soil_diskin.npp, as
     script 02a does), plus the SD across annual images, used as its uncertainty.
     Results are cached.
  5. Turnover = stock / NPP; turnover_q05/q95 use NPP +/- 1.645 SD.
  6. Fit mu, sigma with soil_diskin.calibration.fit_observation and
     lognormal_spec(), as script 03b does, with the atmosphere referenced to
     each profile's sampling year.
  7. Plot calibrated sigma vs. mu with the 03b Balesdent sites for reference.

Radiocarbon is expressed as R = delta14C/1000 + 1, the convention of
data/14C_atm_annot.csv and script 02. ISRaD fraction modern is decay corrected,
Fm = R exp(lambda (year - 1950)), so R = Fm exp(-lambda (year - 1950)) when only Fm
is reported. The atmospheric record ends in 2013, so later profiles are dropped.
As elsewhere in the pipeline, one atmospheric curve is used for both hemispheres.

Outputs (results/experimental/):
    israd_npp.csv: cached NPP mean and SD per location (g C/m²/yr).
    israd_lognormal_calibration.csv: pseudo-bulk profiles with NPP, turnover and fits.
    israd_lognormal_calibration.png: calibrated sigma vs. mu.

Usage:
    python notebooks/experimental/calibrate_lognormal_israd.py \
        [--overlap-km 5] [--max-stock 150] [--n-jobs -1]
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
from soil_diskin.israd import DEPTH, load_layers, pseudo_bulk
from soil_diskin.npp import fetch_modis_npp, initialize
from soil_diskin.radiocarbon_utils import load_atm14c

ISRAD = ROOT/'data/ISRaD/ISRaD_data_flat_layer_v 2.9.9.2025-08-14.csv'
BALESDENT = ROOT/'results/processed_balesdent_2018.csv'
CALIBRATION = ROOT/'results/03_calibrate_models/lognormal_model_optimization_results.csv'
ATM = ROOT/'data/14C_atm_annot.csv'
OUT_DIR = ROOT/'results/experimental'
Z_90 = 1.645  # standard normal 95th percentile


def warn_balesdent_overlap(profiles, max_km, path=BALESDENT):
    """Print a warning for each profile location within max_km of a Balesdent site."""
    sites = pd.read_csv(path)[['Latitude', 'Longitude']].drop_duplicates()
    lat1, lon1 = np.radians(profiles[['latitude', 'longitude']].to_numpy()).T
    lat2, lon2 = np.radians(sites.to_numpy()).T
    # Haversine distance in km between every profile and every Balesdent site.
    a = (np.sin((lat2 - lat1[:, None])/2)**2
         + np.cos(lat1[:, None])*np.cos(lat2)*np.sin((lon2 - lon1[:, None])/2)**2)
    km = 2*6371*np.arcsin(np.sqrt(a))
    near = km.min(axis=1) <= max_km
    for i in np.flatnonzero(near):
        j = km[i].argmin()
        p = profiles.iloc[i]
        print(f'WARNING: {p.entry_name} / {p.site_name} / {p.pro_name} ({p.latitude:.3f}, '
              f'{p.longitude:.3f}) is {km[i, j]:.1f} km from Balesdent site '
              f'({sites.Latitude.iloc[j]:.3f}, {sites.Longitude.iloc[j]:.3f})')
    if not near.any():
        print(f'No ISRaD profile within {max_km} km of a Balesdent site.')
    return near


def fetch_npp(locations):
    """MODIS NPP mean and SD across annual images at 10 km, in g C/m²/yr, fill values masked."""
    initialize()
    return fetch_modis_npp(locations, sd=True)


def site_npp(profiles, cache):
    """NPP for each profile location, fetched once and cached by location."""
    locations = profiles[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    if cache.exists():
        npp = pd.read_csv(cache)
        if len(locations.merge(npp, on=['latitude', 'longitude'])) == len(locations):
            print(f'Using cached NPP from {cache}')
            return profiles.merge(npp, on=['latitude', 'longitude'], how='left')
    npp = fetch_npp(locations)
    cache.parent.mkdir(parents=True, exist_ok=True)
    npp.to_csv(cache, index=False)
    return profiles.merge(npp, on=['latitude', 'longitude'], how='left')


def atmosphere_for_year(year, path=ATM):
    """Atmospheric R with ages counted back from `year`."""
    return load_atm14c(path, reference_year=year)


def calibrate(profiles, n_jobs=-1):
    """Fit mu, sigma to turnover and R, plus the NPP-uncertainty scenarios _05 and _95."""
    atmospheres = {}
    for year in sorted(profiles.lyr_obs_date_y.unique()):
        atmospheres[year] = atmosphere_for_year(year)
        atmospheres[year].log_rate_response()  # Tabulate once here rather than in every worker.
    spec = lognormal_spec()
    out = profiles.copy()
    for column, suffix in [('turnover', ''), ('turnover_q05', '_05'), ('turnover_q95', '_95')]:
        valid = out[column].gt(0) & np.isfinite(out[column])
        fits = Parallel(n_jobs=n_jobs)(
            delayed(fit_observation)(spec, tau, R, atmospheres[year])
            for year, tau, R in out.loc[valid, ['lyr_obs_date_y', column, 'R']].itertuples(index=False))
        out = out.join(pd.DataFrame(fits, index=out.index[valid]).add_suffix(suffix))
    return out


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


def plot(result, reference):
    """Left: calibrated sigma vs. mu colored by turnover; segments span the NPP-uncertainty fits.
    Right: the observed T and 14C/12C that were fitted, colored by calibrated sigma, with the
    Balesdent sites in grey.
    Only exact fits are shown; approximate or failed fits are left out."""
    plt.style.use(ROOT/'notebooks/style.mpl')
    fig, (ax, ax_obs) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    result = result[result.calibration_status.eq('calibrated')]
    norm = LogNorm(result.turnover.min(), result.turnover.max())
    ax.scatter(reference.mu, reference.sigma, s=8, color='lightgrey', label='Balesdent sites (03b)')
    ax.plot(np.vstack([result.mu_05, result.mu_95]), np.vstack([result.sigma_05, result.sigma_95]),
            color='C0', alpha=.3, linewidth=.5, zorder=1)
    points = ax.scatter(result.mu, result.sigma, s=10, c=result.turnover, cmap='viridis',
                        norm=norm, edgecolor='black', linewidth=.3, zorder=2,
                        label='ISRaD pseudo-bulk 0-100 cm')
    fig.colorbar(points, ax=ax, label='turnover T (yr)')
    plot_observations(ax_obs, [
        (reference.turnover, reference.fm_obs, dict(s=8, color='lightgrey'), 'grey'),
        (result.turnover, result.R, dict(c=result.sigma, cmap='plasma', s=10,
                                         edgecolor='black', linewidth=.3), 'C0')],
        colorbar_label=r'calibrated $\sigma$')
    plot_iso_turnover(ax, norm)
    ax.set(xlabel=r'$\mu$', ylabel=r'$\sigma$', title=f'N = {len(result)} calibrated profiles')
    ax.legend(frameon=True)
    fig.tight_layout()
    return fig


def main(overlap_km=5., max_stock=150., n_jobs=-1, output_dir=OUT_DIR):
    output_dir = Path(output_dir)
    profiles = pseudo_bulk(load_layers(ISRAD)).rename(columns={'stock_kg_m2': 'stock'})
    implausible = profiles.stock > max_stock
    for entry, n in profiles[implausible].entry_name.value_counts().items():
        print(f'WARNING: dropping {n} {entry} profile-years with stock > {max_stock:g} kg C/m² '
              f'(median {profiles.stock[implausible & profiles.entry_name.eq(entry)].median():.0f}); '
              'check lyr_soc units')
    profiles = profiles[~implausible]
    print(f'{len(profiles)} profile-years completely covered over 0-{DEPTH:g} cm, '
          f'{len(profiles[["latitude", "longitude"]].drop_duplicates())} locations')
    warn_balesdent_overlap(profiles, overlap_km)

    last_year = pd.read_csv(ATM).year.max()
    late = profiles.lyr_obs_date_y > last_year
    print(f'Dropping {late.sum()} profile-years sampled after {last_year}, the end of the atmospheric record')
    profiles = site_npp(profiles[~late].reset_index(drop=True), output_dir/'israd_npp.csv')

    npp = profiles.NPP/1000  # g C/m²/yr -> kg C/m²/yr
    profiles['turnover'] = profiles.stock/npp
    profiles['turnover_q05'] = profiles.stock/(npp + Z_90*profiles.NPP_sd/1000)
    profiles['turnover_q95'] = profiles.stock/(npp - Z_90*profiles.NPP_sd/1000).where(lambda x: x > 0)
    profiles = profiles[profiles.turnover.gt(0) & np.isfinite(profiles.turnover)]
    result = calibrate(profiles.reset_index(drop=True), n_jobs)

    csv, png = (output_dir/f'israd_lognormal_calibration.{ext}' for ext in ['csv', 'png'])
    result.to_csv(csv, index=False)
    reference = pd.read_csv(CALIBRATION)
    fig = plot(result, reference)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f'Saved {csv} and {png}')
    print(result.calibration_status.value_counts().to_dict())
    print(result[['stock', 'NPP', 'turnover', 'R', 'mu', 'sigma']].describe().loc[['min', '50%', 'max']].round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--overlap-km', type=float, default=5.,
                        help='warn about profiles within this distance of a Balesdent site')
    parser.add_argument('--max-stock', type=float, default=150.,
                        help='drop profiles whose 0-100 cm stock exceeds this (kg C/m²)')
    parser.add_argument('--n-jobs', type=int, default=-1, help='parallel jobs for joblib (-1 uses all cores)')
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR)
    main(**vars(parser.parse_args()))
