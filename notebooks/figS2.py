# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import viz

# %%
plt.style.use('notebooks/style.mpl')
pal = viz.color_palette()

WORLDCLIM_LABELS = {
    'mean_annual_temp':           'MAT',
    'mean_diurnal_temp_range':    'Diurnal ΔT',
    'isothermality':              'Isothermality',
    'temp_seasonality':           'T seasonality',
    'max_temp_warmest_month':     'Max T (warm mo.)',
    'min_temp_coldest_month':     'Min T (cold mo.)',
    'temp_annual_range':          'T annual range',
    'mean_temp_wettest_quarter':  'T (wet qtr.)',
    'mean_temp_driest_quarter':   'T (dry qtr.)',
    'mean_temp_warmest_quarter':  'T (warm qtr.)',
    'mean_temp_coldest_quarter':  'T (cold qtr.)',
    'annual_precip':              'MAP',
    'precip_wettest_month':       'P (wet mo.)',
    'precip_driest_month':        'P (dry mo.)',
    'precip_seasonality':         'P seasonality',
    'precip_wettest_quarter':     'P (wet qtr.)',
    'precip_driest_quarter':      'P (dry qtr.)',
    'precip_warmest_quarter':     'P (warm qtr.)',
    'precip_coldest_quarter':     'P (cold qtr.)',
}

powerlaw_params = pd.read_csv('results/03_calibrate_models/powerlaw_model_optimization_results.csv')
gen_powerlaw_params = pd.read_csv('results/03_calibrate_models/general_powerlaw_model_optimization_results.csv')
gen_powerlaw_params_beta_half = pd.read_csv('results/03_calibrate_models/general_powerlaw_model_optimization_results_beta_half.csv')
lognormal_raw = pd.read_csv('results/03_calibrate_models/03b_lognormal_predictions_calcurve_python.csv')

# Derive mu and sigma from fitted mean age and turnover time
lognormal_raw['mu'] = -np.log(np.sqrt(lognormal_raw['turnover']**3 / lognormal_raw['pred']))
lognormal_raw['sigma'] = np.sqrt(np.log(lognormal_raw['pred'] / lognormal_raw['turnover']))

# %%
def scatter(ax, x, y, color, xlabel, ylabel, xlog=False, ylog=False):
    ax.scatter(x, y, color=color, s=8, linewidths=0.5, edgecolors='k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


# %%
mosaic = [['A', 'B'], ['C', 'D']]
fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(4.76, 4.5), dpi=300, constrained_layout=True)

# --- Row 1: parameter–parameter scatter per model ---
row1_panels = [
    ('A', 'lognormal model',                        lognormal_raw,                pal['dark_blue'],  'mu',    'sigma',  '$\\mu$',    '$\\sigma$',   False, False),
    ('B', 'power law ($\\alpha=1$)',                 powerlaw_params,              pal['blue'],       't_min', 't_max',  '$\\tau_\\text{min}$', '$\\tau_\\text{max}$',   True,  True),
    ('C', 'power law ($\\alpha=e^{-\\gamma}$)',      gen_powerlaw_params,          pal['light_blue'], 't_min', 't_max',  '$\\tau_\\text{min}$', '$\\tau_\\text{max}$',   True,  True),
    ('D', 'power law ($\\alpha=e^{-\\gamma}/2$)',    gen_powerlaw_params_beta_half,pal['dark_grey'],  't_min', 't_max',  '$\\tau_\\text{min}$', '$\\tau_\\text{max}$',   True,  True),
]
for key, title, df, color, xcol, ycol, xlabel, ylabel, xlog, ylog in row1_panels:
    scatter(ax_dict[key], df[xcol], df[ycol], color, xlabel, ylabel, xlog, ylog)
    ax_dict[key].set_title(title)

# --- Panel labels ---
for key, label in zip('ABCD', 'ABCD'):
    ax_dict[key].text(-0.3, 1.08, label, transform=ax_dict[key].transAxes,
                      fontsize=7, va='top', ha='left')

fig.savefig('figures/figS2.png', dpi=600, bbox_inches='tight')
