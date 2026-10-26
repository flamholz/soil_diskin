"""Figure S8: plot observed versus log-uniform-model fraction new carbon.

Also writes figures/figS8_parameters.png: the calibrated rate bounds as time
scales, tau_min = 1/k_max against tau_max = 1/k_min, on log axes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error

import viz


sites = pd.read_csv('results/processed_balesdent_2018.csv')
predictions = pd.read_csv(
    'results/04_model_predictions/loguniform_model_predictions.csv'
)

observed = sites['total_fnew'].to_numpy()
predicted = predictions['predicted_fnew'].to_numpy()
valid = np.isfinite(observed) & np.isfinite(predicted)
kge = RegressionMetric(
    y_true=observed[valid], y_pred=predicted[valid]
).kling_gupta_efficiency()
rmse = root_mean_squared_error(observed[valid], predicted[valid])

uncertainty = predictions[['predicted_fnew_05', 'predicted_fnew_95']].sub(
    predictions['predicted_fnew'], axis=0
).abs().fillna(0).to_numpy().T

plt.style.use('notebooks/style.mpl')
palette = viz.color_palette()
fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300, constrained_layout=True)
ax.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1, zorder=-10)
ax.errorbar(
    observed,
    predicted,
    yerr=uncertainty,
    fmt='o',
    color=palette['dark_blue'],
    ecolor='k',
    elinewidth=0.5,
    capsize=2,
    mec='k',
    mew=0.5,
    markersize=5,
    alpha=0.9,
)
ax.text(
    0.05,
    0.95,
    f'KGE = {kge:.2f}\nRMSE = {rmse:.2f}',
    transform=ax.transAxes,
    fontsize=6,
    va='top',
    bbox=dict(
        boxstyle='round',
        facecolor=palette['light_yellow'],
        edgecolor=palette['dark_grey'],
        alpha=0.8,
    ),
)
ax.set(
    title='log-uniform model',
    xlabel='observed F$_{new}$ ($\delta^{13}$C based)',
    ylabel='predicted F$_{new}$',
    xlim=(-0.05, 1.05),
    ylim=(-0.05, 1.05),
    xticks=np.arange(0, 1.1, 0.5),
    yticks=np.arange(0, 1.1, 0.5),
)
output = Path('figures/figS8.png')
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=600, bbox_inches='tight')


# Second figure: calibrated parameters as time scales. tau_max = 1/k_min is on a
# log axis. tau_min = 1/k_max = exp(-log_k_max) underflows for the broadest fitted
# spectra (log_width up to 5000), so we plot log10(tau_min) on a linear axis
# labeled as powers of ten.
params = pd.read_csv(
    'results/03_calibrate_models/loguniform_model_optimization_results.csv'
)
params = params[params['params_valid'].astype(bool)]
log10_e = np.log10(np.e)

fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300, constrained_layout=True)
# Lines span the fits with 5th/95th-percentile stocks of SoilGrids-backfilled sites.
if {'k_min_05', 'k_min_95'}.issubset(params.columns):
    backfilled = params.dropna(subset=['k_min_05', 'k_min_95'])
    ax.plot(
        -log10_e * backfilled[['log_k_max_05', 'log_k_max_95']].to_numpy().T,
        1 / backfilled[['k_min_05', 'k_min_95']].to_numpy().T,
        color='k',
        lw=0.5,
        zorder=1,
    )
ax.scatter(
    -log10_e * params['log_k_max'],
    1 / params['k_min'],
    color=palette['dark_blue'],
    edgecolor='k',
    linewidth=0.5,
    s=20,
    alpha=0.9,
    zorder=2,
)
ax.set_yscale('log')
# tau_max spans about a decade, so label 1, 2 and 5 times each power of ten.
ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'$10^{{{x:g}}}$'))
ax.set(
    title='log-uniform model parameters',
    xlabel=r'$\tau_{min} = 1/k_{max}$ (yr)',
    ylabel=r'$\tau_{max} = 1/k_{min}$ (yr)',
)
fig.savefig('figures/figS8_parameters.png', dpi=600, bbox_inches='tight')
