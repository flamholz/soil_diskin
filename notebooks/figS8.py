"""Figure S8: plot observed versus log-uniform-model fraction new carbon."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
