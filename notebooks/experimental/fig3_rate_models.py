#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import viz

from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error

if os.getcwd().endswith('notebooks'):
    os.chdir('..')

#%% Load data and predictions
all_sites = pd.read_csv('results/processed_balesdent_2018.csv')

plt.style.use('notebooks/style.mpl')
pal = viz.color_palette()

prediction_files = {
    'Power-law': 'results/04_model_predictions/power_law_model_predictions.csv',
    'Lognormal': 'results/04_model_predictions/lognormal_model_predictions.csv',
    'Gamma': 'results/04_model_predictions/gamma_model_predictions.csv',
    'Gaussian rate': 'results/04_model_predictions/gaussian_rate_model_predictions.csv',
    'Log-uniform rate': 'results/04_model_predictions/loguniform_rate_model_predictions.csv',
}
prediction_dfs = {
    model_name: pd.read_csv(fname)
    for model_name, fname in prediction_files.items()
}
metric_dists = pd.read_csv('results/fig3_rate_models_calcs.csv')


def plot_model_predictions(ax, predictions, model_name, color, pred_err=None):
    """Plot observed vs predicted F_new with RMSE and KGE."""
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', zorder=-10, lw=1)
    if pred_err is not None:
        ax.errorbar(
            all_sites['total_fnew'],
            predictions,
            yerr=pred_err,
            fmt='o',
            color=color,
            ecolor='k',
            elinewidth=0.5,
            capsize=2,
            mec='k',
            mew=0.5,
            markersize=5,
            alpha=0.9,
        )
    else:
        ax.scatter(
            all_sites['total_fnew'],
            predictions,
            color=color,
            edgecolor='k',
            lw=0.5,
            s=20,
            alpha=0.9,
        )

    true_vals = all_sites['total_fnew'].values
    predicted_vals = predictions.values
    mask = np.isfinite(true_vals) & np.isfinite(predicted_vals)
    evaluator = RegressionMetric(
        y_true=true_vals[mask],
        y_pred=predicted_vals[mask],
    )
    rmse = root_mean_squared_error(true_vals[mask], predicted_vals[mask])
    kge = evaluator.kling_gupta_efficiency()

    props = dict(
        boxstyle='round',
        facecolor=pal['light_yellow'],
        edgecolor=pal['dark_grey'],
        alpha=0.8,
    )
    ax.text(
        0.05,
        0.95,
        f'KGE = {kge:.2f}\nRMSE = {rmse:.2f}',
        transform=ax.transAxes,
        fontsize=6,
        verticalalignment='top',
        bbox=props,
    )
    ax.set_title(model_name)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))


#%% Plot rate-model comparison
model_order = ['Power-law', 'Lognormal', 'Gamma', 'Gaussian rate', 'Log-uniform rate']
model_titles = ['power law', 'lognormal', 'gamma', 'Gaussian rate', 'log-uniform rate']
model_colors = [
    pal['dark_blue'],
    pal['blue'],
    pal['light_blue'],
    pal['green'],
    pal['dark_green'],
]

fig, axs = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(7.24, 4.0),
    dpi=300,
    constrained_layout=True,
    sharex=False,
    sharey=False,
)
axs = axs.flatten()

for ax, model_name, title, color in zip(
    axs[:5],
    model_order,
    model_titles,
    model_colors,
):
    predictions = prediction_dfs[model_name]
    pred_err = None
    uncertainty_cols = ['predicted_fnew_05', 'predicted_fnew_95']
    if all(col in predictions.columns for col in uncertainty_cols):
        pred_err = (
            predictions[uncertainty_cols]
            .sub(predictions['predicted_fnew'], axis=0)
            .abs()
            .fillna(0)
            .values
            .T
        )
    plot_model_predictions(
        ax,
        predictions['predicted_fnew'],
        title,
        color,
        pred_err=pred_err,
    )

for ax in axs[:3]:
    ax.set_xlabel('')
for ax in axs[3:5]:
    ax.set_xlabel('observed F$_{new}$')
for ax in axs[[0, 3]]:
    ax.set_ylabel('predicted F$_{new}$')

kge_data = metric_dists[metric_dists['metric'] == 'KGE']
sns.violinplot(data=kge_data, x='model', y='value', ax=axs[5], order=model_order)
axs[5].set_xticks(np.arange(len(model_order)))
axs[5].set_xticklabels(
    ['power law', 'lognormal', 'gamma', 'Gaussian\nrate', 'log-uniform\nrate'],
    rotation=45,
    ha='right',
    fontsize=6,
)
axs[5].set_ylabel('KGE value')
axs[5].set_xlabel('')
axs[5].set_ylim(-1.55, 1.3)
means = kge_data.groupby('model')['value'].mean()
stds = kge_data.groupby('model')['value'].std()
for i, model_name in enumerate(model_order):
    if model_name not in means:
        continue
    axs[5].text(
        i,
        0.95,
        f'{means[model_name]:.2f}+/-{stds[model_name]:.2f}',
        ha='center',
        va='bottom',
        fontsize=5,
    )

for ax, label in zip(axs, 'ABCDEF'):
    ax.text(
        -0.18,
        1.1,
        label,
        transform=ax.transAxes,
        fontsize=7,
        va='top',
        ha='left',
    )

plt.savefig('figures/fig3_rate_models.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig3_rate_models.svg', dpi=300, bbox_inches='tight')
# %%
