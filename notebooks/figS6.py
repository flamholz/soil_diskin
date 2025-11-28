# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from permetrics.regression import RegressionMetric

# load data

site_data = pd.read_csv('results/06_sensitivity_analysis/06c_model_predictions_veg_effects.csv')
# %%
# Plot results

fig, axs = plt.subplots(1, 3, figsize=(12, 5), dpi = 300)

def plot_data(col, ax, title):
    ax.scatter(site_data['total_fnew'], site_data['pred_conv' + col], color='k')
    ax.scatter(site_data['total_fnew'], site_data['pred_no_conv' + col], color='royalblue')

    evaluator_conv = RegressionMetric(y_true=site_data['total_fnew'].values, y_pred=site_data['pred_conv' + col].values)
    evaluator_no_conv = RegressionMetric(y_true=site_data['total_fnew'].values, y_pred=site_data['pred_no_conv' + col].values)

    ax.text(0.05, 0.95, f'KGE: {evaluator_conv.kling_gupta_efficiency():.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.90, f'RMSE: {evaluator_conv.root_mean_squared_error():.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    ax.text(0.05, 0.85, f'KGE: {evaluator_no_conv.kling_gupta_efficiency():.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', color='royalblue')
    ax.text(0.05, 0.80, f'RMSE: {evaluator_no_conv.root_mean_squared_error():.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', color='royalblue')
    ax.set(xlabel='observed fraction of new carbon', ylabel='predicted fraction of new carbon')
    plt.legend(['with vegetation age', 'without vegetation age'], loc='lower right')
    ax.plot([0, 1], [0, 1], color='k', linestyle='--', label='y=x')
    ax.set_title(title)
    ax.set_aspect('equal', 'box')

plot_data('', axs[0], 'Power-law model')
plot_data('_lognorm', axs[1], 'Lognormal model')
plot_data('_gamma', axs[2], 'Gamma model')
plt.tight_layout()

# Save figure
fig.savefig('figures/figS5.png', dpi=600)


