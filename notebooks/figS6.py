# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from permetrics.regression import RegressionMetric
import viz

plt.style.use('notebooks/style.mpl')
pal = viz.color_palette()

# load data

site_data = pd.read_csv('results/06_sensitivity_analysis/06c_model_predictions_veg_effects.csv')
# %%
# Plot results
def plot_data(col, ax, title, df, colors):
    """Helper function to plot the data.

    Args:
        col (str): Column name suffix for the predictions.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        title (str): Title of the plot.
        df (pd.DataFrame): Site data with observed and predicted columns.
        colors (dict): Color palette from viz.color_palette().
    """
    ax.scatter(df['total_fnew'], df['pred_conv' + col],
               color=colors['dark_blue'], lw=0.5, edgecolors='k', s=15,
               label='with vegetation age')
    ax.scatter(df['total_fnew'], df['pred_no_conv' + col],
               color=colors['light_grey'], lw=0.5, edgecolors='k', s=15,
               label='without vegetation age')

    evaluator_conv = RegressionMetric(y_true=df['total_fnew'].values, y_pred=df['pred_conv' + col].values)
    evaluator_no_conv = RegressionMetric(y_true=df['total_fnew'].values, y_pred=df['pred_no_conv' + col].values)

    kge_conv  = evaluator_conv.kling_gupta_efficiency()
    rmse_conv = evaluator_conv.root_mean_squared_error()
    kge_no    = evaluator_no_conv.kling_gupta_efficiency()
    rmse_no   = evaluator_no_conv.root_mean_squared_error()

    # Draw background box, then place each cell individually for per-column coloring.
    # light_grey on light_grey is unreadable, so "no veg." values use dark_grey text.
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.04, 0.78), 0.56, 0.19, transform=ax.transAxes,
                         boxstyle='round,pad=0.01', facecolor='lightgrey',
                         edgecolor='dimgrey', alpha=0.9, zorder=3)
    ax.add_patch(box)

    fs, ff = 5, 'monospace'
    kw = dict(transform=ax.transAxes, fontsize=fs, fontfamily=ff,
              va='top', zorder=4)
    # column x positions
    x0, x1, x2 = 0.06, 0.22, 0.42
    ax.text(x0, 0.95, '',          color='k',                     **kw)
    ax.text(x1, 0.95, 'with veg.', color=colors['dark_blue'],     **kw)
    ax.text(x2, 0.95, 'no veg.',   color=colors['dark_grey'],     **kw)
    ax.text(x0, 0.90, 'KGE',       color='k',                     **kw)
    ax.text(x1, 0.90, f'{kge_conv:.3f}',  color=colors['dark_blue'],  **kw)
    ax.text(x2, 0.90, f'{kge_no:.3f}',    color=colors['dark_grey'],  **kw)
    ax.text(x0, 0.85, 'RMSE',      color='k',                     **kw)
    ax.text(x1, 0.85, f'{rmse_conv:.3f}', color=colors['dark_blue'],  **kw)
    ax.text(x2, 0.85, f'{rmse_no:.3f}',   color=colors['dark_grey'],  **kw)
    ax.set(xlabel='observed fraction of new carbon', ylabel='predicted fraction of new carbon')
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    ax.set_title(title)
    ax.set_aspect('equal', 'box')

fig, axs = plt.subplots(1, 2, figsize=(4.76, 2.98), dpi=300)
plot_data('', axs[0], 'Power-law model', site_data, pal)
plot_data('_lognorm', axs[1], 'Lognormal model', site_data, pal)
axs[1].legend(loc='lower right')
plt.tight_layout()

# Save figure
fig.savefig('figures/figS6.png', dpi=600)


