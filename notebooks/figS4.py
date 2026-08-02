# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error

plt.style.use('notebooks/style.mpl')
import viz
pal = viz.color_palette()

# %%
# Load the data for the power-law model.
pl_data = pd.read_csv('results/06_sensitivity_analysis/powerlaw_turnover_sensitivity_results.csv')
site_data = pd.read_csv('results/processed_balesdent_2018.csv')

pl_data.columns = pl_data.columns.astype(float)
pl_data.reset_index(drop=True, inplace=True)


# %%
# Load the data for the lognormal model.
ln_data = pd.read_csv('results/06_sensitivity_analysis/06a_lognormal_fnew.csv')
ln_data.columns = pl_data.columns

# %%
# Load the predictions based on the q05/q95 C stocks of the sites whose stocks were
# backfilled from SoilGrids. These give the error bars of those sites.
def load_quantiles(fname_fmt):
    quantiles = [pd.read_csv(fname_fmt.format(q)) for q in ['q05', 'q95']]
    for df in quantiles:
        df.columns = pl_data.columns
    return quantiles

pl_05, pl_95 = load_quantiles('results/06_sensitivity_analysis/powerlaw_turnover_sensitivity_results_{}.csv')
ln_05, ln_95 = load_quantiles('results/06_sensitivity_analysis/06a_lognormal_fnew_{}.csv')

soilgrids_mask = site_data['C_data_source'] == 'SoilGrids backfill'

# %%
# Make the figure
fig, axs = plt.subplots(2, 5, figsize=(7.24, 3.02), constrained_layout=True, sharex=True, sharey=True, dpi=300)

col_map = {0.5: 0, 1/1.5: 1, 1: 2, 1.5: 3, 2: 4}
models = [('PowerLaw', pl_data, pl_05, pl_95, 0), ('Lognormal', ln_data, ln_05, ln_95, 1)]

for (model, df, df_05, df_95, row), ratio in product(models, [0.5, 1/1.5, 1, 1.5, 2]):
    col = col_map[ratio]
    ax = axs[row, col]
    jdf = pd.concat([df[ratio], site_data['total_fnew'], df_05[ratio], df_95[ratio],
                     soilgrids_mask],
                    axis=1,
                    keys=['pred', 'obs', 'q05', 'q95', 'soilgrids']).dropna(subset=['pred', 'obs'])

    evaluator = RegressionMetric(y_true=jdf['obs'].values, y_pred=jdf['pred'].values)
    rmse = root_mean_squared_error(jdf['obs'], jdf['pred'])
    props = dict(boxstyle='round', facecolor=pal['light_grey'], edgecolor=pal['dark_grey'], alpha=0.5)
    box_text = f'KGE = {evaluator.kling_gupta_efficiency():.2f}\nRMSE = {rmse:.2f}'
    ax.text(0.05, 0.95, box_text, transform=ax.transAxes, fontsize=6,
            verticalalignment='top', bbox=props)

    if row == 1:
        ax.set(xlabel='observed')
    if col == 0:
        ax.set(ylabel='predicted')

    # Sites with measured C stocks are plotted as circles, sites whose stocks were
    # backfilled from SoilGrids as diamonds with the q05/q95 range as error bars.
    ax.scatter(jdf.loc[~jdf['soilgrids'], 'obs'], jdf.loc[~jdf['soilgrids'], 'pred'],
               color='k', s=10, lw=0)

    sg = jdf[jdf['soilgrids']]
    bounds = sg[['q05', 'q95']]
    yerr = np.abs(np.vstack([bounds.min(axis=1) - sg['pred'],
                             bounds.max(axis=1) - sg['pred']]))
    ax.errorbar(sg['obs'], sg['pred'], yerr=np.nan_to_num(yerr), fmt='D', color='k',
                ecolor='k', elinewidth=0.5, capsize=1.5, markersize=2.5, lw=0)

    ax.plot([0, 1], [0, 1], ls='--', color='k')
    ax.set_title(f'turnover time ratio={round(ratio, 2)}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', 'box')

axs[0, 0].text(-0.5, 0.5, 'Power-law model', transform=axs[0, 0].transAxes, fontsize=7, rotation=90, verticalalignment='center')
axs[1, 0].text(-0.5, 0.5, 'Lognormal model', transform=axs[1, 0].transAxes, fontsize=7, rotation=90, verticalalignment='center')

fig.savefig('figures/figS4.png', dpi=600)
