# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error
from scipy.interpolate import interp1d

# %%
pl_data = pd.read_csv('results/06_sensitivity_analysis/powerlaw_turnover_sensitivity_results.csv')
site_data = pd.read_csv('results/processed_balesdent_2018.csv')
turnover_14C = pd.read_csv('results/all_sites_14C_turnover.csv')

pl_data.columns = pl_data.columns.astype(float)
pl_data.reset_index(drop=True, inplace=True)

# %%
def predict_fnew(cdf):
    ts = cdf.columns.astype(float).values
    predictions = []
    for i, row in site_data.iterrows():
        site_cdf = interp1d(ts, cdf.loc[i, :].values / turnover_14C.loc[i, 'turnover'])
        predictions.append(site_cdf(row['Duration_labeling']))
    return pd.Series(predictions)

ratios = ['0.50', '0.67', '1', '1.50', '2']
ratio_files = [f'results/06_sensitivity_analysis/06a_lognormal_cdfs_{a}.csv' for a in ratios]
ln_data = pd.concat(
    [predict_fnew(pd.read_csv(f)) for f in ratio_files],
    axis=1
)
ln_data.columns = pl_data.columns

# %%
fig, axs = plt.subplots(2, 5, figsize=(12, 5), constrained_layout=True, sharex=True, sharey=True, dpi=300)

col_map = {0.5: 0, 1/1.5: 1, 1: 2, 1.5: 3, 2: 4}
models = [('PowerLaw', pl_data, 0), ('Lognormal', ln_data, 1)]

for (model, df, row), ratio in product(models, [0.5, 1/1.5, 1, 1.5, 2]):
    col = col_map[ratio]
    ax = axs[row, col]
    jdf = pd.concat([df[ratio], site_data['total_fnew']], axis=1, keys=['pred', 'obs']).dropna()

    evaluator = RegressionMetric(y_true=jdf['obs'].values, y_pred=jdf['pred'].values)
    rmse = root_mean_squared_error(jdf['obs'], jdf['pred'])
    ax.text(0.05, 0.95, f'KGE: {evaluator.kling_gupta_efficiency():.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.85, f'RMSE: {rmse:.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    if row == 1:
        ax.set(xlabel='observed')
    if col == 0:
        ax.set(ylabel='predicted')

    ax.scatter(jdf['obs'], jdf['pred'], color='k', s=10, lw=0)
    ax.plot([0, 1], [0, 1], ls='--', color='k')
    ax.set_title(f'turnover time ratio={round(ratio, 2)}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', 'box')

axs[0, 0].text(-0.5, 0.5, 'Power-law model', transform=axs[0, 0].transAxes, fontsize=12, rotation=90, verticalalignment='center')
axs[1, 0].text(-0.5, 0.5, 'Lognormal model', transform=axs[1, 0].transAxes, fontsize=12, rotation=90, verticalalignment='center')

fig.savefig('figures/figS4.png', dpi=600)
