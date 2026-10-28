"""Figure S3: observed vs. predicted F_new for every model, with bootstrap KGE distributions."""
#%% 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import viz

from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error
from matplotlib.colors import LogNorm

import os 

if os.getcwd().endswith('notebooks'):
    os.chdir('..')

#%% Load the site data
all_sites = pd.read_csv('results/processed_balesdent_2018.csv')

# use the style file
plt.style.use('notebooks/style.mpl')
pal = viz.color_palette()

#%% Load the predictions
powerlaw_predictions = pd.read_csv('results/04_model_predictions/power_law_model_predictions.csv')
gen_powerlaw_preds_alpha = pd.read_csv('results/04_model_predictions/general_power_law_model_predictions.csv')
gen_powerlaw_preds_alpha_half = pd.read_csv('results/04_model_predictions/general_power_law_model_predictions_alpha_half.csv')
lognormal_predictions = pd.read_csv('results/04_model_predictions/lognormal_model_predictions.csv')
weibull_predictions = pd.read_csv('results/04_model_predictions/weibull_model_predictions.csv')
CLM45_predictions = pd.read_csv('results/04_model_predictions/CLM45_fnew.csv', header=None, names=['prediction'])
JSBACH_predictions = pd.read_csv('results/04_model_predictions/JSBACH_fnew.csv', header=None, names=['prediction'])
RCM_predictions = pd.read_csv('results/04_model_predictions/RCM.csv')

# Model groupings reused across figures
continuum_model_colors = [pal['dark_blue'], pal['blue'], pal['light_blue']]
continuum_models = [lognormal_predictions, powerlaw_predictions, gen_powerlaw_preds_alpha]
continuum_model_titles = ['lognormal model', 'power law model ($\\alpha = 1$)', 'power law model ($\\alpha = e^{-\\gamma}$)']

ESM_models = [CLM45_predictions, JSBACH_predictions]
ESM_model_titles = ['CLM4.5', 'JSBACH']

RCM_colors = [pal['dark_green'], pal['green'], pal['light_green']]


# Observed vs. predicted F_new, colored by clabel with the given colormap and normalization
def plot_model_predictions_cmap(ax, predictions, model_name, clabel, cmap, norm):
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--',
            label='y=x', zorder=-10, lw=1)
    
    # Plot with different markers for different data sources
    balesdent_mask = all_sites['C_data_source'] == 'Balesdent et al. 2018'
    soilgrids_mask = all_sites['C_data_source'] == 'SoilGrids backfill'
    
    # Plot Balesdent points with circles
    sc = ax.scatter(all_sites.loc[balesdent_mask, 'total_fnew'],
                    predictions[balesdent_mask], label=model_name,
                    c=all_sites.loc[balesdent_mask, clabel], cmap=cmap, norm=norm,
                    edgecolor='k', lw=0.5, s=20, alpha=0.9, marker='o')
    
    # Plot SoilGrids points with diamonds
    ax.scatter(all_sites.loc[soilgrids_mask, 'total_fnew'],
               predictions[soilgrids_mask],
               c=all_sites.loc[soilgrids_mask, clabel], cmap=cmap, norm=norm,
               edgecolor='k', lw=0.5, s=20, alpha=0.9, marker='D')
    # calculate metrics
    true_vals = all_sites['total_fnew'].values
    predictions = predictions.values
    mask = ~pd.isna(true_vals) & ~pd.isna(predictions)
    evaluator = RegressionMetric(y_true=true_vals[mask],
                                 y_pred=predictions[mask])
    rmse = root_mean_squared_error(true_vals[mask], predictions[mask])
    kge = evaluator.kling_gupta_efficiency()

    # Make a single box reporting KGE and RMSE. Black border and grey background
    props = dict(boxstyle='round', facecolor=pal['light_yellow'],
                 edgecolor=pal['dark_grey'], alpha=0.8)
    box_text = f'KGE = {kge:.2f}\nRMSE = {rmse:.2f}'
    ax.text(0.05, 0.95, box_text,
            transform=ax.transAxes, fontsize=6,
            verticalalignment='top', bbox=props)
    ax.set_title(model_name)
    return sc

# %% make a supplementary version of the above plot where the points are colored by the
# timing of the land use change event. Using a log color scale. Also, we now include the 
# plots for the generalized power law model.
mosaic = """ABCDE\nFGHIJ\nKKKKK"""
fig, axs = plt.subplot_mosaic(mosaic, figsize=(7.24, 4),
                              dpi=300, constrained_layout=True,
                              sharex=False, sharey=False)

# make a log-scaled color map for the duration of labeling
cmap = 'viridis'
norm = LogNorm(vmin=all_sites['Duration_labeling'].min(), vmax=all_sites['Duration_labeling'].max())
cnames = 'dark_blue,blue,light_blue,dark_grey'.split(',')
continuum_model_colors = [pal[c] for c in cnames] + ['grey']
continuum_models = [lognormal_predictions, weibull_predictions, powerlaw_predictions,
                    gen_powerlaw_preds_alpha, gen_powerlaw_preds_alpha_half]
continuum_model_titles = ['lognormal model', 'hockey-stick model', 'power law model ($\\alpha = 1$)',
                          'power law ($\\alpha = e^{-\\gamma}$)',
                          'power law ($\\alpha = e^{-\\gamma}/2$)']
my_axs = [axs[c] for c in 'ABCDE']
for ax, predictions, title in zip(my_axs, continuum_models, continuum_model_titles):
    sc = plot_model_predictions_cmap(ax, predictions['predicted_fnew'], title,
                                     clabel='Duration_labeling', cmap=cmap, norm=norm)

my_axs = [axs[c] for c in 'FG']
for ax, predictions, title in zip(my_axs, ESM_models, ESM_model_titles):
    sc = plot_model_predictions_cmap(ax, predictions['prediction'], title,
                                     clabel='Duration_labeling', cmap=cmap, norm=norm)
    
# Reduced complexity model predictions
my_axs = [axs[c] for c in 'HIJ']
for i, col in enumerate(RCM_predictions.columns):
    title = col + ' (RC)'
    sc = plot_model_predictions_cmap(my_axs[i], RCM_predictions[col], title,
                                     clabel='Duration_labeling', cmap=cmap, norm=norm)

colorbar_label = 'time since transition (yrs)'
my_axs = [axs[c] for c in 'ABCDEFGHIJ']
cbar = plt.colorbar(sc, ax=my_axs, orientation='vertical',
                    label=colorbar_label, pad=0.01)

# Set axes labels on the outer plots
for c in 'FGHIJ':
    axs[c].set_xlabel('observed F$_{new}$')
for c in 'AF':
    axs[c].set_ylabel('predicted F$_{new}$')

# Load the bootstrapping calculation to plot KGE distribution in panel K as a boxplot
metric_dists = pd.read_csv('results/fig4_calcs.csv')
kge_data = metric_dists[metric_dists['metric'] == 'KGE']
# Order -- continuum, then ESM, then reduced complexity
order = ['Lognormal', 'Hockey-stick', 'Power-law',
         'Gen. Power-law (a=exp(-gamma))',
         'Gen. Power-law (a=exp(-gamma)/2)',
         'CLM4.5', 'JSBACH', 'CESM1', 'IPSL-CM5A-LR', 'MRI-ESM1']
xlabels = ['lognormal', 'hockey-stick', 'power law\n($\\alpha = 1$)',
           'power law\n($\\alpha = e^{-\\gamma}$)',
           'power law\n($\\alpha = e^{-\\gamma}/2$)',
           'CLM4.5',  'JSBACH', 'CESM1 (RC)', 'IPSL-CM5A-LR (RC)', 'MRI-ESM1 (RC)']
axs['K'].set_xticklabels(xlabels, rotation=45, ha='right', fontsize=6)
sns.violinplot(
    data=kge_data, x='model', y='value',
    ax=axs['K'], order=order)

# add mean and standard deviation to each boxplot
means = kge_data.groupby('model')['value'].mean()
stds = kge_data.groupby('model')['value'].std()
axs['K'].set_ylim(-1.55, 1.3)
for i, model in enumerate(order):
    mean = means[model]
    std = stds[model]
    axs['K'].text(i, 0.95, f'{mean:.2f}±{std:.2f}',
                  ha='center', va='bottom', fontsize=5)
# set y label
axs['K'].set_ylabel('KGE value')
axs['K'].set_xlabel('')

# add labels to each subplot
for i, c in enumerate("ABCDEFGHIJ"):
    axs[c].text(
        -0.25, 1.2, c, transform=axs[c].transAxes,
        fontsize=7, va='top', ha='left')

# make the tick labels smaller
for c in 'ABCDEFGHIJ':
    axs[c].tick_params(axis='both', which='major', labelsize=5)
    axs[c].set_xticks(np.arange(0, 1.1, 0.5))
    axs[c].set_yticks(np.arange(0, 1.1, 0.5))
# add subpanel label
axs['K'].text(-0.04, 1.1, 'K', transform=axs['K'].transAxes,
             fontsize=7, va='top', ha='left')

axs['K'].tick_params(axis='y', which='major', labelsize=5)
axs['K'].set_yticks(np.arange(-1, 1.1, 1.0))

plt.savefig('figures/figS3.png', dpi=300, bbox_inches='tight')

# %%
