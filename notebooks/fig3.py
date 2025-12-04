#%% 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import viz

from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error
from matplotlib.colors import LogNorm
# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')
#%% Load the site data
all_sites = pd.read_csv('results/processed_balesdent_2018.csv')

# use the style file
plt.style.use('notebooks/style.mpl')
pal = viz.color_palette()

#%% Load the predictions
powerlaw_predictions = pd.read_csv(f'results/04_model_predictions/power_law.csv',header=None, names=['prediction'])
gen_powerlaw_preds_beta = pd.read_csv(f'results/04_model_predictions/general_power_law.csv', header=None, names=['prediction'])
gen_powerlaw_preds_beta_half = pd.read_csv(f'results/04_model_predictions/general_power_law_beta_half.csv', header=None, names=['prediction'])
lognormal_predictions = pd.read_csv(f'results/04_model_predictions/lognormal.csv',header=None, names=['prediction'])
gamma_predictions = pd.read_csv(f'results/04_model_predictions/gamma.csv',header=None, names=['prediction'])    
CLM45_predictions = pd.read_csv(f'results/04_model_predictions/CLM45_fnew.csv', header=None, names=['prediction'])
JSBACH_predictions = pd.read_csv(f'results/04_model_predictions/JSBACH_fnew.csv', header=None, names=['prediction'])
RCM_predictions = pd.read_csv(f'results/04_model_predictions/RCM.csv')


# %% Define function to plot model predictions
def plot_model_predictions(ax, predictions, model_name, color):
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--',
            label='y=x', zorder=-10, lw=1)
    ax.scatter(all_sites['total_fnew'],
               predictions, label=model_name, color=color,
               edgecolor='k', lw=0.5, s=20, alpha=0.9)

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
    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))


# Alternate version with colormap and color normalization
def plot_model_predictions_cmap(ax, predictions, model_name, clabel, cmap, norm):
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--',
            label='y=x', zorder=-10, lw=1)
    sc = ax.scatter(all_sites['total_fnew'],
               predictions, label=model_name,
               c=all_sites[clabel], cmap=cmap, norm=norm,
               edgecolor='k', lw=0.5, s=20, alpha=0.9)

    # calculate metrics
    true_vals = all_sites['total_fnew'].values
    predictions = predictions.values
    mask = ~pd.isna(true_vals) & ~pd.isna(predictions)
    evaluator = RegressionMetric(y_true=true_vals[mask],
                                 y_pred=predictions[mask])
    rmse = root_mean_squared_error(true_vals[mask], predictions[mask])
    kge = evaluator.kling_gupta_efficiency()

    rmse = root_mean_squared_error(all_sites['total_fnew'], predictions)
    # Make a single box reporting KGE and RMSE. Black border and grey background
    props = dict(boxstyle='round', facecolor=pal['light_yellow'],
                 edgecolor=pal['dark_grey'], alpha=0.8)
    box_text = f'KGE = {kge:.2f}\nRMSE = {rmse:.2f}'
    ax.text(0.05, 0.95, box_text,
            transform=ax.transAxes, fontsize=6,
            verticalalignment='top', bbox=props)
    ax.set_title(model_name)
    return sc

#%% 
# Plot the generalized power law model predictions to check
fig, axs = plt.subplots(1, 3, figsize=(7.24, 2), dpi=300, constrained_layout=True)

plt.sca(axs[0])
plot_model_predictions(axs[0], powerlaw_predictions['prediction'], 'power law model', pal['dark_blue'])

plt.sca(axs[1])
plot_model_predictions(axs[1], gen_powerlaw_preds_beta['prediction'], 'generalized power law model', pal['blue'])

plt.sca(axs[2])
plot_model_predictions(axs[2], gen_powerlaw_preds_beta_half['prediction'], 'generalized power law model (beta/2)', pal['light_blue'])

plt.savefig('figures/gen_powerlaw.png', dpi=300, bbox_inches='tight')

# %% Plot the predictions predictions
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(7.24, 3.5),
                        dpi=300, constrained_layout=True,
                        sharex=True, sharey=True)
axs = axs.flatten()

continuum_model_colors = [pal['dark_blue'], pal['blue'], pal['light_blue']]
continuum_models = [powerlaw_predictions, lognormal_predictions, gamma_predictions]
continuum_model_titles = ['power law model', 'lognormal model', 'gamma model']

for ax, predictions, title, color in zip(axs[:3], continuum_models,
                                         continuum_model_titles, continuum_model_colors):
    plot_model_predictions(ax, predictions['prediction'], title, color)

ESM_model_colors = [pal['dark_purple'], pal['purple']]
ESM_models = [CLM45_predictions, JSBACH_predictions]
ESM_model_titles = ['CLM4.5', 'JSBACH']
for ax, predictions, title, color in zip(axs[3:5], ESM_models,
                                         ESM_model_titles, ESM_model_colors):
    plot_model_predictions(ax, predictions['prediction'], title, color)

# Reduced complexity model predictions
RCM_colors = [pal['dark_green'], pal['green'], pal['light_green']]
for i, col in enumerate(RCM_predictions.columns):
    title = col +  ' ($^{14} C$ corrected)'
    plot_model_predictions(
        axs[5 + i], RCM_predictions[col], title, RCM_colors[i])
    
# Set axes labels on the outer plots
for ax in axs[4:8]:
    ax.set_xlabel('observed F$_{new}$ ($\delta^{13}C$ based)')
for ax in axs[[0, 4]]:
    ax.set_ylabel('predicted F$_{new}$')

# add labels to each subplot
for i, (ax, label) in enumerate(zip(axs, "ABCDEFGH")):
    if label == 'A' or label == 'E':
        ax.text(
            -0.25, 1.1, label, transform=ax.transAxes,
            fontsize=7, va='top', ha='left')
    else:
        ax.text(
            -0.15, 1.1, label, transform=ax.transAxes,
            fontsize=7, va='top', ha='left')
# %% Save the figure
out_fname = f'figures/fig3.png'
plt.savefig(out_fname, dpi=300, bbox_inches='tight')

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
continuum_models = [lognormal_predictions, gamma_predictions, powerlaw_predictions,
                    gen_powerlaw_preds_beta, gen_powerlaw_preds_beta_half]
continuum_model_titles = ['lognormal model', 'gamma model', 'power law model',
                          'gen. power law ($\\beta = e^{-\\gamma}$)',
                          'gen. power law ($\\beta = e^{-\\gamma}/2$)']
my_axs = [axs[c] for c in 'ABCDE']
for ax, predictions, title, color in zip(my_axs, continuum_models,
                                         continuum_model_titles, continuum_model_colors):
    sc = plot_model_predictions_cmap(ax, predictions['prediction'], title,
                                     clabel='Duration_labeling', cmap=cmap, norm=norm)

my_axs = [axs[c] for c in 'FG']
for ax, predictions, title, color in zip(my_axs, ESM_models,
                                         ESM_model_titles, ESM_model_colors):
    sc = plot_model_predictions_cmap(ax, predictions['prediction'], title,
                                     clabel='Duration_labeling', cmap=cmap, norm=norm)
    
# Reduced complexity model predictions
my_axs = [axs[c] for c in 'HIJ']
for i, col in enumerate(RCM_predictions.columns):
    title = col
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

# add labels to each subplot
for i, c in enumerate("ABCDEFGHIJ"):
    if c == 'A' or c == 'F':
        axs[c].text(
            -0.25, 1.1, c, transform=axs[c].transAxes,
            fontsize=7, va='top', ha='left')
    else:
        axs[c].text(
            -0.15, 1.1, c, transform=axs[c].transAxes,
            fontsize=7, va='top', ha='left')

# Load the bootstrapping calculation to plot KGE distribution in panel K as a boxplot
metric_dists = pd.read_csv('results/fig3_calcs.csv')
kge_data = metric_dists[metric_dists['metric'] == 'KGE']
# Order -- continuum, then ESM, then reduced complexity
order = ['Lognormal', 'Gamma', 'Power-law',
         'Gen. Power-law (b=exp(-gamma))',
         'Gen. Power-law (b=exp(-gamma)/2)',
         'CLM4.5', 'JSBACH', 'CESM1', 'IPSL-CM5A-LR', 'MRI-ESM1']
xlabels = ['lognormal', 'gamma', 'power law',
           'gen. power law\n($\\beta = e^{-\\gamma}$)',
           'gen. power law\n($\\beta = e^{-\\gamma}/2$)',
           'CLM4.5',  'JSBACH', 'CESM1', 'IPSL-CM5A-LR', 'MRI-ESM1']
axs['K'].set_xticklabels(xlabels, rotation=45, ha='right', fontsize=6)
sns.boxplot(data=kge_data, x='model', y='value', ax=axs['K'],
            order=order)
# add mean and standard deviation to each boxplot
means = kge_data.groupby('model')['value'].mean()
stds = kge_data.groupby('model')['value'].std()
axs['K'].set_ylim(-1.55, 1.25)
for i, model in enumerate(order):
    mean = means[model]
    std = stds[model]
    axs['K'].text(i, 0.93, f'{mean:.2f}±{std:.2f}',
                  ha='center', va='bottom', fontsize=5)
# set y label
axs['K'].set_ylabel('KGE value')
# add subpanel label
axs['K'].text(-0.06, 1.1, 'K', transform=axs['K'].transAxes,
             fontsize=7, va='top', ha='left')

out_fname = f'figures/figS3.png'
plt.savefig(out_fname, dpi=300, bbox_inches='tight')

# %% make a presentation version with no subpanel labels
# show only the continuum models and ESMs
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(7.24, 1.75),
                        dpi=300, constrained_layout=True,
                        sharex=False, sharey=True)

# for presentation, want lognormal, gamma, powerlaw, in that order
continuum_model_colors = [pal['dark_blue'], pal['blue'], pal['light_blue']]
continuum_models = [lognormal_predictions, gamma_predictions, powerlaw_predictions]
continuum_model_titles = ['lognormal model', 'gamma model', 'power law model']
for ax, predictions, title, color in zip(axs[:3], continuum_models,
                                         continuum_model_titles, continuum_model_colors):
    plot_model_predictions(ax, predictions['prediction'], title, color) 

for ax, predictions, title, color in zip(axs[3:5], ESM_models,
                                         ESM_model_titles, ESM_model_colors):
    plot_model_predictions(ax, predictions['prediction'], title, color)

# Set axes labels on the outer plots
for ax in axs:
    ax.set_xlabel('observed F$_{new}$')
axs[0].set_ylabel('predicted F$_{new}$')

# %% Save the presentation figure
out_fname = f'figures/fig3_presentation.png'
plt.savefig(out_fname, dpi=300, bbox_inches='tight')

# Make a version of the presentation figure where the points are colored by the
# timing of the land use change event. 
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(7.24, 1.75),
                        dpi=300, constrained_layout=True,
                        sharex=False, sharey=True)
# for presentation, want lognormal, gamma, powerlaw, in that order
continuum_model_colors = [pal['dark_blue'], pal['blue'], pal['light_blue']]
continuum_models = [lognormal_predictions, gamma_predictions, powerlaw_predictions]
continuum_model_titles = ['lognormal model', 'gamma model', 'power law model']

# make a log-scaled color map for the duration of labeling
cmap = 'viridis'
norm = LogNorm(vmin=all_sites['Duration_labeling'].min(), vmax=all_sites['Duration_labeling'].max())
for ax, predictions, title, color in zip(axs[:3], continuum_models,
                                         continuum_model_titles, continuum_model_colors):
    sc = plot_model_predictions_cmap(ax, predictions['prediction'], title,
                                     clabel='Duration_labeling', cmap=cmap, norm=norm)

for ax, predictions, title, color in zip(axs[3:5], ESM_models,
                                         ESM_model_titles, ESM_model_colors):
    sc = plot_model_predictions_cmap(ax, predictions['prediction'], title,
                                     clabel='Duration_labeling', cmap=cmap, norm=norm)
colorbar_label = 'time since transition (yrs)'
cbar = plt.colorbar(sc, ax=axs, orientation='vertical', label=colorbar_label, pad=0.01)

plt.savefig('figures/fig3_presentation_colored_by_labeling_duration.png', dpi=300, bbox_inches='tight')

# %%
