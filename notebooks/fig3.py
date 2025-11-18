import pandas as pd
import matplotlib.pyplot as plt
import viz

from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error


#%% Load the site data
tropical_sites = pd.read_csv('results/processed_balesdent_2018.csv')

# use the style file
plt.style.use('notebooks/style.mpl')


pal = viz.color_palette()

#%% Load the predictions
# powerlaw_predictions = pd.read_csv('results/04_model_predictions/power_law_16-07-2025.csv',header=None, names=['prediction'])
powerlaw_predictions = pd.read_csv(f'results/04_model_predictions/power_law.csv',header=None, names=['prediction'])
lognormal_predictions = pd.read_csv(f'results/04_model_predictions/lognormal.csv',header=None, names=['prediction'])
gamma_predictions = pd.read_csv(f'results/04_model_predictions/gamma.csv',header=None, names=['prediction'])    
CLM45_predictions = pd.read_csv(f'results/04_model_predictions/CLM45_fnew.csv', header=None, names=['prediction'])
JSBACH_predictions = pd.read_csv(f'results/04_model_predictions/JSBACH_fnew.csv', header=None, names=['prediction'])
RCM_predictions = pd.read_csv(f'results/04_model_predictions/RCM.csv')
# %% Define function to plot model predictions
def plot_model_predictions(ax, predictions, model_name, color):
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--',
            label='y=x', zorder=-10, lw=1)
    ax.scatter(tropical_sites['total_fnew'],
               predictions, label=model_name, color=color,
               edgecolor='k', lw=0.5, s=20, alpha=0.9)

    evaluator = RegressionMetric(y_true=tropical_sites['total_fnew'].values, y_pred=predictions.values)
    rmse = root_mean_squared_error(tropical_sites['total_fnew'], predictions)
    # Make a single box reporting KGE and RMSE. Black border and grey background
    props = dict(boxstyle='round', facecolor=pal['light_yellow'],
                 edgecolor=pal['dark_grey'], alpha=0.8)
    box_text = f'KGE = {evaluator.kling_gupta_efficiency():.2f}\nRMSE = {rmse:.2f}'
    ax.text(0.05, 0.95, box_text,
            transform=ax.transAxes, fontsize=6,
            verticalalignment='top', bbox=props)

    # ax.text(0.05, 0.95, f'KGE = {evaluator.kling_gupta_efficiency():.2f}',
    #         transform=ax.transAxes, fontsize=6, verticalalignment='top')
    # ax.text(0.05, 0.90, f'RMSE = {rmse:.2f}', transform=ax.transAxes,
    #         fontsize=6, verticalalignment='top')
    #ax.set(xlabel='observed', ylabel='predicted')
    ax.set_title(model_name) 

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
    ax.set_xlabel('observed F$_{new}$')
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