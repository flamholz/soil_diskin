#%%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

# %% Import libraries
import pickle
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import pearsonr, spearmanr
from notebooks.models import PowerLawDisKin
from notebooks.constants import INTERP_R_14C, C14_DATA
import matplotlib.pyplot as plt
from permetrics.regression import RegressionMetric


#%% Load the site data
tropical_sites = pd.read_csv('results/processed_balesdant_2018.csv')

#%% Load the predictions
gamma_predictions = pd.read_csv('results/04_model_predictions/gamma_28-07-2025.csv', header=None, names=['prediction'])
powerlaw_predictions = pd.read_csv('results/04_model_predictions/power_law_16-07-2025.csv',header=None, names=['prediction'])
generalized_powerlaw_predictions = pd.read_csv('results/04_model_predictions/general_power_law_23-07-2025.csv',header=None, names=['prediction'])
lognormal_predictions = pd.read_csv('results/04_model_predictions/lognormal_16-07-2025.csv',header=None, names=['prediction'])
CLM45_predictions = pd.read_csv('results/04_model_predictions/CLM45_fnew_17-07-2025.csv', header=None, names=['prediction'])
JSBACH_predictions = pd.read_csv('results/04_model_predictions/JSBACH_fnew_17-07-2025.csv', header=None, names=['prediction'])
RCM_predictions = pd.read_csv('results/04_model_predictions/RCM_17-07-2025.csv')
# %% Define function to plot model predictions
def plot_model_predictions(ax, predictions, model_name, color):
    ax.plot([0, 1], [0, 1], color='k', linestyle='-', label='y=x', zorder = 0)
    df = pd.concat([tropical_sites['total_fnew'], predictions], axis=1, keys=['observed', 'predicted'])
    df = df.dropna()
    ax.scatter(df['observed'], df['predicted'], label=model_name, color=color, s= 20, zorder=1)

    evaluator = RegressionMetric(y_true=df['observed'].values, y_pred=df['predicted'].values)
    rmse = root_mean_squared_error(df['observed'], df['predicted'])
    ax.text(0.05, 0.95, f'KGE: {evaluator.kling_gupta_efficiency():.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.90, f'RMSE: {rmse:.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.set(xlabel='observed fraction of new carbon', ylabel='predicted fraction of new carbon')
    ax.set_title(model_name)

# %% Plot the predictions predictions

fig, axs = plt.subplots(3,3,figsize=(12, 12), dpi=800,constrained_layout=True)
axs = axs.flatten()
colors = plt.cm.tab10.colors
# power-law model predictions
plot_model_predictions(axs[0], powerlaw_predictions['prediction'], 'Power Law Model (Slope = -1)', colors[0])

# generalized power-law model predictions
plot_model_predictions(axs[1], generalized_powerlaw_predictions['prediction'], 'Power Law Model (Slope = $-e^{-\gamma}$)', colors[1])

# lognormal model predictions
plot_model_predictions(axs[2], lognormal_predictions['prediction'], 'Lognormal Model', colors[2])

# gamma model predictions
plot_model_predictions(axs[3], gamma_predictions['prediction'], 'Gamma Model', colors[3])


# CLM4.5 predictions
plot_model_predictions(axs[4], CLM45_predictions['prediction'], 'CLM4.5 Model', colors[4])

# JSBACH predictions
plot_model_predictions(axs[5], JSBACH_predictions['prediction'], 'JSBACH Model', colors[5])

# Reduced complexity model predictions
for i, col in enumerate(RCM_predictions.columns):
    plot_model_predictions(axs[6 + i], RCM_predictions[col], col + ' (corrected for $^{14} C$)', colors[6 + i])

# remove empty subplots
# for i in range(4 + len(RCM_predictions.columns), 8):
#     fig.delaxes(axs[i])


# %% Save the figure
fig.savefig(f'figures/model_predictions_{pd.Timestamp.now().date().strftime("%d-%m-%Y")}.png', dpi=800, bbox_inches='tight')


# %% Save each model by itself
predictions = [powerlaw_predictions['prediction'], generalized_powerlaw_predictions['prediction'], lognormal_predictions['prediction'], gamma_predictions['prediction'], CLM45_predictions['prediction'], JSBACH_predictions['prediction']] + [RCM_predictions[col] for col in RCM_predictions.columns]
titles = ['Power Law Model (Slope = -1)', 'Power Law Model (Slope = $-e^{-\gamma}$)', 'Lognormal Model', 'Gamma Model', 'CLM4.5 Model', 'JSBACH Model'] + [col + ' (corrected for $^{14} C$)' for col in RCM_predictions.columns]
colors = plt.cm.tab10.colors
for i, (prediction, title) in enumerate(zip(predictions, titles)):
    fig = plt.figure(figsize=(5, 5), dpi=600)
    ax = fig.add_subplot(111)
    plot_model_predictions(ax, prediction, title, colors[i])
    plt.savefig(f'figures/model_predictions_{title.replace(" ", "_").lower()}_{pd.Timestamp.now().date().strftime("%d-%m-%Y")}.png', dpi=600, bbox_inches='tight')