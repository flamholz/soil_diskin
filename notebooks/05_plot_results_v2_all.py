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
tropical_sites = pd.read_csv('results/processed_balesdent_2018.csv')

#%% Load the predictions
# powerlaw_predictions = pd.read_csv('results/04_model_predictions/power_law_16-07-2025.csv',header=None, names=['prediction'])
powerlaw_predictions = pd.read_csv(f'results/04_model_predictions/power_law.csv',header=None, names=['prediction'])
lognormal_predictions = pd.read_csv(f'results/04_model_predictions/lognormal.csv',header=None, names=['prediction'])
CLM45_predictions = pd.read_csv(f'results/04_model_predictions/CLM45_fnew.csv', header=None, names=['prediction'])
JSBACH_predictions = pd.read_csv(f'results/04_model_predictions/JSBACH_fnew.csv', header=None, names=['prediction'])
RCM_predictions = pd.read_csv(f'results/04_model_predictions/RCM.csv')
# %% Define function to plot model predictions
def plot_model_predictions(ax, predictions, model_name):
    ax.plot([0, 1], [0, 1], color='k', linestyle='-', label='y=x')
    ax.scatter(tropical_sites['total_fnew'], predictions, label=model_name)

    evaluator = RegressionMetric(y_true=tropical_sites['total_fnew'].values, y_pred=predictions.values)
    rmse = root_mean_squared_error(tropical_sites['total_fnew'], predictions)
    ax.text(0.05, 0.95, f'KGE: {evaluator.kling_gupta_efficiency():.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.90, f'RMSE: {rmse:.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.set(xlabel='observed', ylabel='predicted')
    ax.set_title(model_name) 

fig, axs = plt.subplots(dpi=300)
plot_model_predictions(axs, powerlaw_predictions['prediction'], 'Power Law DisKin Model')
# %% Plot the predictions predictions

fig, axs = plt.subplots(2,4,figsize=(15, 8), dpi=600,constrained_layout=True)
axs = axs.flatten()

# power-law model predictions
plot_model_predictions(axs[0], powerlaw_predictions['prediction'], 'Power Law DisKin Model')

# lognormal model predictions
plot_model_predictions(axs[1], lognormal_predictions['prediction'], 'Lognormal Model')

# CLM4.5 predictions
plot_model_predictions(axs[2], CLM45_predictions['prediction'], 'CLM4.5 Model')

# JSBACH predictions
plot_model_predictions(axs[3], JSBACH_predictions['prediction'], 'JSBACH Model')

# Reduced complexity model predictions
for i, col in enumerate(RCM_predictions.columns):
    plot_model_predictions(axs[4 + i], RCM_predictions[col], col)

# remove empty subplots
for i in range(4 + len(RCM_predictions.columns), 8):
    fig.delaxes(axs[i])

# %% Save the figure
out_fname = f'figures/model_predictions.png'
plt.savefig(out_fname, dpi=600, bbox_inches='tight')