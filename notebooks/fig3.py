import pandas as pd
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from permetrics.regression import RegressionMetric

#%% Load the site data
tropical_sites = pd.read_csv('results/processed_balesdent_2018.csv')

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
    ax.plot([0, 1], [0, 1], color='k', linestyle='-', label='y=x')
    ax.scatter(tropical_sites['total_fnew'], predictions, label=model_name, color=color)

    evaluator = RegressionMetric(y_true=tropical_sites['total_fnew'].values, y_pred=predictions.values)
    rmse = root_mean_squared_error(tropical_sites['total_fnew'], predictions)
    ax.text(0.05, 0.95, f'KGE: {evaluator.kling_gupta_efficiency():.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.90, f'RMSE: {rmse:.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.set(xlabel='observed', ylabel='predicted')
    ax.set_title(model_name) 

# %% Plot the predictions predictions
colors = plt.cm.tab10.colors


fig, axs = plt.subplots(2,4,figsize=(15, 8), dpi=600,constrained_layout=True)
axs = axs.flatten()

# power-law model predictions
plot_model_predictions(axs[0], powerlaw_predictions['prediction'], 'Power Law Model', colors[0])

# lognormal model predictions
plot_model_predictions(axs[1], lognormal_predictions['prediction'], 'Lognormal Model', colors[1])

# gamma model predictions
plot_model_predictions(axs[2], gamma_predictions['prediction'], 'Gamma Model', colors[2])

# CLM4.5 predictions
plot_model_predictions(axs[3], CLM45_predictions['prediction'], 'CLM4.5 Model', colors[3])

# JSBACH predictions
plot_model_predictions(axs[4], JSBACH_predictions['prediction'], 'JSBACH Model', colors[4])

# Reduced complexity model predictions
for i, col in enumerate(RCM_predictions.columns):
    plot_model_predictions(axs[5 + i], RCM_predictions[col], col +  ' (corrected for $^{14} C$)', colors[5 + i])

# # remove empty subplots
# for i in range(4 + len(RCM_predictions.columns), 8):
#     fig.delaxes(axs[i])

# add labels to each subplot
for i, ax in enumerate(axs):
    # uppercase letters
    ax.text(0, 1.05, chr(65 + i), transform=ax.transAxes, fontsize=14, va='top', ha='right')

# %% Save the figure
out_fname = f'figures/fig3.png'
plt.savefig(out_fname, dpi=600, bbox_inches='tight')