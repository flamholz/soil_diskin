# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

# %%
# Load libraries
from notebooks.models import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from permetrics.regression import RegressionMetric
from scipy.signal import convolve
from scipy.interpolate import interp1d

# %%
# Read the raw data and calculate the ratio in C stocks between the reference and the experiment sites.
raw_site_data = pd.read_excel('data/balesdent_2018/balesdent_2018_raw.xlsx', skiprows=7).query('MAT_C>17.0 & PANN_mm>1000.0 & `P to PET ratio` >0.8')
site_data = pd.read_csv('results/processed_balesdent_2018.csv')
J_ratio = raw_site_data['Cref_0-100estim'] / raw_site_data['Ctotal_0-100estim']


# %% [markdown]
# #### Test vegetation effects

# %%
# load the powerlaw and gamma model parameters and lognormal cdfs
powerlaw_params = pd.read_csv('results/03_calibrate_models/powerlaw_model_optimization_results.csv')
gamma_params = pd.read_csv('results/03_calibrate_models/gamma_model_optimization_results.csv')
lognormal_cdfs = pd.read_csv('results/06_sensitivity_analysis/06a_lognormal_cdfs_1.csv')
lognormal_cdfs.columns = lognormal_cdfs.columns.astype(float)

# %%
# Generate an age distribution for vegetation carbon assuming an exponential age distribution with mean residence time of 450/50 = 9 years
ts = np.arange(0, 10000000, 0.1)  # time in years

ts_veg = np.arange(0, 1000, 0.1)  # time in years
veg_age = stats.expon(scale = 450/50) # flux is NPP - 50 GtC yr-1 and stocks are 450 GtC
veg_p = veg_age.pdf(ts_veg)


def make_predictions(ts, veg_p, par1, par2, label_time, model, pA=None):
    '''
    Make predictions using the specified model and parameters.

    Parameters:
    ts : array-like
        Time steps for the model.
    veg_p : array-like
        Vegetation age distribution.
    par1 : float
        First parameter for the model.
    par2 : float
        Second parameter for the model.
    label_time : array-like
        Times at which to evaluate fnew.
    model : class
        Model class to use (e.g., PowerLawModel, GammaModel).
    pA : pd.DataFrame, optional
        Precomputed age distribution. If provided, it will be used instead of generating a new one. Defaults to None.
    
    Returns:
    conv_fnew : array-like
        fnew predictions with convolution.
    no_conv_fnew : array-like
        fnew predictions without convolution.
    '''
    
    if pA is not None:
        t = pA.index.values
        model_pa = np.diff(interp1d(t, pA.values)(ts))
        ts = ts[:-1]
    else:
        model = model(par1, par2)
        model_pa = model.pA(ts)
    conv_p = convolve(model_pa, veg_p, mode='full')[:len(ts)]
    conv_cdf = cumulative_trapezoid(conv_p, ts) / cumulative_trapezoid(conv_p, ts)[-1]
    conv_fnew = interp1d(ts[1:], conv_cdf, bounds_error=False, fill_value=0)(label_time)
    
    no_conv_cdf = cumulative_trapezoid(model_pa, ts) / cumulative_trapezoid(model_pa, ts)[-1]
    no_conv_fnew = interp1d(ts[1:], no_conv_cdf, bounds_error=False, fill_value=0)(label_time)
    
    return conv_fnew, no_conv_fnew


# %%
# # Power-law model predictions
for i, row in tqdm(powerlaw_params.iterrows()):
    conv_fnew, model_cdf = make_predictions(ts, veg_p, row.loc['tau_0'], row.loc['tau_inf'], site_data.loc[i, 'Duration_labeling'], PowerLawDisKin)
    site_data.loc[i, 'pred_conv'] = conv_fnew
    site_data.loc[i, 'pred_no_conv'] = model_cdf

# Gamma model predictions
for i, row in tqdm(gamma_params.iterrows()):
    conv_fnew, model_cdf = make_predictions(ts, veg_p, row.loc['a'], row.loc['b'], site_data.loc[i, 'Duration_labeling'], GammaDisKin)
    site_data.loc[i, 'pred_conv_gamma'] = conv_fnew
    site_data.loc[i, 'pred_no_conv_gamma'] = model_cdf

# %%
# Lognormal model predictions
ts_ln = np.arange(0.1, 1e5, 0.1)  # time in years
for i, row in tqdm(lognormal_cdfs.iterrows()):
    conv_fnew, model_cdf = make_predictions(ts_ln, veg_p, 0, 0, site_data.loc[i, 'Duration_labeling'], None, lognormal_cdfs.iloc[i])
    site_data.loc[i, 'pred_conv_lognorm'] = conv_fnew
    site_data.loc[i, 'pred_no_conv_lognorm'] = model_cdf

# Save results
site_data.to_csv('results/06_sensitivity_analysis/06c_model_predictions_veg_effects.csv', index=False)