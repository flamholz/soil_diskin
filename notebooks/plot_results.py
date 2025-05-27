
# %% Import libraries
import pickle
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from models import PowerLawDisKin
from constants import INTERP_R_14C, C14_DATA
import matplotlib.pyplot as plt

#%% Load data
tropical_sites = pd.read_csv('../results/processed_balesdant_2018.csv')
powerlaw_params = pd.read_csv('../results/powerlaw_model_optimization_results.csv')
# load data
date = "20241101"
date_DAYCENT = "20241104"
date_ROTH = "20241104"
date_CLM5 = '20241206'
ts_Roth,RothC_site_age_dist,cum_RothC_site_age_dist = pickle.load(open(f'../results/analyze_balesdent/RothC_site_age_dist_{date_ROTH}.pkl','rb'))
ts_Yasso,Yasso_site_age_dist,cum_Yasso_site_age_dist = pickle.load(open(f'../results/analyze_balesdent/Yasso_site_age_dist_{date}.pkl','rb'))
ts_DAYCENT,DAYCENT_site_age_dist,cum_DAYCENT_site_age_dist = pickle.load(open(f'../results/analyze_balesdent/DAYCENT_site_age_dist_{date_DAYCENT}.pkl','rb'))
CLM_site_age_dist,cum_CLM_site_age_dist = pickle.load(open(f'../results/analyze_balesdent/CLM_site_age_dist.pkl','rb'))
ts_CLM = np.arange(0,300,1/1200)
ts_CLM5, cum_CLM5_site_age_dist = pickle.load(open(f'../results/analyze_balesdent/CLM5_site_age_dist_20000_{date_CLM5}.pkl','rb'))
# ages1, ages2,model_site_age_dist,model_site_age_dist_uncorrected,cum_model_site_age_dist,cum_model_site_age_dist_uncorrected = pickle.load(open(f'../results/analyze_balesdent/CESM_IPSL_MRI_site_age_dist_{date}.pkl','rb'))



#%%

ts = np.logspace(-2,5,1000,base=10)
predictions = []
for i, row in tropical_sites.iterrows():
    if np.isnan(powerlaw_params.loc[i,'params']).sum() != 0:
        predictions.append(np.nan)
        continue
    model = PowerLawDisKin(*powerlaw_params.loc[i,'params'])
    predict_pA = np.vectorize(model.radiocarbon_age_integrand, otypes=[np.float64])
    cdf = interp1d(ts[:-1], cumulative_trapezoid(predict_pA(ts),ts))
    predictions.append(cdf(row['Duration_labeling']))

fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
plt.scatter(tropical_sites['total_fnew'],predictions)
ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='y=x')

# calculate the R^2 value
from sklearn.metrics import r2_score
mask = ~np.isnan(tropical_sites['total_fnew']) & ~np.isnan(predictions)
r2 = r2_score(tropical_sites['total_fnew'][mask], np.stack(predictions)[mask])
print(f'R^2: {r2:.3f}')

# calculate correlation coefficient
from scipy.stats import pearsonr
corr, _ = pearsonr(tropical_sites['total_fnew'][mask], np.stack(predictions)[mask])
print(f'Pearson correlation coefficient: {corr:.3f}')

# calculate spearman correlation
from scipy.stats import spearmanr
spearman_corr, _ = spearmanr(tropical_sites['total_fnew'][mask], np.stack(predictions)[mask])
print(f'Spearman correlation coefficient: {spearman_corr:.3f}')



# %%

model = PowerLawDisKin(*np.stack(powerlaw_params.dropna()['params'].values).mean(axis=0))
predictions = []
for i, row in tropical_sites.iterrows():
    if np.isnan(powerlaw_params.loc[i,'params']).sum() != 0:
        predictions.append(np.nan)
        continue
    
    predict_pA = np.vectorize(model.radiocarbon_age_integrand, otypes=[np.float64])
    cdf = interp1d(ts[:-1], cumulative_trapezoid(predict_pA(ts),ts))
    predictions.append(cdf(row['Duration_labeling']))

fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
plt.scatter(tropical_sites['total_fnew'],predictions)
ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='y=x')
mask = ~np.isnan(tropical_sites['total_fnew']) & ~np.isnan(predictions)
r2 = r2_score(tropical_sites['total_fnew'][mask], np.stack(predictions)[mask])
print(f'R^2: {r2:.3f}')
corr, _ = pearsonr(tropical_sites['total_fnew'][mask], np.stack(predictions)[mask])
print(f'Pearson correlation coefficient: {corr:.3f}')

#%%
initial_guess = [0.5, 10000]
# optimize the parameters using a simple optimization method

    # Minimize the objective function for each site
res = minimize(objective_function, initial_guess, args=(merged_site_data.mean(),), method='Nelder-Mead')

model = PowerLawDisKin(*res.x)

predictions = []
for i, row in tropical_sites.iterrows():    
    predict_pA = np.vectorize(model.radiocarbon_age_integrand, otypes=[np.float64])
    cdf = interp1d(ts[:-1], cumulative_trapezoid(predict_pA(ts),ts))
    predictions.append(cdf(row['Duration_labeling']))

fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
plt.scatter(tropical_sites['total_fnew'],predictions)
ax.plot([0, 1], [0, 1], color='red', linestyle='--', label='y=x')
mask = ~np.isnan(tropical_sites['total_fnew']) & ~np.isnan(predictions)
r2 = r2_score(tropical_sites['total_fnew'][mask], np.stack(predictions)[mask])
print(f'R^2: {r2:.3f}')
corr, _ = pearsonr(tropical_sites['total_fnew'][mask], np.stack(predictions)[mask])
print(f'Pearson correlation coefficient: {corr:.3f}')

#%%
# extract site GPP
GPP_df = GPP.to_dataframe().dropna()['GPP_mean']
kdt = KDTree(np.stack(GPP_df.index.to_numpy()),leaf_size=30, metric='euclidean')
distance,ind = kdt.query(unique_tropical_sites[['Latitude','Longitude']].values,k=1)
site_GPP = GPP_df.values[ind]
site_GPP[distance>1] = np.nan
site_GPP = site_GPP.squeeze()

site_SOC = tropical_sites.groupby(['Latitude','Longitude'])[['Cref_0-100estim']].mean()['Cref_0-100estim']*1e3 # kgC/m2 to gC/m2
merged_site_data = pd.DataFrame([site_1m_14C,site_GPP,site_SOC.values],index = ['14C','GPP','SOC'],columns= unique_tropical_sites.set_index(['Latitude','Longitude']).index).T
merged_site_data['turnover'] = merged_site_data['SOC']/merged_site_data['GPP']
merged_site_data['14C_ratio'] = 1+merged_site_data['14C']/1e3

# TODO
# 1. Think about generating a model for each site specifically based on the turnover and 14C data there
# 2. Think about different averaging methods for the model - averaging turnover and 14C data, or averaging the model parameters or the predictions of the models
# 3. Use NPP instead of GPP - maybe use several different models to get uncertainty, and also just use 1/2 GPP for the NPP
# %%

