
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
ts = np.logspace(-2,5,1000,base=10)

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


#%% Plot the model predictions for each model

fig, axs = plt.subplots(2,3,figsize=(12, 8), dpi=600,constrained_layout=True)
axs = axs.flatten()

predictions = []
for i, row in tropical_sites.iterrows():
    if np.isnan(powerlaw_params.loc[i,['tau_0','tau_inf']]).sum() != 0:
        predictions.append(np.nan)
        continue
    model = PowerLawDisKin(*powerlaw_params.loc[i,['tau_0','tau_inf']])
    predict_pA = np.vectorize(model.radiocarbon_age_integrand, otypes=[np.float64])
    cdf = interp1d(ts[:-1], cumulative_trapezoid(predict_pA(ts),ts))
    predictions.append(cdf(row['Duration_labeling']))


axs[0].plot([0, 1], [0, 1], color='k', linestyle='-', label='y=x')
mask = ~np.isnan(tropical_sites['total_fnew']) & ~np.isnan(predictions)
r2 = r2_score(tropical_sites['total_fnew'][mask], np.stack(predictions)[mask])
resid = (tropical_sites['total_fnew'][mask] - np.stack(predictions)[mask])**2
axs[0].scatter(tropical_sites['total_fnew'],predictions,s=resid*100)
axs[0].text(0.05, 0.95, f'$R^2$: {r2:.3f}', transform=axs[0].transAxes, fontsize=10, verticalalignment='top')
axs[0].set(xlabel='Total fraction of new C', ylabel='Predicted fraction of new C')
axs[0].set_title('Power Law DisKin Model\nper site')


model = PowerLawDisKin(*np.stack(powerlaw_params.dropna()[['tau_0','tau_inf']].values).mean(axis=0))
predictions = []
for i, row in tropical_sites.iterrows():
    if np.isnan(powerlaw_params.loc[i,['tau_0','tau_inf']]).sum() != 0:
        predictions.append(np.nan)
        continue
    
    predict_pA = np.vectorize(model.radiocarbon_age_integrand, otypes=[np.float64])
    cdf = interp1d(ts[:-1], cumulative_trapezoid(predict_pA(ts),ts))
    predictions.append(cdf(row['Duration_labeling']))



axs[1].plot([0, 1], [0, 1], color='k', linestyle='-', label='y=x')
mask = ~np.isnan(tropical_sites['total_fnew']) & ~np.isnan(predictions)
r2 = r2_score(tropical_sites['total_fnew'][mask], np.stack(predictions)[mask])
resid = (tropical_sites['total_fnew'][mask] - np.stack(predictions)[mask])**2
axs[1].scatter(tropical_sites['total_fnew'],predictions,s=resid*100)
axs[1].text(0.05, 0.95, f'$R^2$: {r2:.3f}', transform=axs[1].transAxes, fontsize=10, verticalalignment='top')
axs[1].set(xlabel='Total fraction of new C', ylabel='Predicted fraction of new C')
axs[1].set_title('Power Law DisKin Model\nmean parameters')

#%% Plot the compartmental model predictions
unique_sites = tropical_sites.groupby(['Latitude','Longitude']).mean().reset_index()
tss = [ts_Roth, ts_Yasso, ts_DAYCENT, ts_CLM5]
pAs = [RothC_site_age_dist, Yasso_site_age_dist, DAYCENT_site_age_dist]
cum_pAs = [np.stack([cumulative_trapezoid(pA[i,:],t)/cumulative_trapezoid(pA[i,:],t)[-1] for i in range(pA.shape[0])]) for (pA,t) in zip(pAs,tss[:-1])] + [cum_CLM5_site_age_dist]
titles = ['RothC', 'Yasso', 'DAYCENT', 'CLM5']

for i, (ts, pA) in enumerate(zip(tss, cum_pAs)):
# for i in [0]:
    predictions = []
    for j, row in unique_sites.iterrows():
        if tss[i].shape[0] != pA.shape[1]:
            interp = interp1d(tss[i][:-1],pA[j,:], bounds_error=False, fill_value='extrapolate')
        else:
            interp = interp1d(tss[i],pA[j,:], bounds_error=False, fill_value='extrapolate')
        sites = tropical_sites[(tropical_sites['Longitude'] == row['Longitude']) & (tropical_sites['Latitude'] == row['Latitude'])]
        
        sites.loc[:, 'prediction'] = np.stack(sites.apply(lambda x: interp(x['Duration_labeling']),axis=1).values)
        predictions.append(sites)
        # for _, site_row in sites.iterrows():
        #     pred_row = pd.Series([site_row['Latitude'], site_row['Longitude'], site_row['Duration_labeling'],interp(site_row['Duration_labeling'])], index=['Latitude', 'Longitude', 'Duration_labeling','prediction'])
        #     predictions.append(pred_row)

    # sites_with_prediction = tropical_sites.merge(pd.concat(predictions,axis=1).T,left_on=['Latitude', 'Longitude', 'Duration_labeling'], right_on=['Latitude', 'Longitude', 'Duration_labeling'], how='left')               
    sites_with_prediction = tropical_sites.merge(pd.concat(predictions)[['Latitude', 'Longitude', 'Duration_labeling','prediction']],left_on=['Latitude', 'Longitude', 'Duration_labeling'], right_on=['Latitude', 'Longitude', 'Duration_labeling'], how='left')               
    
    axs[i+2].plot([0, 1], [0, 1], color='k', linestyle='-', label='y=x')
    # mask = ~np.isnan(tropical_sites['total_fnew']) & ~np.isnan(sites_with_prediction['prediction'])
    mask = sites_with_prediction['total_fnew'].notna() & sites_with_prediction['prediction'].notna()
    # r2 = r2_score(tropical_sites['total_fnew'][mask], np.stack(sites_with_prediction['prediction'])[mask])
    r2 = r2_score(sites_with_prediction['total_fnew'][mask], sites_with_prediction['prediction'][mask])
    resid = (sites_with_prediction['total_fnew'][mask] - sites_with_prediction['prediction'][mask])**2
    axs[i+2].scatter(sites_with_prediction['total_fnew'][mask], sites_with_prediction['prediction'][mask],s = resid*100)
    # r2 = r2_score(tropical_sites['total_fnew'], np.stack(sites_with_prediction['prediction']))
    axs[i+2].text(0.05, 0.95, f'$R^2$: {r2:.3f}', transform=axs[i+2].transAxes, fontsize=10, verticalalignment='top')
    axs[i+2].set(xlabel='Total fraction of new C', ylabel='Predicted fraction of new C')
    axs[i+2].set_title(f'{titles[i]} Model')
fig
#%%

# do linear regression to the predictions vs the total_fnew using stat
from sklearn.linear_model import LinearRegression
mask = sites_with_prediction['total_fnew'].notna() & sites_with_prediction['prediction'].notna()
X = sites_with_prediction['total_fnew'][mask].values.reshape(-1, 1)
y = sites_with_prediction['prediction'][mask].values.reshape(-1, 1)
model = LinearRegression()
p = model.fit(X, y)
# get the slope and intercept
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
