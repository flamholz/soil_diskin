
# %%
import pickle
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
import pandas as pd
import numpy as np
final_data = pd.read_csv('../results/processed_balesdant_2018.csv')
import seaborn as sns
import matplotlib.pyplot as plt
fig,ax = plt.subplots(dpi=300)

ax.semilogx(final_data['Duration_labeling'],final_data['total_fnew'],lw=0,marker='o',color='k')

# run a lowess  regression to the data
# from statsmodels.nonparametric.smoothers_lowess import lowess
# lowess_data = lowess(final_data['total_fnew'],final_data['Duration_labeling'], frac=.5)
# ax.plot(lowess_data[:,0],lowess_data[:,1],color='k',lw=1.5)

mat_data = pd.read_csv('../results/diskin_predictions/shortpas_tropics_mathematica_20241101_0.1_200_000.csv',header=None)
ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'disordered kinetics\nmodel')

mat_data = pd.read_csv('../results/diskin_predictions/pas_tropics_powerlaw_mathematica_20241104_100000_0.1.csv',header=None)
ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Royalblue',label=f'power law model')
ax.set(xlim=[0.1,100_000],ylim=[0,1],)
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
ax.set(ylim=(0,1.01),xlabel='Age (years)',ylabel='Cummulative fraction of soil carbon')
N_CLM5 = (~np.isnan(cum_CLM5_site_age_dist.sum(axis=1))).sum()
CLM5_se = np.nanstd(cum_CLM5_site_age_dist,axis=0)/np.sqrt(N_CLM5-1)
CLM5_mean = np.nanmean(cum_CLM5_site_age_dist,axis=0)
ax.plot(ts_CLM5[:-1],CLM5_mean,label='CLM5',c='C06')
ax.fill_between(ts_CLM5[:-1],CLM5_mean+1.96*CLM5_se,CLM5_mean-1.96*CLM5_se,alpha=.3,color='C06',lw=0)

DAYCENT_se = cum_DAYCENT_site_age_dist.std(axis=1)/np.sqrt(cum_DAYCENT_site_age_dist.shape[1]-1)
DAYCENT_mean = cum_DAYCENT_site_age_dist.mean(axis=1)
ax.plot(ts_DAYCENT,DAYCENT_mean,label='DAYCENT',c='C05')
ax.fill_between(ts_DAYCENT,DAYCENT_mean+1.96*DAYCENT_se,DAYCENT_mean-1.96*DAYCENT_se,alpha=.3,color='C05',lw=0)

Yasso_se = cum_Yasso_site_age_dist.std(axis=1)/np.sqrt(cum_Yasso_site_age_dist.shape[1]-1)
Yasso_mean = cum_Yasso_site_age_dist.mean(axis=1)
ax.plot(ts_Yasso,Yasso_mean,label='Yasso',c='C04')
ax.fill_between(ts_Yasso,Yasso_mean+1.96*Yasso_se,Yasso_mean-1.96*Yasso_se,alpha=.3,color='C04',lw=0)

roth_se = cum_RothC_site_age_dist.std(axis=1)/np.sqrt(cum_RothC_site_age_dist.shape[1]-1)
roth_mean = cum_RothC_site_age_dist.mean(axis=1)
ax.plot(ts_Roth,roth_mean,label='RothC',c='C03')
ax.fill_between(ts_Roth,roth_mean+1.96*roth_se,roth_mean-1.96*roth_se,alpha=.3,color='C03',lw=0)

plt.legend()
# %%
powerlaw_data = pd.read_csv('../results/diskin_predictions/pas_tropics_powerlaw_mathematica_20241104_100000_0.1.csv',header=None)
powerlaw_data['pred'] = np.cumsum(powerlaw_data[1])/powerlaw_data[1].sum()
powerlaw_pred = powerlaw_data.merge(final_data,left_on=0,right_on='Duration_labeling')

lgnorm_data = pd.read_csv('../results/diskin_predictions/shortpas_tropics_mathematica_20241101_0.1_200_000.csv',header=None)
lgnorm_data['pred'] = np.cumsum(lgnorm_data[1])/lgnorm_data[1].sum()
lgnorm_pred = lgnorm_data.merge(final_data,left_on=0,right_on='Duration_labeling')

plt.scatter(powerlaw_pred['total_fnew'],powerlaw_pred['pred'],label='power law model',color='Royalblue')

# calculate r2 for the power law model
from sklearn.metrics import r2_score
r2_powerlaw = r2_score(powerlaw_pred['total_fnew'],powerlaw_pred['pred'])
print(f'R2 for power law model: {r2_powerlaw:.3f}')

plt.scatter(lgnorm_pred['total_fnew'],lgnorm_pred['pred'],label='disordered kinetics model',color='Indianred')
# calculate r2 for the disordered kinetics model
r2_lgnorm = r2_score(lgnorm_pred['total_fnew'],lgnorm_pred['pred'])
print(f'R2 for disordered kinetics model: {r2_lgnorm:.3f}')

# CLM5_df = pd.DataFrame([ts_CLM5[:-1].round(0),CLM5_mean]).T.astype(float)

from scipy.interpolate import interp1d

# np.interp(ts_CLM5[:-1],CLM5_mean,final_data['Duration_labeling'])
CLM5_pred = final_data.copy()
CLM5_pred['pred'] = interp1d(ts_CLM5[:-1],CLM5_mean,fill_value='extrapolate')(final_data['Duration_labeling'])
plt.scatter(CLM5_pred['total_fnew'],CLM5_pred['pred'],label='CLM5',color='C06')
# calculate r2 for the CLM5 model
r2_CLM5 = r2_score(CLM5_pred['total_fnew'],CLM5_pred['pred'])
print(f'R2 for CLM5 model: {r2_CLM5:.3f}')

# do the same for DAYCENT
DAYCENT_pred = final_data.copy()
DAYCENT_pred['pred'] = interp1d(ts_DAYCENT,DAYCENT_mean,fill_value='extrapolate')(final_data['Duration_labeling'])
plt.scatter(DAYCENT_pred['total_fnew'],DAYCENT_pred['pred'],label='DAYCENT',color='C05')
# calculate r2 for the DAYCENT model
r2_DAYCENT = r2_score(DAYCENT_pred['total_fnew'],DAYCENT_pred['pred'])
print(f'R2 for DAYCENT model: {r2_DAYCENT:.3f}')

plt.legend(['power law model','disordered kinetics model','CLM5','DAYCENT'])
plt.plot([0,1],[0,1],color='k',lw=1.5)
plt.xlim(0,1)
plt.ylim(0,1)
# %%



#%%
import matplotlib.pyplot as plt
ts = np.logspace(-2,5,1000,base=10)
from scipy.integrate import  cumulative_trapezoid
from scipy.interpolate import interp1d
predictions = []
for i, row in tropical_sites.iterrows():
    if np.isnan(result_df.loc[i,'params']).sum() != 0:
        predictions.append(np.nan)
        continue
    model = PowerLawDisKin(*result_df.loc[i,'params'])
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

model = PowerLawDisKin(*np.stack(result_df.dropna()['params'].values).mean(axis=0))
predictions = []
for i, row in tropical_sites.iterrows():
    if np.isnan(result_df.loc[i,'params']).sum() != 0:
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

