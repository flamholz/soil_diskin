
#%%

from models import PowerLawDisKin
from constants import INTERP_R_14C, C14_DATA
from scipy.integrate import quad
from scipy.optimize import minimize

# optimize the two parameters of the model to match the turnover and 14C data

def objective_function(params, merged_site_data):
    """
    Objective function to minimize the difference between the model predictions and the observed data.
    """
    a, b = params
    model = PowerLawDisKin(a, b)
    
    # Calculate the predicted 14C ratio and turnover
    predicted_14C_ratio = quad(model.radiocarbon_age_integrand, 0, np.inf)[0]
    
    # Calculate the difference between the predicted and observed data
    diff_14C = np.nansum((predicted_14C_ratio - merged_site_data['fm'])**2)
    diff_turnover = np.nansum((model.T - merged_site_data['turnover'])**2)
    
    return diff_14C + diff_turnover

# initial guess for the parameters
initial_guess = [0.5, 10000]
# optimize the parameters using a simple optimization method

result = []
from tqdm import tqdm
for i, row in tqdm(merged_site_data.iterrows(), total=len(merged_site_data)):
    row['turnover'] = row['turnover'] *2 
    # Check if the row has valid data for 14C and turnover
    if np.isnan(row['14C']) or np.isnan(row['turnover']):
        result.append([np.nan, np.nan])
        continue
    
    # Minimize the objective function for each site
    res = minimize(objective_function, initial_guess, args=(row,), method='Nelder-Mead')
    
    # Append the optimized parameters to the result list
    result.append([res.x,res.fun])

result_df = pd.DataFrame(result,columns=['params','objective_value'],index=merged_site_data.index)
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

