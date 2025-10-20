# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from permetrics.regression import RegressionMetric
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.interpolate import interp1d

# %%
ages = 10**np.arange(3,5.5,(5.5-3)/100)
ages = np.concatenate([ages, np.array([10**5.5])])
age_turnover = pd.read_csv('results/all_sites_14C_turnover.csv')
ratios = np.array(['0.50', '0.67', '1', '1.50', '2'])

def get_pred(age_scan):
    preds = []
    for site in age_turnover.index:
        calcurve = interp1d(age_scan.iloc[site], ages)
        age_pred = calcurve(age_turnover.loc[site, 'fm'])
        preds.append(age_pred)

    preds = np.array(preds)
    return preds

results = pd.concat([pd.Series(get_pred(pd.read_csv(f'results/06_sensitivity_analysis/06a_lognormal_age_scan{a}.csv', header=None))) for a in ratios], axis=1, keys=ratios)

results.to_csv('results/06_sensitivity_analysis/lognormal_age_predictions.csv', index=False)


