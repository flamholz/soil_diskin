import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

# Read the age scan data and the radiocarbon data
# the Mathematica script must be run first to generate the age scan
# (notebooks/03b_calibrate_lognormal_model.wls)
site_data = pd.read_csv('results/tropical_sites_14C_turnover.csv')
age_scan = pd.read_csv('results/03_calibrate_models/03b_lognormal_model_age_scan.csv', header=None)
ages = 10**np.arange(3,5.5,(5.5-3)/100)
ages = np.concatenate([ages, np.array([10**5.5])])

predictions = pd.read_csv('results/03b_lognormal_site_parameters.csv', header=None)
# predict the mean age for each site based on the calibration curve
preds = []
for site in site_data.index:
    calcurve = interp1d(age_scan.iloc[site], ages)
    age_pred = calcurve(site_data.loc[site, 'fm'])
    preds.append(age_pred)

preds = np.array(preds)

# save the predictions
site_data['pred'] = preds
site_data.to_csv('results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv', index=False)
