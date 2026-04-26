#%%
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

# The aim of this script is to use a calibration curve generated in Mathematica
# between radiocarbon fraction (fm) and age for the lognormal DisKin model
# to predict the mean age for each site based on its measured fm value

# Read the age scan data and the radiocarbon data
# the Mathematica script must be run first to generate the age scan
# (notebooks/03b_lognormal_age_scan.wls)
site_data = pd.read_csv('results/all_sites_14C_turnover.csv')
age_scan = pd.read_csv('results/03_calibrate_models/03b_lognormal_model_age_scan.csv', index_col=0)
age_scan_05 = pd.read_csv('results/03_calibrate_models/03b_lognormal_model_age_scan_05.csv', index_col=0)
age_scan_95 = pd.read_csv('results/03_calibrate_models/03b_lognormal_model_age_scan_95.csv', index_col=0)

backfilled_sites = site_data[site_data['turnover_q05'].notna() & site_data['turnover_q95'].notna()]
#%%
# Extract ages from the age_scan column headers (already in log10 space)
ages = age_scan.columns.astype(float).values

# predict the mean age for each site by interpolating the age scan 
# calibration curve generated in Mathematica
def get_prediction(df, ascan):
    '''
    Interpolate the calibration curve for the relation between fm and age
    
    Inputs:
    df: DataFrame with 'fm' column
    ascan: DataFrame with age scan data, rows are sites, columns are ages, values are fm predictions
    '''

    preds = []
    for i, site in enumerate(df.index):
        # Get the calibration curve for this site (ages are the column values)
        site_ages = ascan.columns.astype(float).values
        site_fms = ascan.iloc[i].values
        smoothed = lowess(site_fms, site_ages, frac=0.2)  # smooth the calibration curve using LOWESS
        calcurve = interp1d(smoothed[:,1], smoothed[:,0], fill_value="extrapolate")  # interpolate the smoothed curve
        age_pred = calcurve(df.loc[site, 'fm'])  # predict the age based on the fm value
        preds.append(age_pred)
    return np.array(preds)

site_data['pred'] = get_prediction(site_data, age_scan)
backfilled_sites['pred_05'] = get_prediction(backfilled_sites, age_scan_05)
backfilled_sites['pred_95'] = get_prediction(backfilled_sites, age_scan_95)

merged_df = pd.merge(site_data, backfilled_sites[['pred_05', 'pred_95']], left_index=True, right_index=True, how='left')
# save the predictions
# TODO: give this column a more descriptive name
merged_df.to_csv('results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv', index=False)

# %%
