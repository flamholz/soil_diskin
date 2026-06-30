# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

import numpy as np
import pandas as pd
import rasterio
from scipy import stats

# WorldClim BIO variable names (BIO1–BIO19)
WORLDCLIM_VARS = {
    1:  'mean_annual_temp',
    2:  'mean_diurnal_temp_range',
    3:  'isothermality',
    4:  'temp_seasonality',
    5:  'max_temp_warmest_month',
    6:  'min_temp_coldest_month',
    7:  'temp_annual_range',
    8:  'mean_temp_wettest_quarter',
    9:  'mean_temp_driest_quarter',
    10: 'mean_temp_warmest_quarter',
    11: 'mean_temp_coldest_quarter',
    12: 'annual_precip',
    13: 'precip_wettest_month',
    14: 'precip_driest_month',
    15: 'precip_seasonality',
    16: 'precip_wettest_quarter',
    17: 'precip_driest_quarter',
    18: 'precip_warmest_quarter',
    19: 'precip_coldest_quarter',
}

def lognormal_mu_sigma(age, tau):
    """Derive lognormal mu and sigma from mean age and turnover time."""
    sigma = np.sqrt(np.log(age / tau))
    mu = -np.log(np.sqrt(tau**3 / age))
    return mu, sigma


def load_lognormal_params(csv_path):
    """Load lognormal calibration output and add derived mu and sigma columns."""
    df = pd.read_csv(csv_path)
    df['mu'], df['sigma'] = lognormal_mu_sigma(df['pred'].values, df['turnover'].values)
    return df


# Model parameter definitions: (label, csv_path_or_df, [param_cols])
# For lognormal we pass the DataFrame directly after deriving mu/sigma.
MODELS = [
    ('lognormal',
     'results/03_calibrate_models/03b_lognormal_predictions_calcurve_python.csv',
     ['mu', 'sigma']),
    ('powerlaw_alpha1',
     'results/03_calibrate_models/powerlaw_model_optimization_results.csv',
     ['t_min', 't_max']),
    ('powerlaw_alpha_exp_gamma',
     'results/03_calibrate_models/general_powerlaw_model_optimization_results.csv',
     ['t_min', 't_max']),
    ('powerlaw_alpha_exp_gamma_half',
     'results/03_calibrate_models/general_powerlaw_model_optimization_results_beta_half.csv',
     ['t_min', 't_max']),
]

# %%
site_data = pd.read_csv('results/all_sites_14C_turnover.csv')
coords = list(zip(site_data['Longitude'], site_data['Latitude']))

# %%
for bio_num, col_name in WORLDCLIM_VARS.items():
    tif_path = f'data/worldclim/wc2.1_10m_bio_{bio_num}.tif'
    with rasterio.open(tif_path) as src:
        values = [v[0] for v in src.sample(coords)]
    nodata = src.nodata
    arr = np.array(values, dtype=float)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    site_data[col_name] = arr

site_data.to_csv('results/figS2_calcs.csv', index=False)
print(f'Saved {len(site_data)} sites with {len(WORLDCLIM_VARS)} WorldClim variables.')

# %%
bio_cols = list(WORLDCLIM_VARS.values())

def correlations_with_worldclim(param_series, site_df, bio_cols):
    """Return DataFrame of Pearson and Spearman r for one parameter vs all WorldClim vars."""
    rows = []
    for col in bio_cols:
        mask = param_series.notna() & site_df[col].notna()
        x = param_series[mask].values
        y = site_df.loc[mask, col].values
        if len(x) < 3:
            rows.append({'worldclim_var': col, 'pearson_r': np.nan, 'pearson_p': np.nan,
                     'spearman_r': np.nan, 'spearman_p': np.nan})
            continue
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        rows.append({'worldclim_var': col, 'pearson_r': pearson_r, 'pearson_p': pearson_p,
                     'spearman_r': spearman_r, 'spearman_p': spearman_p})
    return pd.DataFrame(rows).set_index('worldclim_var')


for model_label, csv_path, param_cols in MODELS:
    if model_label == 'lognormal':
        params = load_lognormal_params(csv_path)
    else:
        params = pd.read_csv(csv_path)
    summary_frames = []
    for param in param_cols:
        corr_df = correlations_with_worldclim(params[param], site_data, bio_cols)
        corr_df.columns = [f'{param}_{c}' for c in corr_df.columns]
        summary_frames.append(corr_df)

    summary = pd.concat(summary_frames, axis=1)
    out_path = f'results/figS2_calcs_{model_label}_correlations.csv'
    summary.to_csv(out_path)

    # Print all Spearman correlations for each parameter, most positive to most negative
    print(f'\n=== {model_label} ===')
    for param in param_cols:
        col = f'{param}_spearman_r'
        ranked = summary[col].dropna().sort_values(ascending=False)
        print(f'  {param} — Spearman r (most positive to most negative):')
        for var, r in ranked.items():
            print(f'    {var:35s}  r = {r:+.3f}')
