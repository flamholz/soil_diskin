import numpy as np
import pandas as pd
import xarray as xr
from notebooks.models import * 
from collections import namedtuple
from soil_diskin.age_dist_utils import predict_fnew
from notebooks.constants import *
from scipy.interpolate import interp1d
from joblib import Parallel, delayed, parallel_backend


# Load the site data
site_data = pd.read_csv('results/processed_balesdent_2018.csv')
turnover_14C = pd.read_csv('results/all_sites_14C_turnover.csv')

print("Generating JSBACH model predictions...")
# from https://pure.mpg.de/rest/items/item_3279802_26/component/file_3316522/content#page=107.51 and 
# https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/src/carbon/mo_carbon_process.f90
a_i=np.array([0.72, 5.9, 0.28, 0.031]) # Eq. 6.28
a_h = 0.0016 # Eq. 6.34
b1 = 9.5e-2; b2 = -1.4e-3; gamma = -1.21; # Eq. 6.30 - T in C and P in m/yr
phi1 = -1.71; phi2 = 0.86; r = -0.306; # Eq. 6.31

# Load the JSBACH forcing data and calculate the monthly means
JSBACH_tas = xr.open_dataarray('data/model_params/JSBACH/JSBACH_S3_tas.nc').groupby("time.month").mean()
JSBACH_pr = xr.open_dataarray('data/model_params/JSBACH/JSBACH_S3_pr.nc').groupby("time.month").mean()
JSBACH_npp = xr.open_dataarray('data/model_params/JSBACH/JSBACH_S3_npp.nc').groupby("time.month").mean()

def interp_da(da):
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_nodata(np.nan)
    da = da.rio.interpolate_na()
    return da
JSBACH_tas = interp_da(JSBACH_tas)
JSBACH_pr = interp_da(JSBACH_pr)
JSBACH_npp = interp_da(JSBACH_npp)

JSBACH_config = namedtuple('Config', ['a_i', 'a_h', 'b1', 'b2', 'gamma', 'phi1', 'phi2', 'r'])
JSBACH_config.a_i = a_i
JSBACH_config.a_h = a_h
JSBACH_config.b1 = b1
JSBACH_config.b2 = b2
JSBACH_config.gamma = gamma
JSBACH_config.phi1 = phi1
JSBACH_config.phi2 = phi2
JSBACH_config.r = r

def get_env_params(row):
    """Get the environmental parameters for a given site."""
    I = JSBACH_npp.sel(longitude=row.loc['Longitude'], latitude=row.loc['Latitude'], method='nearest').values * SECS_PER_DAY * DAYS_PER_YEAR * 1000  # convert kgC/m2/s to gC/m2/yr
    T = JSBACH_tas.sel(longitude=row.loc['Longitude'], latitude=row.loc['Latitude'], method='nearest').values - T_MELT  # convert K to C
    P = JSBACH_pr.sel(longitude=row.loc['Longitude'], latitude=row.loc['Latitude'], method='nearest').values * SECS_PER_DAY * DAYS_PER_YEAR / 1000  # convert kg/m2/s to m/yr
    d = 4  # from WoodLitterSize in https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/data/lctlib_nlct21.def it is 4  # Example diameter values in cm
    return namedtuple('EnvParams', ['I', 'T', 'P', 'd'])(I, T, P, d)


def JSBACH_predict_site(row, tmax):
    """Predict the age distribution for a given site using the JSBACH model."""
    env_params = get_env_params(row)
    age_CDF = predict_fnew(JSBACH, JSBACH_config, env_params, tmax)

    return age_CDF

tmax = 100_000  # maximum time in years

with parallel_backend('loky', n_jobs=-1):
    JSBACH_predictions = Parallel(verbose=1)(
        delayed(JSBACH_predict_site)(site_data.iloc[i], tmax) for i in range(len(site_data))
    )
# JSBACH_predictions = pd.DataFrame([JSBACH_predict_site(site_data.iloc[i], tmax) for i in range(len(site_data))])
JSBACH_predictions = pd.DataFrame(JSBACH_predictions)
JSBACH_predictions.columns = np.logspace(-1, np.log10(tmax), 1000)  # time in years
JSBACH_fnew_predictions = np.array([interp1d(JSBACH_predictions.columns, JSBACH_predictions.iloc[1])(site_data.iloc[i]['Duration_labeling']) for i in range(len(site_data))])

# Save the model predictions
out_fname = f'results/04_model_predictions/JSBACH.csv'
print(f"Saving JSBACH predictions to {out_fname} ...")
JSBACH_predictions.to_csv(out_fname, index=False)

new_format_fname = f'results/04_model_predictions/JSBACH_fnew.csv'
print(f"Saving JSBACH fnew predictions to {new_format_fname} ...")
np.savetxt(new_format_fname, JSBACH_fnew_predictions)
