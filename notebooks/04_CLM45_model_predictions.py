import numpy as np
import pandas as pd
import xarray as xr
from notebooks.models import * 
from soil_diskin.age_dist_utils import predict_fnew
from soil_diskin.data_wrangling import parse_he_data
from notebooks.constants import *
from scipy.io import loadmat
from joblib import Parallel, delayed, parallel_backend

"""
Runs the CLM4.5 model for the sites in the Balesdent dataset
and saves the predictions to a pickle file.
"""

# Timestamp for saving files
current_date = pd.Timestamp.now().date().strftime("%d-%m-%Y")

# Load the site data
site_data = pd.read_csv('results/processed_balesdent_2018.csv')
turnover_14C = pd.read_csv('results/all_sites_14C_turnover.csv')

#%% Run the CLM4.5 model
print("Generating CLM4.5 model predictions...")

# load soil depth mat file 
fn = 'data/CLM5_global_simulation/soildepth.mat'
mat = loadmat(fn)
zisoi = mat['zisoi'].squeeze()
zsoi = mat['zsoi'].squeeze()
dz = mat['dz'].squeeze()
dz_node = mat['dz_node'].squeeze()

# load gridded nc file with the inputs, initial values, and the environmental variables
global_da = xr.open_dataset('data/CLM5_global_simulation/global_demo_in.nc')
global_da = global_da.rename({'LON':'x','LAT':'y'})
def fix_lon(ds):
    ds['x'] = xr.where(ds['x']>=180,ds['x']-360,ds['x'])
    return ds.sortby('x')

global_da = fix_lon(global_da)
global_da = global_da.rio.write_crs("EPSG:4326", inplace=True)

# define model parameters
CLM_params = xr.open_dataset('data/CLM5_global_simulation/clm5_params.c171117.nc')
taus = np.array([CLM_params['tau_cwd'],CLM_params['tau_l1'],CLM_params['tau_l2_l3'],CLM_params['tau_l2_l3'],CLM_params['tau_s1'],CLM_params['tau_s2'],CLM_params['tau_s3']]).squeeze()
Gamma_soil = 1e-4 
F_soil = 0

# create global configuration parameters
config = ConfigParams(decomp_depth_efolding=0.5, taus=taus, Gamma_soil=Gamma_soil, F_soil=F_soil,
                      zsoi=zsoi, zisoi=zisoi, dz=dz, dz_node=dz_node, nlevels=10, npools=7)
global_data = GlobalData(global_da)

tmax = 10_000  # years

def CLM45_predict_site(row, tmax):
    """Predict the age distribution for a given site using the CLM4.5 model."""
    ldd = global_data.make_ldd(*row[['Latitude','Longitude']].values)
    age_CDF = predict_fnew(CLM5, config, ldd, tmax)
    return age_CDF

with parallel_backend('loky', n_jobs=-1):
    predictions = Parallel(verbose=1)(
        delayed(CLM45_predict_site)(site_data.iloc[i], tmax) for i in range(len(site_data))
    )
#TODO: fix the tmax - increase to 30_000 or 100_000 years

predictions = np.array(predictions)
predictions = pd.DataFrame(predictions)
predictions.columns = np.logspace(-1, tmax, 1000)  # time in years

fnew_predictions = np.array([interp1d(predictions.columns, predictions.iloc[1])(site_data.iloc[i]['Duration_labeling']) for i in range(len(site_data))])

# Save the model predictions
out_fname = f'results/04_model_predictions/CLM45_{current_date}.csv'
print(f"Saving CLM4.5 model predictions to {out_fname}")
predictions.to_csv(out_fname, index=False)

new_format_fname = f'results/04_model_predictions/CLM45_fnew_{current_date}.csv'
print(f"Saving CLM4.5 model fnew predictions to {new_format_fname}")
np.savetxt(new_format_fname, fnew_predictions)