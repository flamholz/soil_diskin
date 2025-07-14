#%% import libraries
import numpy as np
import pandas as pd
import xarray as xr
from models import * 
from collections import namedtuple
import pickle
from utils import download_file
from constants import *
from scipy.integrate import odeint
#%% Run CABLE (CASA-CNP) model

# Create the config object for the model based on parameters for Evergreen broadleaf foress in Test S5. 
# in [Xia et al. 2013](https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.12172)

# We use only the 6 pools for litter and soil carbon. To calulcate u for the soil pools, we use the
# u values for the plant pools, and thre transfer coefficients from the plant pools to the litter pools
CABLE_config = namedtuple('Config', ['K','eta','A','u'])
CABLE_config.K = np.array([10., 0.95, 0.49, 2.19, 0.12, 0.0027])
CABLE_config.eta = 0.4 * np.ones(6)  
CABLE_config.u = np.array([0.249 * 0.69 + 0.551 * 0.6, 0.249 * 0.31 + 0.551 * 0.4, 0.2, 0, 0, 0])
CABLE_config.A = -np.diag(np.ones(6))  # Assuming 5 pools for simplicity
CABLE_config.A[3,0] = 0.45
CABLE_config.A[3,1] = 0.36
CABLE_config.A[3,2] = 0.24
CABLE_config.A[4,1] = 0.14
CABLE_config.A[4,2] = 0.28
CABLE_config.A[4,3] = 0.39
CABLE_config.A[5,3] = 0.006
CABLE_config.A[5,4] = 0.003

# Create the CABLE model instance
cable_model = CABLE(CABLE_config)
ages_cable = np.arange(0,100_000,0.1)
pA_cable = cable_model.pA_ss(ages_cable)

# Save the model predictions
with open(f'../results/model_predictions/CABLE_{pd.Timestamp.now().date().strftime("%d-%m-%Y")}.pkl','wb') as f:
    pickle.dump([ages_cable, pA_cable.squeeze()],f)

#%% Run the CLM5 model

#%% Run the JSBACH model

# from https://pure.mpg.de/rest/items/item_3279802_26/component/file_3316522/content#page=107.51 and 
# https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/src/carbon/mo_carbon_process.f90
a_i=np.array([0.72, 5.9, 0.28, 0.031]) # Eq. 6.28
a_h = 0.0016 # Eq. 6.34
b1 = 9.5e-2; b2 = -1.4e-3; gamma = -1.21; # Eq. 6.30 - T in C and P in m/yr
phi1 = -1.71; phi2 = 0.86; r = -0.306; # Eq. 6.31

# Download temperature and precipitation data from the JSBACH model
download_file("https://gcbo-opendata.s3.eu-west-2.amazonaws.com/trendyv12-gcb2023/JSBACH/S3/JSBACH_S3_tas.nc", "../data/model_params/JSBACH", "JSBACH_S3_tas.nc")
download_file("https://gcbo-opendata.s3.eu-west-2.amazonaws.com/trendyv12-gcb2023/JSBACH/S3/JSBACH_S3_pr.nc", "../data/model_params/JSBACH", "JSBACH_S3_pr.nc")
download_file("https://gcbo-opendata.s3.eu-west-2.amazonaws.com/trendyv12-gcb2023/JSBACH/S3/JSBACH_S3_npp.nc", "../data/model_params/JSBACH", "JSBACH_S3_npp.nc")

# Load the JSBACH forcing data and calculate the monthly means
JSBACH_tas = xr.open_dataarray('../data/model_params/JSBACH/JSBACH_S3_tas.nc').groupby("time.month").mean()
JSBACH_pr = xr.open_dataarray('../data/model_params/JSBACH/JSBACH_S3_pr.nc').groupby("time.month").mean()
JSBACH_npp = xr.open_dataarray('../data/model_params/JSBACH/JSBACH_S3_npp.nc').groupby("time.month").mean()

site_data = pd.read_csv('../results/tropical_sites_14C_turnover.csv')

JSBACH_config = namedtuple('Config', ['a_i', 'a_h', 'b1', 'b2', 'gamma', 'phi1', 'phi2', 'r'])
JSBACH_config.a_i = a_i
JSBACH_config.a_h = a_h
JSBACH_config.b1 = b1
JSBACH_config.b2 = b2
JSBACH_config.gamma = gamma
JSBACH_config.phi1 = phi1
JSBACH_config.phi2 = phi2
JSBACH_config.r = r


row = site_data.iloc[0]

def get_env_params(row):
    """Get the environmental parameters for a given site."""
    I = JSBACH_npp.sel(longitude=row.loc['Longitude'], latitude=row.loc['Latitude'], method='nearest').values
    T = JSBACH_tas.sel(longitude=row.loc['Longitude'], latitude=row.loc['Latitude'], method='nearest').values - T_MELT  # convert K to C
    P = JSBACH_pr.sel(longitude=row.loc['Longitude'], latitude=row.loc['Latitude'], method='nearest').values * SECS_PER_DAY * DAYS_PER_YEAR / 1000  # convert kg/m2/s to m/yr
    d = 4  # from WoodLitterSize in https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/data/lctlib_nlct21.def it is 4  # Example diameter values in cm
    return namedtuple('EnvParams', ['I', 'T', 'P', 'd'])(I, T, P, d)
# JSBACH_env_params = namedtuple('EnvParams', ['I','T', 'P', 'd'])
# JSBACH_env_params.I = JSBACH_npp.sel(longitude=row.loc['Longitude'],latitude=row.loc['Latitude'],method='nearest').values
# JSBACH_env_params.T = JSBACH_tas.sel(longitude=row.loc['Longitude'],latitude=row.loc['Latitude'],method='nearest').values - T_MELT # convert K to C - - T_MELT
# JSBACH_env_params.P = JSBACH_pr * SECS_PER_DAY * DAYS_PER_YEAR / 1000 # convert kg/m2/s to m/yr - *sec_per_year/1000._wp
# JSBACH_env_params.d = 4 

JSBACH_env_params = get_env_params(row)

JSBACH_model = JSBACH(config=JSBACH_config,
                 env_params=JSBACH_env_params)

#%%
from scipy.integrate import solve_ivp

tmax = 200_000
ts = np.logspace(-5,np.log10(tmax),1000000)
solution = solve_ivp(JSBACH_model._impulse, t_span=(0,tmax*1.01), y0 = JSBACH_model.u, method="LSODA", t_eval=ts)
ys = solution.y.sum(axis=0)

t_inds = ((ts % (1 / 12)) * 12 * 12).astype(int) # t is in units of years, and the dt for the I, T and P is in months, so we multiply by 12 to get the index
pA = ys * JSBACH_model.I[t_inds]

#%% test against Fortran implementation
  
# distribute NPP
fract_npp_2_woodPool = 0.3 # fraction of NPP that goes to wood pool
fract_npp_2_reservePool = 0.05 # fraction of NPP that goes to reserve pool
fract_npp_2_exudates = 0.05 # fraction of NPP that goes to root exudates
# NPP_2_woodPool    = fract_npp_2_woodPool * NPP # NPP mol(C)/m2/yr
# NPP_2_reservePool = fract_npp_2_reservePool * NPP
# NPP_2_rootExudates= fract_npp_2_exudates * NPP
# NPP_2_greenPool = (1 - fract_npp_2_woodPool - fract_npp_2_reservePool - fract_npp_2_exudates) * NPP


LeafLit_coef = np.array([0.4651, 0.304, 0.0942, 0.1367, 0.]) #coefficient to distribute leaf litter into 5 classes of chemical composition
WoodLit_coef = np.array([0.65, 0.025, 0.025, 0.3, 0.]) #coefficient to distribute woody litter into 5 classes of chemical composition

fract_wood_aboveGround = 0.7 # !< Fraction of C above ground in wood pool (for separation of woody litter into above and below ground litter pools)
fract_green_aboveGround = 0.5 # !< Fraction of C above ground in green pool (for separation of green litter into above and below ground litter pools)

NPP_fractions = np.array([1. - (fract_npp_2_woodPool + fract_npp_2_exudates), fract_npp_2_woodPool, fract_npp_2_exudates])
above_below = np.array([[fract_green_aboveGround, 1 - fract_green_aboveGround],
                        [fract_wood_aboveGround, 1 - fract_wood_aboveGround],
                        [0, 1]])
litter_pool_split = np.stack([LeafLit_coef,WoodLit_coef,np.array([0,1,0,0,0])])
#NPP * NPP_fractions (3,1 - green, wood, exudates) * litter_fractions (4,3)  * above_below (3,2) 
# ((NPP_fractions * litter_pool_split[:,:-1].T) @ above_below).T.flatten()
NPP_to_litter_pools = (NPP_fractions * litter_pool_split[:,:-1].T) # size 4,3

NPP_pools_above_below = np.stack([NPP_to_litter_pools,NPP_to_litter_pools],axis=2) * above_below # size 4,3,2
veg_to_litter = np.array([[1, 0, 1], [0, 1, 0]]) # the vegetation pools that contribute to the non-woody and woody litter pools
B_temp = (NPP_pools_above_below.transpose(0,2,1) @ veg_to_litter.T) # size 4,2,2
B = np.concatenate([B_temp.transpose(1,0,2).reshape(8,2),  np.zeros((1,2))]).T.flatten() # size 18,

one_vec = np.ones(12) # 12 pools in the JSBACH model
JSBACH_env_params = namedtuple('EnvParams', ['I', 'T', 'P', 'd'])(one_vec * DAYS_PER_YEAR, 25 * one_vec, one_vec, 4)

JSBACH_model = JSBACH(config=JSBACH_config,
                 env_params=JSBACH_env_params)

JSBACH_output = JSBACH_model._dX(t = 0, X = np.ones(18))[:9]
fortran_output = pd.read_csv('../../jsbach/yasso_output.csv')

assert np.allclose(fortran_output['Value'].values[:9],JSBACH_output[:9] * (1/DAYS_PER_YEAR) + np.ones(9), rtol=1e-3, atol=1e-3), "JSBACH model output does not match Fortran implementation output"
#%%


    # sole the ode
    term = ODETerm(ode)
    solver = Dopri5()
    t_max = 30_000
    # ts = jnp.concatenate([jnp.arange(0,1.01,0.01),jnp.arange(2,t_max,1)])
    CLM5_ts = jnp.logspace(-2,jnp.log10(t_max),1000)
    solution = diffeqsolve(term, solver, t0=0, t1=t_max, dt0=0.1, y0=u,max_steps=1000000,saveat=SaveAt(ts = CLM5_ts))

    ys = solution.ys.sum(axis=1)
    CLM5_site_cum_age_dist.append(cumtrapz(ys,x=CLM5_ts)/trapz(ys,x=CLM5_ts))

CLM5_site_cum_age_dist = np.stack(CLM5_site_cum_age_dist)

# %%
