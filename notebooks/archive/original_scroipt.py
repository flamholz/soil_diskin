# %%
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rio
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import scipy as sp
from sklearn.neighbors import KDTree

from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob
import pickle
from datetime import datetime

# from CLM_vertical import *
# from CLM_vertical_utils import *

from CLM_vertical_jax import *
from CLM_vertical_utils_jax import *
from  scipy.io import loadmat

from jax import numpy as jnp
import jax
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
from scipy.integrate import cumtrapz, trapz, odeint
# automatically reload external modules
%load_ext autoreload
%autoreload 2

# %% [markdown]
# ## Define Functions

# %%
ages = np.arange(0,100_000)

def calc_age_dist(A,u,ages):
    d = A.shape[0]
    one = np.ones((d,1))
    beta = u/u.sum()
    zT = -1 * one.T @ A
    xss = (-1 * np.linalg.inv(A)) @ u
    X = np.diag(xss)
    eta = xss/xss.sum()
    age_dens = np.array([zT @ sp.linalg.expm(A * a) @ eta for a in ages])
    return age_dens
    # xss = (-1 * solve(A)) %*% u

def get_A_u(params,model='CESM',tau_fac = {'CESM':3.7, 'IPSL':14, 'MRI':13},rs_fac = {'CESM':0.34, 'IPSL':0.07, 'MRI':0.34}, correct= True):
    A = np.diag(-1/params[:3])
    if correct:
        A[1,0] = params[3]/params[0] if model !='MRI' else 0.46*params[3]/params[0]
        A[2,2] = -1/(params[2]*tau_fac[model])
        A[2,1] = rs_fac[model]*params[4]/params[1]
    else:
        A[1,0] = params[3]/params[0] 
        A[2,1] = params[4]/params[1]
    
    u = np.array([params[5],0.0,0.0])

    return A,u


# %%
# tmax sets the length of the simulation
# keep it small so that runtimes are reasonable
# timestep = 0.2 # yrs
# tmax = 5000 # yrs
def dynamic_age_dist(A_t,u,timestep,tmax):
    
    # define the time steps
    ts = np.arange(0,tmax,timestep)

    # a matrix of zeros for timesteps and ks
    # each row is a k, each column is a timestep
    state = np.zeros((u.shape[0], ts.size))

    for i, t in enumerate(ts):    
        # Haven't added new material yet, can just multiply 
        # the whole matrix by the fractional decay
        state += A_t(t) @ state*timestep

        # new input of biomass 
        state[:,i] = u*timestep
    return state

def nonlinear_age_dist(A_t,u,timestep,tmax):
    
    # define the time steps
    ts = np.arange(0,tmax,timestep)

    # a matrix of zeros for timesteps and ks
    # each row is a k, each column is a timestep
    state = np.zeros((u.shape[0], ts.size))

    for i, t in enumerate(ts):    
        # Haven't added new material yet, can just multiply 
        # the whole matrix by the fractional decay
        state += A_t(state)*timestep

        # new input of biomass 
        state[:,i] = u*timestep
    return state

@jax.jit
def dynamic_age_dist_jax(A_t,u,timestep,tmax):
    
    # define the time steps
    ts = np.arange(0,tmax,timestep)

    # a matrix of zeros for timesteps and ks
    # each row is a k, each column is a timestep
    state = np.zeros((u.shape[0], ts.size))

    for i, t in enumerate(ts):    
        # Haven't added new material yet, can just multiply 
        # the whole matrix by the fractional decay
        state += A_t(t) @ state*timestep

        # new input of biomass 
        state[:,i] = u*timestep
    return state


# %% [markdown]
# ## Get results from Balesdent et al. 2018

# %%
# load data
raw_nature_data = pd.read_csv("../../data/age_distribution/41586_2018_328_MOESM5_ESM.csv",skiprows=1,sep=';')
nature_data = raw_nature_data.iloc[:11,:-1]
nature_data = nature_data.apply(lambda x: x.str.replace(',','.')).astype(float)
nature_data = nature_data.set_index('depth_cm')
nature_data.columns = [int(c.split(' ')[0]) for c in nature_data.columns]

nature_data_std = raw_nature_data.iloc[14:,:-1]
nature_data_std = nature_data_std.apply(lambda x: x.str.replace(',','.')).astype(float)
nature_data_std = nature_data_std.set_index('depth_cm')
nature_data_std.columns = [int(c.split(' ')[0]) for c in nature_data_std.columns]

# load site data
nature_site_data = pd.read_excel('../../data/age_distribution/Soil_natural_13C_labelling_profiles_Database.xlsx',skiprows=7)

# take only sites with MAT>17, PANN>1000, P/PET>0.8
mask = (nature_site_data.MAT_C>17.) & (nature_site_data.PANN_mm>1000.) & (nature_site_data['P to PET ratio'] >0.8) #& nature_site_data['Previous vegetation'].isin(['prevFOREST','prevGRASSLAND'])
tropical_sites = nature_site_data[mask]

# calculate the weights for different depths to estimate the 0-1m integrated signal
cols = nature_site_data.columns[nature_site_data.columns.str.startswith("Ctotal_")][:-1]
C_dens = tropical_sites[cols].diff(axis=1).iloc[:,1:]

# Original method
C_weights = C_dens.div(C_dens.sum(axis=1),axis=0).values
C_weights = np.nanmean(C_weights,axis=0)

# 2024-11-04 method:
C_weights = (C_dens.mean()/C_dens.mean().sum()).values

fcols = tropical_sites.columns[tropical_sites.columns.str.startswith('f_') & ~tropical_sites.columns.str.contains('-')]

# Original method
total_fnew = tropical_sites[fcols].T.rolling(2).mean().values[1:,:].T * C_weights
tropical_sites['total_fnew'] = total_fnew.sum(axis=1)

# 2024-11-04 method:
C_vals = tropical_sites[fcols].T.rolling(2).mean().values[1:,:].T
tot_C_weights = np.nansum((C_vals/C_vals) * C_weights,axis=1)
total_fnew = tropical_sites[fcols].T.rolling(2).mean().values[1:,:].T * C_weights
tropical_sites['total_fnew'] = np.nansum(total_fnew,axis=1)/tot_C_weights


unique_tropical_sites = tropical_sites.drop_duplicates(subset=['Latitude', 'Longitude'])


# %% [markdown]
# ### Get turnover time and 14C from the sites

# %%


def find_nearest(coords, ds,tolerance):
    '''
    Find the nearest pixel in a dataset to a set of coordinates
    '''
    if type(coords) == pd.DataFrame:
        coords = coords.values
    df = ds.to_dataframe().dropna()[ds.name]
    kdt = KDTree(np.stack(df.index.to_numpy()),leaf_size=30, metric='euclidean')
    distance,ind = kdt.query(coords,k=1)
    res = df.values[ind]
    res[distance>tolerance] = np.nan
    return res

# %%
c14_data = xr.open_dataset('../../data/age_distribution/global_delta_14C.nc')['temp'].rio.write_crs("EPSG:4326", inplace=True).rio.write_nodata(np.nan).rename({'lon':'x','lat':'y'})
c14_data_extrapolated = c14_data.rio.interpolate_na(method='nearest')
GPP = xr.open_mfdataset('../../../biomass_turnover_change/data/GPP/ST_CFE-Hybrid_NT/CEDAR-GPP_v01_ST_CFE-Hybrid_NT_*.nc')['GPP_mean']
GPP = GPP.where(GPP>0).mean(dim='time')
GPP.rio.write_nodata(np.nan,inplace=True)
GPP.rio.write_crs("EPSG:4326",inplace=True);

# %%
def extract_tropical_sites(ds):
    return xr.concat([ds.sel(y=row['Latitude'],x=row['Longitude'],method='nearest') for i,row in unique_tropical_sites.iterrows()],dim='site')

# extract 1m integrated 14C signal
site_1m_14C = (extract_tropical_sites(c14_data_extrapolated).values.reshape((-1,10,10)).mean(axis=2) @ C_weights)

# extract site GPP
GPP_df = GPP.to_dataframe().dropna()['GPP_mean']
kdt = KDTree(np.stack(GPP_df.index.to_numpy()),leaf_size=30, metric='euclidean')
distance,ind = kdt.query(unique_tropical_sites[['Latitude','Longitude']].values,k=1)
site_GPP = GPP_df.values[ind]
site_GPP[distance>1] = np.nan
site_GPP = site_GPP.squeeze()

site_SOC = tropical_sites.groupby(['Latitude','Longitude'])[['Cref_0-100estim']].mean()['Cref_0-100estim']*1e3 # kgC/m2 to gC/m2

# %%
merged_site_data = pd.DataFrame([site_1m_14C,site_GPP,site_SOC.values],index = ['14C','GPP','SOC'],columns= unique_tropical_sites.set_index(['Latitude','Longitude']).index).T
merged_site_data['turnover'] = merged_site_data['SOC']/merged_site_data['GPP']
merged_site_data['14C_ratio'] = 1+merged_site_data['14C']/1e3

# %%
merged_site_data.std()/np.sqrt(41)

# %% [markdown]
# ## Load forcing data

# %% [markdown]
# ### SoilGrids data

# %%
# load clay content data
def parse_soilgrids(varname):
    file_names = glob(f'../../data/soilgrids/SoilGrids2/{varname}/*.tif')
    depth_top = np.array([int(x.split('_')[-3].split('-')[0]) for x in file_names])
    depth_bot = np.array([int(x.split('_')[-3].split('-')[1][:-2]) for x in file_names])
    depth = depth_bot - depth_top
    ds = xr.concat([xr.open_dataarray(i).rio.reproject('EPSG:4326').sel(band=1) for i in file_names],dim=pd.Index(depth_top,name='depth_top'))
    if varname in ['clay','sand']:
        mean_ds = (ds.transpose('y','x','depth_top')*depth/depth.sum()).sum(dim='depth_top')/1e3*1e2
    else:
        mean_ds = (ds.transpose('y','x','depth_top')*depth/depth.sum()).sum(dim='depth_top')/10
    mean_ds = mean_ds.where(mean_ds>0)
    mean_ds_ext = mean_ds.rio.write_nodata(np.nan).rio.interpolate_na()
    return mean_ds, mean_ds_ext

mean_clay, mean_clay_extrapolated = parse_soilgrids('clay')
mean_sand, mean_sand_extrapolated = parse_soilgrids('sand')
mean_pH, mean_pH_extrapolated = parse_soilgrids('pH')

# %% [markdown]
# ### WorldClim data

# %%
def parse_worldclim(varnum):
    ds = xr.open_dataarray(f'../../Persistence/CodeData/SoilClim/wc2.0_bio_10m_{varnum}.tif').sel(band=1)
    ds_ext = ds.rio.write_nodata(np.nan).rio.interpolate_na()
    return ds, ds_ext

mat, mat_extrapolated = parse_worldclim('01')
Tmax, Tmax_extrapolated = parse_worldclim('05')
Tmin, Tmin_extrapolated = parse_worldclim('06')
Pa, Pa_extrapolated = parse_worldclim('12')

# %% [markdown]
# ### CLM5 historical simulation data

# %%
files = glob('/home/wyujie/DATASERVER/model/CESM/CliMA_Land_Leaf_Optics/IHistClm50BgcCrop_control_v2/lnd/hist/*.nc')
l = xr.open_dataset(files[0])
[x for x in list(l.variables.keys()) if 'FP' in x]

# %%
# find the output of the historical data from CLM5 for soil temperature and soil moisture
files = glob('/home/wyujie/DATASERVER/model/CESM/CliMA_Land_Leaf_Optics/IHistClm50BgcCrop_control_v2/lnd/hist/*.nc')

# calculate the weighting of the soil layers based on their depth
d = xr.open_dataset(files[0])
depths = np.concatenate([d['levgrnd'][0].values[np.newaxis],d['levgrnd'][:5].diff(dim='levgrnd').values])
depths = depths/depths.sum()

# read the files and extract the mean soil temperature and soil moisture for the top 26cm
T_soil=[]
SMP =[]
NPP = []
pfts = []
# SOIL_C = []

for i in tqdm(files):
    
    ds = xr.open_dataset(i)
    T_soil.append((ds['TSOI'][0,:5,:,:].T*depths).sum(dim='levgrnd').T)
    SMP.append((ds['SMP'][0,:5,:,:].T*depths).sum(dim='levgrnd').T)
    pfts.append(ds['PCT_NAT_PFT'][0]*ds['PCT_LANDUNIT'][0,0]/100)
    NPP.append(ds['NPP'][0]*3600*24*365)
    # SOIL_C.append(ds['SOIL1C_vr'][0,:5,:,:])

# calculate the monthly mean of the soil temperature and soil moisture

CLM_params = xr.concat([xr.concat(ds,dim='time').groupby('time.month').mean() for ds in [T_soil,SMP,NPP]],dim='var')

CLM_params['var'] = ['T_soil','SMP','NPP']
def fix_lon(ds):
    ds['lon'] = xr.where(ds['lon']>=180,ds['lon']-360,ds['lon'])
    return ds.sortby('lon')

CLM_params = fix_lon(CLM_params)
CLM_params[CLM_params['var']=='T_soil'] = CLM_params[CLM_params['var']=='T_soil'].where(CLM_params[CLM_params['var']=='T_soil']>0)
CLM_params[CLM_params['var']=='SMP'] = CLM_params[CLM_params['var']=='SMP'].where(CLM_params[CLM_params['var']=='SMP']<0)
CLM_params = CLM_params.rename({'lon':'x','lat':'y'}).rio.write_crs("EPSG:4326", inplace=True).rio.write_nodata(np.nan)
CLM_params_extrapolated = xr.concat([CLM_params[i].rio.interpolate_na() for i in range(3)],dim='var')

PFT_mean = xr.concat(pfts,dim='time').mean(dim='time')
PFT_vars = pd.read_csv('../../data/weider_et_al_2014/text_s03_table.csv',index_col=0)

# %% [markdown]
# ### extract the forcing data for the sites 

# %%
def extract_tropical_sites(ds):
    return xr.concat([ds.sel(y=row['Latitude'],x=row['Longitude'],method='nearest') for i,row in unique_tropical_sites.iterrows()],dim='site')

soil_clim_tropical_site_data = xr.concat(list(map(extract_tropical_sites, [mean_clay_extrapolated,mean_sand_extrapolated,mean_pH_extrapolated,mat_extrapolated,Tmax_extrapolated,Tmin_extrapolated,Pa_extrapolated])),dim='var')
soil_clim_tropical_site_data['var'] = ['clay','sand','pH','MAT','Tmax','Tmin','Pa']
soil_clim_tropical_site_data_df = soil_clim_tropical_site_data.to_dataframe(name='data')['data'].unstack().T
CLM_tropical_site_data = extract_tropical_sites(CLM_params_extrapolated)


# %% [markdown]
# ## Get model parameters for ESMs

# %% [markdown]
# ### CLM5 vertical model

# %%
### CODE FOR FINAL IMPLEMENTATION IN THE ACTUAL CLM5 MODEL ###

# # find the output of the historical data from CLM5 for soil temperature and soil moisture
# files = glob('/home/wyujie/DATASERVER/model/CESM/CliMA_Land_Leaf_Optics/IHistClm50BgcCrop_control_v2/lnd/hist/*.nc')

# # calculate the weighting of the soil layers based on their depth
# d = xr.open_dataset(files[0])
# zsoi = d['levgrnd'].values

# # Taken from https://escomp.github.io/ctsm-docs/versions/release-clm5.0/html/tech_note/Ecosystem/CLM50_Tech_Note_Ecosystem.html#soil-layers
# # zisoi = np.array([0.02, 0.06, 0.12, 0.2, 0.32, 0.48, 0.68, 0.92, 1.2, 1.52, 1.88, 2.28, 2.72, 3.26, 3.9, 4.64, 5.48, 6.42, 7.46, 8.6, 10.99, 15.666, 23.301, 34.441, 49.556])
# # define  the interface between the soil layers
# zisoi = np.zeros_like(zsoi)
# zisoi[0] = zsoi[0]*2
# for i in range(1,len(zisoi)):
#     zisoi[i] = (zsoi[i] - zisoi[i-1])*2 + zisoi[i-1]


# # dz is the difference between interfaces
# dz = np.diff(np.pad(zisoi, (1,0), 'constant', constant_values=0))

# # dz_node is the difference between nodes
# dz_node = np.diff(np.pad(zsoi, (1,0), 'constant', constant_values=0))


# %%
# load soil depth mat file 
fn = '../../data/CLM5/soildepth.mat'
mat = loadmat(fn)
zisoi = mat['zisoi'].squeeze()
zsoi = mat['zsoi'].squeeze()
dz = mat['dz'].squeeze()
dz_node = mat['dz_node'].squeeze()

# load gridded nc file with the inputs, initial values, and the environmental variables
global_da = xr.open_dataset('../../data/CLM5/global_demo_in.nc')
global_da = global_da.rename({'LON':'x','LAT':'y'})
def fix_lon(ds):
    ds['x'] = xr.where(ds['x']>=180,ds['x']-360,ds['x'])
    return ds.sortby('x')

global_da = fix_lon(global_da)
global_da = global_da.rio.write_crs("EPSG:4326", inplace=True)

# define model parameters
CLM_params = xr.open_dataset('../../data/CLM5/clm5_params.c171117.nc')
taus = np.array([CLM_params['tau_cwd'],CLM_params['tau_l1'],CLM_params['tau_l2_l3'],CLM_params['tau_l2_l3'],CLM_params['tau_s1'],CLM_params['tau_s2'],CLM_params['tau_s3']]).squeeze()
Gamma_soil = 1e-4 
F_soil = 0


# %%
config = ConfigParams(decomp_depth_efolding=0.5, taus=taus, Gamma_soil=Gamma_soil, F_soil=F_soil,
                      zsoi=zsoi, zisoi=zisoi, dz=dz, dz_node=dz_node, nlevels=10, npools=7)
global_data = GlobalData(global_da)



# %%
CLM5_site_cum_age_dist = []
for i, row in tqdm(unique_tropical_sites.iterrows()):
    ldd = global_data.make_ldd(*row[['Latitude','Longitude']].values)
    CLM_model = CLM_vertical(config, ldd)

    # define the key matrices for the calculation
    A = jnp.array(CLM_model.A)
    V = jnp.array(CLM_model.V)
    K_t = jnp.stack([make_K_matrix(taus, zsoi,
                        ldd.w[t,:], ldd.t[t,:], ldd.o[t,:], ldd.n[t,:],
                        config.decomp_depth_efolding, config.nlevels) for t in range(12)])


    u = jnp.array(ldd.I.mean(dim='TIME1').values)


    def ode(t,y,args):
        # define the ode for an impulse input
        t_ind = jnp.array((t % (1/12))*12*12,int)    
        A_t = jnp.subtract(jnp.dot(A, K_t[t_ind,:,:]), V)
        return jnp.dot(A_t , y)


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
with open(f'../../results/analyze_balesdent/CLM5_site_age_dist_20000_{datetime.now().strftime("%Y%m%d")}.pkl','wb') as f:
    # save CLM_site_age_dist amd cum_CLM_site_age_dist
    pickle.dump([CLM5_ts,CLM5_site_cum_age_dist],f)

# %%

# Function to run the model for a single grid cell
def run_model_for_cell(lat, lon):
    ldd = global_data.make_ldd(lat, lon)
    if np.isnan(ldd.w[0, 0]):
        res = empty_gridcell
        res['LAT'] = lat
        res['LON'] = lon
        return res.expand_dims('LAT').expand_dims('LON').stack(cell=('LAT', 'LON'))
    CLM_model = CLM_vertical(config, ldd)
    res = CLM_model.run(timesteps=range(11), dt=secspday * 30).expand_dims('LAT').expand_dims('LON').stack(cell=('LAT', 'LON'))
    return res


lat_values = global_da['LAT'].values
lon_values = global_da['LON'].values

# Load the line_profiler extension
%load_ext line_profiler

# Profile the function
# %lprun -f run_model_for_cell run_model_for_cell(lat_values[43], lon_values[118])
# %lprun -f CLM_model.run CLM_model.run(timesteps=range(11), dt=secspday * 30)

%lprun -f CLM_model._CLM_vertical CLM_model._CLM_vertical(CLM_model.ldd.X0, 0)

# %% [markdown]
# ### RothC

# %%

def build_RothC_model(MAT,clay):
    
    ksRC = np.array([10,0.3,0.66,0.02])
    # FYMsplit=c(0.49,0.49,0.02)
    DR = 1.44; In = 1.7;
    # x=1.67*(1.85+1.60*np.exp(-0.0786*clay)) # from the Carlos Sierra paper
    x = 1.67*(1.21+2.24*np.exp(-0.085*0.45*clay)) # from the Carlos Sierra paper
    B = 0.46/(x+1) # Proportion that goes to the BIO pool
    H = 0.54/(x+1) # Proportion that goes to the HUM pool

    a = 47.9/(1+np.exp(106/(MAT+18.3)))
    b = 1 # assuming no water limitation
    c = 0.6 # assuming vegetation grows constantly

    ksRC_cor = ksRC * a * b * c

    ai3 = B*ksRC_cor
    ai4 = H*ksRC_cor

    A = np.diag(-ksRC_cor)
    A[2,:] = A[2,:]+ai3
    A[3,:] = A[3,:]+ai4

    u = np.array([In*(DR/(DR+1)),In*(1/(DR+1)),0,0])

    return A,u



# %%
# Implementation as in JULES - https://gmd.copernicus.org/articles/4/701/2011/gmd-4-701-2011.pdf#page=11.83
def build_RothC_model(MAT,clay):
    
    ksRC = np.array([3.22e-7,9.65e-9,2.12e-8,6.42e-10]) *3600*24*365 # from Table 8 but converted into yr-1
    # FYMsplit=c(0.49,0.49,0.02)
    DR = 1.44; In = 1.7;
    # x=1.67*(1.85+1.60*np.exp(-0.0786*clay)) # from the Carlos Sierra paper
    x = 1.67*(1.21+2.24*np.exp(-0.085*0.45*clay)) # from the Carlos Sierra paper
    B = 0.46/(x+1) # Proportion that goes to the BIO pool
    H = 0.54/(x+1) # Proportion that goes to the HUM pool

    a = 47.9/(1+np.exp(106/(MAT+18.3))) 
    b = 1 # assuming no water limitation
    c = 0.6 # assuming vegetation grows constantly

    ksRC_cor = ksRC * a * b * c

    ai3 = B*ksRC_cor
    ai4 = H*ksRC_cor

    A = np.diag(-ksRC_cor)
    A[2,:] = A[2,:]+ai3
    A[3,:] = A[3,:]+ai4

    u = np.array([In*(DR/(DR+1)),In*(1/(DR+1)),0,0])

    return A,u



# %%
def process_row_RothC(MAT,clay):
    ts = np.arange(0,1000,0.01)
    tropical_site_A, tropical_site_u = build_RothC_model(MAT,clay)
    return calc_age_dist(tropical_site_A, tropical_site_u, ts)

# Number of parallel jobs (adjust based on your system's capabilities)
n_jobs = -1  # Use all available cores

site_age_dist = Parallel(n_jobs=n_jobs)(
    delayed(process_row_RothC)(row['MAT'],row['clay']) for i,row in soil_clim_tropical_site_data_df.iterrows())


RothC_site_age_dist = np.stack(site_age_dist).squeeze()
cum_RothC_site_age_dist = np.cumsum(RothC_site_age_dist,axis=1).T/np.sum(RothC_site_age_dist,axis=1).T
ts_Roth = np.arange(0,1000,0.01)

# %%
with open(f'../../results/analyze_balesdent/RothC_site_age_dist_{datetime.now().strftime("%Y%m%d")}.pkl','wb') as f:
    # save CLM_site_age_dist amd cum_CLM_site_age_dist
    pickle.dump([ts_Roth,RothC_site_age_dist,cum_RothC_site_age_dist],f)

# %% [markdown]
# ### JULES

# %%
# load soil depth mat file 
fn = '../../data/CLM5/soildepth.mat'
mat = loadmat(fn)
zisoi = mat['zisoi'].squeeze()
zsoi = mat['zsoi'].squeeze()
dz = mat['dz'].squeeze()
dz_node = mat['dz_node'].squeeze()

# load gridded nc file with the inputs, initial values, and the environmental variables
global_da = xr.open_dataset('../../data/CLM5/global_demo_in.nc')
global_da = global_da.rename({'LON':'x','LAT':'y'})
def fix_lon(ds):
    ds['x'] = xr.where(ds['x']>=180,ds['x']-360,ds['x'])
    return ds.sortby('x')

global_da = fix_lon(global_da)
global_da = global_da.rio.write_crs("EPSG:4326", inplace=True)

# define model parameters
CLM_params = xr.open_dataset('../../data/CLM5/clm5_params.c171117.nc')
taus = np.array([CLM_params['tau_cwd'],CLM_params['tau_l1'],CLM_params['tau_l2_l3'],CLM_params['tau_l2_l3'],CLM_params['tau_s1'],CLM_params['tau_s2'],CLM_params['tau_s3']]).squeeze()
Gamma_soil = 1e-4 
F_soil = 0


# %%
# Implementation as in JULES - https://gmd.copernicus.org/articles/4/701/2011/gmd-4-701-2011.pdf#page=11.83
def build_RothC_model(Tsoil,clay,zsoi,decomp_depth_efolding,T_type='Q10',nlevels=10):
    
    ksRC = np.array([3.22e-7,9.65e-9,2.12e-8,6.43e-10]) *3600*24*365 # from Table 8 but converted into yr-1
    # FYMsplit=c(0.49,0.49,0.02)
    
    # x=1.67*(1.85+1.60*np.exp(-0.0786*clay)) # from the Carlos Sierra paper
    x = 1.67*(1.21+2.24*np.exp(-0.085*0.45*clay)) # from the original RothC paper
    B = 0.46/(x+1) # Proportion that goes to the BIO pool
    H = 0.54/(x+1) # Proportion that goes to the HUM pool

    
    F_T_Q10 = 2**((Tsoil-25)/10)
    F_T_Roth =  47.9/(1+np.exp(106/(Tsoil+18.3)))
    F_T = F_T_Roth if T_type == 'Roth' else F_T_Q10
    F_s = 1 # assuming no water limitation
    F_v = 0.6 # assuming fully vegetated soil

    depth_scalar = np.exp(-zsoi[:nlevels]/decomp_depth_efolding)
    ksRC_cor = ksRC * F_T * F_s * F_v * depth_scalar

    ai3 = B*ksRC_cor
    ai4 = H*ksRC_cor

    A = np.diag(-ksRC_cor)
    A[2,:] = A[2,:]+ai3
    A[3,:] = A[3,:]+ai4

    return block_diag(*[np.diag(k * k_modifier * n_scalar if i in [1,2,3] else k * k_modifier) for i,k in enumerate(ks)]) # only for the litter pools (pools 2,3,4) do we multiply by n_scalar

    u = np.array([In*(DR/(DR+1)),In*(1/(DR+1)),0,0])

    return A,u



# %%


# %%
    # calculate k's from tau's
    ks = 1 / (taus)
    depth_scalar = np.exp(-zsoi[:nlevels]/decomp_depth_efolding)
    k_modifier = (t_scalar * w_scalar * o_scalar * depth_scalar)
    return block_diag(*[np.diag(k * k_modifier * n_scalar if i in [1,2,3] else k * k_modifier) for i,k in enumerate(ks)]) # only for the litter pools (pools 2,3,4) do we multiply by n_scalar


# %% [markdown]
# ### JSBACH

# %%

def build_JSBACH_model(params):
    # from https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/src/carbon/mo_carbon_process.f90 line 1061
    # which is based on Table 1 in https://www.sciencedirect.com/science/article/pii/S1364815211001265#tbl1 and https://www.sciencedirect.com/science/article/pii/S0304380010005855#tbl0020
    a_i=np.array([0.72, 5.9, 0.28, 0.031])
    aH = 0.0016 # for the humus pool
    b1 = 9.5e-2; b2 = -1.4e-3; gamma = -1.21; phi1 = -1.71; phi2 = 0.86; r = -0.306;

    p4 = 0.99; p7 = .0; p10 = .0;
    p1 = 0.48; p8 = .0; p11 = .015;
    p2 = 0.01; p5 = .0; p12 = .95;
    p3 = 0.83; p6 = .01; p9 = .02;
    pH = 4.5e-3;

    Ap=-np.diag(np.ones(4))
    Ap[0,1]=p1; Ap[0,2]=p2; Ap[0,3]=p3; Ap[1,0]=p4; Ap[1,2]=p5; Ap[1,3]=p6
    Ap[2,0]=p7; Ap[2,1]=p8; Ap[2,3]=p9; Ap[3,0]=p10; Ap[3,1]=p11; Ap[3,2]=p12; 

    Ap = block_diag(Ap,Ap)
    Ap_final = np.zeros((9,9))
    Ap_final[:8,:8] = Ap
    Ap_final[8,:-1] = pH
    Ap_final[-1,-1] = -1

    d_precip = 1-np.exp(gamma*params['Pa'])
    d_size   = np.min([1.,(1. + phi1 * params['wood_litter_size'] + phi2 * params['wood_litter_size']**2)**r])

    def A_t(y,t):
        T =  params['MAT'] + 0.5*(params['Tmax']-params['Tmin'])*np.sin(2*np.pi*t) # assume the time is in years
        d_temp = np.exp(b1 * T + b2 * T ** 2)
        kC_woody = a_i * d_temp * d_precip * d_size
        kC_nonwoody = a_i * d_temp * d_precip # no size effect for non-woody litter
        ks = np.hstack([kC_nonwoody, kC_woody,[aH * d_temp * d_precip]])
        A1 = np.diag(ks)
        A = Ap_final @ A1

        # concatenate for non-woody and 
        A_final = block_diag(A,A) @ y

        return A_final

    # build the input vector
    
    NPP_woody = params['NPP'] * params['NPP_to_wood']
    NPP_nonwoody = params['NPP'] * (1-params['NPP_to_wood'] - params['NPP_to_exudates'])

    u = np.hstack([NPP_nonwoody * params['frac_nonwoody_ag'] * params['input_allocation_nonwoody'],
               NPP_nonwoody * (1-params['frac_nonwoody_ag']) * params['input_allocation_nonwoody'],
               [0], # no input to the humus pool
               NPP_woody * params['frac_woody_ag'] * params['input_allocation_woody'],
               NPP_woody * (1-params['frac_woody_ag']) * params['input_allocation_woody'],
               [0] # no input to the humus pool
              ])
    
    u[5] += params['NPP'] * params['NPP_to_exudates'] # add the exudates flux to the water soluble pool in the non-woody below ground pool

    return A_t,u



# %%
JSBACH_params = {'MAT': 25, 
                'Tmin': 15, 
                'Tmax': 35, 
                'Pa': 1000, 
                'wood_litter_size': 4, 
                'NPP': 1,
                'frac_nonwoody_ag': 0.7,
                'frac_woody_ag': 0.5,
                'NPP_to_wood': 0.3,
                'NPP_to_reserve': 0.1,
                'NPP_to_exudates': 0.05,
                'input_allocation_nonwoody': np.array([0.4651,0.3040,0.0942,0.1367]),
                'input_allocation_woody': np.array([0.65,0.025,0.025,0.3])
                }


A_t, u = build_JSBACH_model(JSBACH_params)
t_max = 10_000
JSBACH_ts = np.logspace(-2,np.log10(t_max),1000)
# solution = diffeqsolve(term, solver, t0=0, t1=t_max, dt0=0.1, y0=u,max_steps=1000000,saveat=SaveAt(ts = CLM5_ts))
solution = odeint(A_t, t=JSBACH_ts, y0=u)
ys = solution.sum(axis=1)
JSBACH_cum_age_dist = cumtrapz(ys,x=JSBACH_ts)/trapz(ys,x=JSBACH_ts)

# %%
plt.semilogx(JSBACH_ts[:-1],JSBACH_cum_age_dist)

# %% [markdown]
# ### Yasso07

# %%

def build_Yasso_model(mat,Tmin,Tmax,Pa):
    a_i=np.array([0.66, 4.3, 0.35, 0.22, 0.0033])
    b1 = 7.6e-2; b2 = -8.9e-4; gamma = -1.27;
    p1=0.32;p2=0.01;p3=0.93;p4=0.34;p5=0;p6=0;p7=0.035;p8=0.005;p9=0.01;p10=0.0005;p11=0.03;p12=0.92;pH=0.04

    Ap=-np.ones((5,5))
    Ap[0,1]=p1; Ap[0,2]=p2; Ap[0,3]=p3; Ap[1,0]=p4; Ap[1,2]=p5; Ap[1,3]=p6
    Ap[2,0]=p7; Ap[2,1]=p8; Ap[2,3]=p9; Ap[3,0]=p10; Ap[3,1]=p11; Ap[3,2]=p12; Ap[4,:4]=pH

    u= np.array([10,0,0,0,0])

    def A_t(t):
        T =  mat + 0.5*(Tmax-Tmin)*np.sin(2*np.pi*t) # assume the time is in years
        kC = a_i*np.exp(b1*T + b2*T**2)*(1-np.exp(gamma*Pa))
        A1=np.diag(kC)
        A=Ap @ A1

        return A
    
    return A_t,u



# %%
def process_row_Yasso(MAT,Tmin,Tmax,Pa):
    tropical_site_A, tropical_site_u = build_Yasso_model(MAT,Tmin,Tmax,Pa)
    ans = dynamic_age_dist(tropical_site_A,tropical_site_u,0.1,5000)
    return np.flip(ans.sum(axis=0))

# Number of parallel jobs (adjust based on your system's capabilities)
n_jobs = -1  # Use all available cores

site_age_dist = Parallel(n_jobs=n_jobs)(
    delayed(process_row_Yasso)(row['MAT'],row['Tmin'],row['Tmax'],row['Pa']) for i,row in soil_clim_tropical_site_data_df.iterrows())
    # delayed(process_row_Yasso)(i,j,k,z) for i,j,k,z in zip(tropical_site_mat.values,tropical_site_Tmin.values,tropical_site_Tmax.values,tropical_site_Pa.values)
# )

Yasso_site_age_dist = np.stack(site_age_dist).squeeze()
cum_Yasso_site_age_dist = np.cumsum(Yasso_site_age_dist,axis=1).T/np.sum(Yasso_site_age_dist,axis=1).T
ts_Yasso = np.arange(0,5000,0.1)

# %%
with open(f'../../results/analyze_balesdent/Yasso_site_age_dist_{datetime.now().strftime("%Y%m%d")}.pkl','wb') as f:
    # save CLM_site_age_dist amd cum_CLM_site_age_dist
    pickle.dump([ts_Yasso,Yasso_site_age_dist,cum_Yasso_site_age_dist],f)

# %% [markdown]
# ### CLM4cn

# %%
CLM_site_data = []
for ds in [T_soil_mean,SMP_mean]:
    ds.rio.set_spatial_dims('lon', 'lat', inplace=True)
    ds.rio.write_crs("EPSG:4326", inplace=True)
    ds = ds.rio.write_nodata(np.nan).rio.interpolate_na()
    site_ds = xr.concat([ds.sel(lat=unique_tropical_sites.iloc[i]['Latitude'],lon=unique_tropical_sites.iloc[i]['Longitude'],method='nearest') for i in range(unique_tropical_sites.shape[0])],dim='site')
    CLM_site_data.append(site_ds)

CLM_site_data = xr.concat(CLM_site_data,dim='var')
CLM_site_data['var'] = ['Tsoil','SMP']

# %%
def build_CLM_model(T,SMP,sand,u,timestep):
    B = np.zeros((8,3))
    B[0,:2] = 0.25; B[1,:2] = 0.5; B[2,:2] = 0.25; B[3,2] = 1;

    ks = np.array([434.,26.47,5.145,0.3652,26.47,5.145,0.5114,0.0365])
    K = np.diag(ks)

    f_T = lambda T: (1.5)**((T-25)/10)

    # calculate psi_sat_i from equation 7.54 in https://www2.cesm.ucar.edu/models/cesm2/land/CLM50_Tech_Note.pdf assuming f_om is negligible
    # thus psi_sat_i = psi_sat_min_i from equation 7.55
    psi_sat = lambda sand: -10.*10**(1.88-0.0131*sand) # in units of mm

    # convert from soil matric potential to soil water head using the equation psi = g*rho_w*h --> h = psi/(g*rho_w) where g is 9.81 m/s^2 and rho_w is 1000 kg/m^3
    # so converts from [mm] to [m] * [kg/m^3] * [m/s^2] = [kg / m s^2] = [N/m^2] = [Pa]
    psi_dry = -10e6/9.81/1000*1e3 # from MPa to mm

    f_psi = lambda psi,sand: np.log(psi_dry/np.max([psi_dry,psi]))/np.log(psi_dry/psi_sat(sand))
    cdi = lambda T,psi,sand: f_T(T) * f_psi(psi,sand)
    xi = lambda T,psi,sand: np.diag(np.ones(8)*cdi(T,psi,sand))

    A = np.diag(-np.ones(8))
    A[1,3] = 0.76; A[2,3] = 0.24; A[4,0] = 0.61; A[5,1] = 0.45; A[6,2] = 0.71; A[5,4] = 0.72; A[6,5] = 0.54; A[7,6] = 0.45
    u_res = B @ u
    A_res = lambda t: A @ xi(T[int(t*12*timestep) % 12],SMP[int(t*12*timestep) % 12],sand) @ K

    return A_res,u_res

# %%
[[lat,lon] for i,(lat,lon) in unique_tropical_sites[['Latitude','Longitude']].iterrows()]

# %%
def process_row_CLM(T,SMP,sand):
    
    T = T-273.15 # convert kelvin to celsius
    dt = 1/1200
    tmax = 300
    timestep_res = round((1/12)/dt)
    u = PFT_vars.loc['Broadleaf evergreen tree – tropical',['Leaf','Root','CWD']].astype(float).values
    tropical_site_A, tropical_site_u = build_CLM_model(T,SMP,sand,u,timestep=timestep_res)
    ans = dynamic_age_dist(tropical_site_A,tropical_site_u,dt,tmax) # run at monthly time steps
    return np.flip(ans.sum(axis=0))

# a = process_row_CLM(CLM_site_data.sel(var='Tsoil').values[0],CLM_site_data.sel(var='SMP').values[0],30)
# # Number of parallel jobs (adjust based on your system's capabilities)
n_jobs = -1  # Use all available cores

site_age_dist = Parallel(n_jobs=n_jobs)(
    # delayed(process_row_CLM)(i,j,30) for i,j in zip(CLM_site_data.sel(var='Tsoil').values,CLM_site_data.sel(var='SMP').values)
    delayed(process_row_CLM)(T, SMP, sand) for [T,SMP],sand in zip(CLM_tropical_site_data.values,soil_clim_tropical_site_data.sel(var='sand').values)
)

CLM_site_age_dist = np.stack(site_age_dist).squeeze()
cum_CLM_site_age_dist = np.cumsum(CLM_site_age_dist,axis=1).T/np.sum(CLM_site_age_dist,axis=1).T


# %%

with open('../../results/analyze_balesdent/CLM_site_age_dist.pkl','wb') as f:
    # save CLM_site_age_dist amd cum_CLM_site_age_dist
    pickle.dump([CLM_site_age_dist,cum_CLM_site_age_dist],f)

# %% [markdown]
# #### Calculate steady state SOC

# %%
# Take mean T and SMP
Ts = T_soil_mean.mean(dim='month')
Ts = Ts.where(Ts!=0) - 273.15
SMPs = SMP_mean.mean(dim='month')

# calculate the u in each pixel based on mean PFT cover there
nat_pft_sum = PFT_mean.sum(dim='natpft')
crops = abs(100-nat_pft_sum.where(nat_pft_sum!=0))
all_PFTs = xr.concat([PFT_mean,crops],dim='natpft')/100
all_PFTs = fix_lon(all_PFTs)
us = all_PFTs[0:3,:,:].copy()
us.data = (all_PFTs.T.values @ PFT_vars[['Leaf','Root','CWD']].replace({'-':'0'}).astype(float).values).T

# define the function to calculate the steady state soil C content
X_ss = lambda A,u: -sp.linalg.inv(A) @ u

# %%
res = np.zeros((8,Ts.shape[0],Ts.shape[1]))
for i in range(Ts.shape[0]):
    for j in range(Ts.shape[1]):
        if np.isnan(Ts[i,j]):
            res[:,i,j] = np.nan
        else:
            tropical_site_A, tropical_site_u = build_CLM_model([Ts[i,j].values],[SMPs[i,j].values],30,us[:,i,j].values,timestep=1)
            if np.linalg.matrix_rank(tropical_site_A(0)) > 0:
                res[:,i,j] = X_ss(tropical_site_A(0),tropical_site_u)
            else:
                res[:,i,j] = np.nan



# %%
s = Ts.copy()
s.data = res.sum(axis=0)
s.plot(vmin=0,vmax=25000)

# %%
(SMPs/1e6).plot(vmin=-1)

# %%
fig,ax=plt.subplots()
all_PFTs[[0,-4,-3],:,:].sum(dim='natpft').plot(ax=ax)
# ax.scatter(130,-25,c='r')
all_PFTs.sel(lon=130,lat=-25,method='nearest')

# %%
T = [Ts[i,j].values]
SMP = [SMPs[i,j].values]
sand = 30
u = us[:,i,j].values

B = np.zeros((8,3))
B[0,:2] = 0.25; B[1,:2] = 0.5; B[2,:2] = 0.25; B[3,2] = 1;

ks = np.array([434.,26.47,5.145,0.3652,26.47,5.145,0.5114,0.0365])
K = np.diag(ks)

f_T = lambda T: (1.5)**((T-25)/10)

# calculate psi_sat_i from equation 7.54 in https://www2.cesm.ucar.edu/models/cesm2/land/CLM50_Tech_Note.pdf assuming f_om is negligible
# thus psi_sat_i = psi_sat_min_i from equation 7.55
psi_sat = lambda sand: -10.*10**(1.88-0.0131*sand) # in units of mm

# convert from soil matric potential to soil water head using the equation psi = g*rho_w*h --> h = psi/(g*rho_w) where g is 9.81 m/s^2 and rho_w is 1000 kg/m^3
# so converts from [mm] to [m] * [kg/m^3] * [m/s^2] = [kg / m s^2] = [N/m^2] = [Pa]
psi_dry = -10e6/9.81/1000*1e3 # from MPa to mm

f_psi = lambda psi,sand: np.log(psi_dry/np.max([psi_dry,psi]))/np.log(psi_dry/psi_sat(sand))
cdi = lambda T,psi,sand: f_T(T) * f_psi(psi,sand)
xi = lambda T,psi,sand: np.diag(np.ones(8)*cdi(T,psi,sand))

A = np.diag(-np.ones(8))
A[1,3] = 0.76; A[2,3] = 0.24; A[4,0] = 0.61; A[5,1] = 0.45; A[6,2] = 0.71; A[5,4] = 0.72; A[6,5] = 0.54; A[7,6] = 0.45
u_res = B @ u
A_res = lambda t: A @ xi(T[int(t*12*timestep) % 12],SMP[int(t*12*timestep) % 12],sand) @ K


# %%
ddSMP[0]/1e6

# %%
T[i,j]

# %%
fig,ax = plt.subplots(dpi=300)
T.plot()
ax.scatter(T['lon'][j],T['lat'][i],c='g')

# %% [markdown]
# ### DAYCENT

# %%
def build_DAYCENT_model(T,SMP,sand,clay,lig_N,pH,u,timestep):
    u = np.array([u[0],u[1],u[2]/3,u[2]/3,u[2]/3]) # split CWD into 3 pools
    B = np.zeros((12,5))
    f_met = 0.85-0.013*lig_N
    B[0,0] = f_met; B[1,1] = f_met; B[2,0] = 1-f_met; B[3,1] = 1-f_met; B[4,2] = 1; B[5,3] = 1; B[6,4] = 1; 
    # B[0,:2] = 0.25; B[1,:2] = 0.5; B[2,:2] = 0.25; B[3,2] = 1;

    ks = np.array([8.,18.5,2.,4.9,1.5,0.02,0.1,6.,11.,0.08,0.4,0.0033])

    # Deep DAYCENT
    ks[8] = ks[8]* 0.6
    ks[10] = ks[10]* 0.6

    K = np.diag(ks)

    f_T = lambda T: (1.5)**((T-25)/10)

    # calculate psi_sat_i from equation 7.54 in https://www2.cesm.ucar.edu/models/cesm2/land/CLM50_Tech_Note.pdf assuming f_om is negligible
    # thus psi_sat_i = psi_sat_min_i from equation 7.55
    psi_sat = lambda sand: -10.*10**(1.88-0.0131*sand) # in units of mm

    # convert from soil matric potential to soil water head using the equation psi = g*rho_w*h --> h = psi/(g*rho_w) where g is 9.81 m/s^2 and rho_w is 1000 kg/m^3
    # so converts from [mm] to [m] * [kg/m^3] * [m/s^2] = [kg / m s^2] = [N/m^2] = [Pa]
    psi_dry = -10e6/9.81/1000*1e3 # from MPa to mm

    f_psi = lambda psi,sand: np.log(psi_dry/np.max([psi_dry,psi]))/np.log(psi_dry/psi_sat(sand))
    cdi = lambda T,psi,sand: f_T(T) * f_psi(psi,sand)

    f_pH = lambda a,b,c,d: b + (c/np.pi)*np.arctan(np.pi*c*(pH-a))
    f_o2 = 1;
    f_lig = 0.25;
    f_cult = 1;
    k_mix = 0.25;
    f_pH_met = f_pH(a=4.8,b=0.5,c=1.14,d=0.7)
    f_pH_base = f_pH(a=4.,b=0.5,c=1.1,d=0.7)
    f_pH_passive = f_pH(a=3.,b=0.5,c=1.1,d=0.7)
    
    def xi(T,psi):
        xis = np.array([cdi(T,psi,sand)*f_pH_met,
                        cdi(T,psi,sand)*f_pH_met*f_o2,
                        cdi(T,psi,sand)*f_pH_base*np.exp(-3*f_lig),
                        cdi(T,psi,sand)*f_pH_base*np.exp(-3*f_lig)*f_o2*f_cult,
                        cdi(T,psi,sand)*f_pH_base*np.exp(-3*f_lig),
                        cdi(T,psi,sand)*f_pH_base*np.exp(-3*f_lig),
                        cdi(T,psi,sand)*f_pH_base*np.exp(-3*f_lig)*f_o2,
                        cdi(T,psi,sand)*f_pH_base,
                        cdi(T,psi,sand)*f_pH_met*f_o2*(0.25+0.75*sand/100)*f_cult,
                        cdi(T,psi,sand)*(f_pH_base + k_mix/ks[9]),
                        cdi(T,psi,sand)*f_pH_base*f_o2*f_cult,
                        cdi(T,psi,sand)*f_pH_passive*f_o2*f_cult
                        ])
        return np.diag(xis)

    A = np.diag(-np.ones(12))
    fs = np.diag(-np.zeros(12))
    rs = np.diag(-np.zeros(12))

    
    f_12_9 = (0.003 + 0.032*clay/100)
    f_12_11 = (0.003 + 0.009*clay/100)

    # Deep DAYCENT
    f_12_9 = f_12_9 * 1.4
    f_12_11 = f_12_11 * 1.4

    f_mix = k_mix/(ks[9]*f_pH_base+k_mix)
    
    fs[7,0] = 1; fs[8,1] = 1; fs[9,2] = f_lig; fs[10,3] = f_lig; 
    fs[7,2] = 1 - f_lig; fs[8,3] = 1 - f_lig; fs[9,4] = 0.25;
    fs[9,5] = 0.25; fs[10,6] = 0.25; fs[7,4] = 0.75; fs[7,5] = 0.75; fs[8,6] = 0.75; fs[9,7] = 1; fs[10,8] = 1- f_12_9; fs[11,8] = f_12_9; fs[10,9] = f_mix; fs[11,10] = f_12_11;
    fs[7,9] = 1 - f_mix; fs[8,10] = 1-f_12_11; fs[8,11] = 1;

    r_11_9 = (0.17+0.68*sand/100)/fs[10,8]
    r_9_11 = 0.55/fs[8,10]
    rs[7,0] = 0.55; rs[8,1] = 0.55; rs[9,2] = 0.3; rs[10,3] = 0.3;
    rs[7,2] = 0.45; rs[8,3] = 0.55; rs[9,4] = 0.3; rs[9,5] = 0.3; rs[10,6] = 0.3;
    rs[7,4] = 0.45; rs[7,5] = 0.45; rs[8,6] = 0.55; rs[9,7] = 0.6; rs[10,8] = r_11_9; 
    rs[7,9] = 0.55; rs[8,10] = r_9_11; rs[8,11] = 0.55; 

    A = (1-rs) * fs + A

    u_res = B @ u
    A_res = lambda t: A @ xi(T[int(t*12*timestep) % 12],SMP[int(t*12*timestep) % 12]) @ K

    return A_res,u_res #, lambda t: xi(T[int(t*12*timestep) % 12],SMP[int(t*12*timestep) % 12]), K, B, fs, rs

# %%
def process_row_DAYCENT(T,SMP,sand,clay,pH):
    
    T = T-273.15 # convert kelvin to celsius
    dt = 1/12
    tmax = 5000
    timestep_res = round((1/12)/dt)
    u = PFT_vars.loc['Broadleaf evergreen tree – tropical',['Leaf','Root','CWD']].astype(float).values
    lig_N = float(PFT_vars.loc['Broadleaf evergreen tree – tropical','Lignin'])/float(PFT_vars.loc['Broadleaf evergreen tree – tropical','N'])
    tropical_site_A, tropical_site_u = build_DAYCENT_model(T,SMP,sand,clay,lig_N,pH,u,timestep=timestep_res)
    ans = dynamic_age_dist(tropical_site_A,tropical_site_u,dt,tmax) # run at monthly time steps
    return np.flip(ans.sum(axis=0))

# a = process_row_CLM(CLM_site_data.sel(var='Tsoil').values[0],CLM_site_data.sel(var='SMP').values[0],30)
# Number of parallel jobs (adjust based on your system's capabilities)
n_jobs = -1  # Use all available cores

site_age_dist = Parallel(n_jobs=n_jobs)(
    delayed(process_row_DAYCENT)(T, SMP, sand,clay,pH) for [T,SMP,_],[sand,clay,pH] in zip(CLM_tropical_site_data.values,soil_clim_tropical_site_data.sel(var=['sand','clay','pH']).values.T)
    # delayed(process_row_DAYCENT)(i,j,30,30,5.5) for i,j in zip(CLM_site_data.sel(var='Tsoil').values,CLM_site_data.sel(var='SMP').values)
)

DAYCENT_site_age_dist = np.stack(site_age_dist).squeeze()
cum_DAYCENT_site_age_dist = np.cumsum(DAYCENT_site_age_dist,axis=1).T/np.sum(DAYCENT_site_age_dist,axis=1).T
ts_DAYCENT = np.arange(0,5000,1/12)

# %%
with open(f'../../results/analyze_balesdent/DAYCENT_site_age_dist_{datetime.now().strftime("%Y%m%d")}.pkl','wb') as f:
    # save CLM_site_age_dist amd cum_CLM_site_age_dist
    pickle.dump([ts_DAYCENT,DAYCENT_site_age_dist,cum_DAYCENT_site_age_dist],f)

# %% [markdown]
# #### Calculate steady state SOC

# %%
# Take mean T and SMP
Ts = T_soil_mean.mean(dim='month')
Ts = Ts.where(Ts!=0) - 273.15
SMPs = SMP_mean.mean(dim='month')

# calculate the u in each pixel based on mean PFT cover there
nat_pft_sum = PFT_mean.sum(dim='natpft')
crops = abs(100-nat_pft_sum.where(nat_pft_sum!=0))
all_PFTs = xr.concat([PFT_mean,crops],dim='natpft')/100
all_PFTs = fix_lon(all_PFTs)
us = all_PFTs[0:3,:,:].copy()
lig_Ns = all_PFTs[0,:,:].copy()
us.data = (all_PFTs.T.values @ PFT_vars[['Leaf','Root','CWD']].replace({'-':'0'}).astype(float).values).T

l = PFT_vars.replace({'-':'0'}).astype(float)
PFT_vars['lig_N'] = l['Lignin']/l['N']
lig_Ns.data = (all_PFTs.T.values @ PFT_vars['lig_N'].fillna(PFT_vars['lig_N'].mean()).astype(float).values).T

# define the function to calculate the steady state soil C content
X_ss = lambda A,u: -sp.linalg.inv(A) @ u

# %%
res = np.zeros((12,Ts.shape[0],Ts.shape[1]))
for i in range(Ts.shape[0]):
    for j in range(Ts.shape[1]):
        if np.isnan(Ts[i,j]):
            res[:,i,j] = np.nan
        else:
            tropical_site_A, tropical_site_u = build_DAYCENT_model([Ts[i,j].values],[SMPs[i,j].values],30,30,lig_Ns[i,j],5.5,us[:,i,j].values,timestep=1)
            if np.linalg.matrix_rank(tropical_site_A(0)) > 0:
                res[:,i,j] = X_ss(tropical_site_A(0),tropical_site_u)
            else:
                res[:,i,j] = np.nan



# %%
s = Ts.copy()
s.data = res.sum(axis=0)
s.plot(vmin=0,vmax=45000)

# %% [markdown]
# ### MIMICS

# %%
fixed_params = {
    'V_slope': 0.063,
    'f_i': np.array([0.05,0.05]),
    'V_int': 5.47,
    'a_V': 8e-6,
    'V_mod_r': np.array([10,2,10]),
    'V_mod_k': np.array([3,3,2]),
    'K_slope': np.array([0.017,0.027,0.017]),
    'K_int': 3.19,
    'a_K': 10,
    'MGE': np.array([0.55, 0.25, 0.75, 0.35]),
    'KO': np.array([4,4])
}

# uppack the dictionary into variables
locals().update(fixed_params)
def calc_varying_params(lig_N, clay, NPP):
    f_m = 0.85 - 0.013 * lig_N
    p_scalar = 1/(2*np.exp(-2*np.sqrt(clay)))
    K_mod_r = np.array([8,2,4 * p_scalar])
    K_mod_k = np.array([8,2,4 * p_scalar])
    tau_mod = np.max([0.8,np.min([1.2,np.sqrt(NPP/100)])])
    tau = np.array([5.2e-4 * np.exp(0.3 * f_m) * tau_mod, 2.4e-4 * np.exp(0.1 * f_m) * tau_mod])
    f_p = np.array([0.3 * np.exp(1.3 * clay), 0.2 * np.exp(0.8 * clay)])
    f_c = np.array([0.1 * np.exp(-3 * f_m), 0.3 * np.exp(-3 * f_m)])
    f_a = 1 - (f_p + f_c)
    D = 1.5e-5 * np.exp(-1.5 * clay)
    return f_m, K_mod_r, K_mod_k, tau, f_p, f_c, f_a, D
# f_m = 0.85 - 0.013 * lig_N
# p_scalar = 1/(2*np.exp(-2*np.sqrt(clay)))
# K_mod_r = np.array([8,2,4 * p_scalar])
# K_mod_k = np.array([8,2,4 * p_scalar])
# tau_mod = np.max([0.8,np.min([1.2,np.sqrt(NPP/100)])])
# tau = np.array([5.2e-4 * np.exp(0.3 * f_m) * tau_mod, 2.4e-4 * np.exp(0.1 * f_m) * tau_mod])
# f_p = np.array([0.3 * np.exp(1.3 * clay), 0.2 * np.exp(0.8 * clay)])
# f_c = np.array([0.1 * np.exp(-3 * f_m), 0.3 * np.exp(-3 * f_m)])
# f_a = 1 - (f_p + f_c)
# D = 1.5e-5 * np.exp(-1.5 * clay)


# %%
def build_MIMICs_model(lig_N, clay, NPP, T):

    # unpack the parameters
    # f_m, f_i, V_slope, V_int, a_V, V_mod_r, V_mod_k, K_slope, K_int, a_K, K_mod_r, K_mod_k, p_scalar, MGE, tau, tau_mod, f_p, f_c, f_a, D, KO = p
    f_m, K_mod_r, K_mod_k, tau, f_p, f_c, f_a, D = calc_varying_params(lig_N, clay, NPP)
    
    
    # calculate the parameters used in Eq. A1-A17
    V_K_calc = lambda slope, intercept, a, mod, T: np.exp(intercept + slope * T) * a * mod
    
    V_max_r1, V_max_r2, V_max_r3 = [V_K_calc(V_slope, V_int, a_V, i, T) for i in V_mod_r]
    V_max_k1, V_max_k2, V_max_k3 = [V_K_calc(V_slope, V_int, a_K, i, T) for i in V_mod_k]
    Km_r1, Km_r2, Km_r3 = [V_K_calc(i, K_int, a_K, j, T) for i,j in zip(K_slope,K_mod_r)]
    Km_k1, Km_k2, Km_k3 = [V_K_calc(i, K_int, a_K, j, T) for i,j in zip(K_slope,K_mod_k)]
    tau_r, tau_k = tau
    KO_r, KO_k = KO
    f_i_met, f_i_struc = f_i
    f_p_r, f_p_k = f_p
    f_c_r, f_c_k = f_c
    f_a_r, f_a_k = f_a

    
    def A_state(state):    
        # unpack the variables
        # LIT_m, LIT_s, MIC_r, MIC_k, SOM_a, SOM_p, SOM_c = state

        LIT_m_MIC_r = state[2,:] * V_max_r1 * state[0,:] / (Km_r1 + state[0,:])    # A1
        LIT_s_MIC_r = state[2,:] * V_max_r2 * state[1,:] / (Km_r2 + state[1,:])    # A2

        SOM_a_MIC_r = state[2,:] * V_max_r3 * state[4,:] / (Km_r3 + state[4,:])    # A3
        MIC_r_SOM = state[2,:] * tau_r                                   # A4
        
        LIT_m_MIC_k = state[3,:] * V_max_k1 * state[0,:] / (Km_k1 + state[0,:])    # A5
        LIT_s_MIC_k = state[3,:] * V_max_k2 * state[1,:] / (Km_k2 + state[1,:])    # A6

        SOM_a_MIC_k = state[3,:] * V_max_k3 * state[4,:] / (Km_k3 + state[4,:])    # A7
        MIC_k_SOM = state[3,:] * tau_k                                   # A8

        SOM_p_SOM_a = state[5,:] * D                                     # A9
        SOM_c_SOM_a = (state[2,:] * V_max_r2 * state[6,:] / (KO_r * Km_r2 + state[6,:])) + (state[3,:] * V_max_k2 * state[6,:] / (KO_k * Km_k2 + state[6,:])) # A10


        dLIT_m = - LIT_m_MIC_r - LIT_m_MIC_k # A11 # I moved the NPP * f_m * (1-f_i_met) into u
        dLIT_s = - LIT_s_MIC_r - LIT_s_MIC_k # A12 # I moved the NPP * (1 - f_m) * f_i_struc into u
        dMIC_r = MGE[0] * LIT_m_MIC_r + MGE[1] * LIT_s_MIC_r + MGE[0] * SOM_a_MIC_r - MIC_r_SOM # A13
        dMIC_k = MGE[2] * LIT_m_MIC_k + MGE[3] * LIT_s_MIC_k + MGE[2] * SOM_a_MIC_k - MIC_k_SOM # A14
        dSOM_p = f_p_r * MIC_r_SOM + f_p_k * MIC_k_SOM - SOM_p_SOM_a # A15 I moved the NPP * f_m * f_i_met into u
        dSOM_c = f_c_r * MIC_r_SOM + f_c_k * MIC_k_SOM - SOM_c_SOM_a # A16 I moved the NPP * (1 - f_m) * f_i_struc into u
        dSOM_a = f_a_r * MIC_r_SOM + f_a_k * MIC_k_SOM + SOM_p_SOM_a + SOM_c_SOM_a - SOM_a_MIC_r - SOM_a_MIC_k # A17

        dA = np.array([dLIT_m,dLIT_s,dMIC_r,dMIC_k,dSOM_a,dSOM_p,dSOM_c])
        return dA
    # A_state = np.vectorize(A_state,signature='(m)->(m)')
    u = np.array([
                    NPP * f_m * (1-f_i_met),
                    NPP * (1 - f_m) * f_i_struc,
                    0,
                    0,
                    NPP * f_m * f_i_met,
                    NPP * (1 - f_m) * f_i_struc,
                    0
                ])

    return A_state, u

# %%
# process_row_MIMICS(30, 500, 25+273.15)

NPP = 500
clay = 30
T = 25
# T = T-273.15 # convert kelvin to celsius
dt = 1/365/24
tmax = 100
timestep_res = round((1/12)/dt)
lig_N = 18

A_t, u = build_MIMICs_model(lig_N, clay, NPP, T)
# ans = nonlinear_age_dist(tropical_site_A,tropical_site_u,dt,tmax) # run at monthly time steps

# define the time steps
ts = np.arange(0,tmax,dt)

# a matrix of zeros for timesteps and ks
# each row is a k, each column is a timestep
state = np.zeros((u.shape[0], ts.size))
state[:,0] = u*dt
state[2,0] = NPP * (0.85 - 0.013 * lig_N) *dt
state[3,0] = NPP * (1-(0.85 - 0.013 * lig_N)) * dt

for i, t in enumerate(tqdm(ts)):    
    # new input of biomass 
    

    # Haven't added new material yet, can just multiply 
    # the whole matrix by the fractional decay
    state += A_t(state)*dt
    if i>0:
        state[:,i] = u*dt
    


# %%
state[:,0]

# %%


# %%
k = np.flip(state.sum(axis=0))

# %%
A_t(state)

# %%
plt.plot(ts,np.cumsum(k)/k.sum())

# %%

def process_row_MIMICS(clay, NPP, T):
    
    T = T-273.15 # convert kelvin to celsius
    dt = 1/12
    tmax = 5000
    timestep_res = round((1/12)/dt)
    lig_N = float(PFT_vars.loc['Broadleaf evergreen tree – tropical','Lignin'])/float(PFT_vars.loc['Broadleaf evergreen tree – tropical','N'])
    tropical_site_A, tropical_site_u = build_MIMICs_model(lig_N, clay, NPP, T)
    ans = nonlinear_age_dist(tropical_site_A,tropical_site_u,dt,tmax) # run at monthly time steps
    return np.flip(ans.sum(axis=0))

# a = process_row_CLM(CLM_site_data.sel(var='Tsoil').values[0],CLM_site_data.sel(var='SMP').values[0],30)
# Number of parallel jobs (adjust based on your system's capabilities)
n_jobs = -1  # Use all available cores

site_age_dist = Parallel(n_jobs=n_jobs)(
    delayed(process_row_MIMICS)(clay,NPP, MAT) for [T,SMP,NPP],[clay,MAT] in zip(CLM_tropical_site_data.values,soil_clim_tropical_site_data.sel(var=['clay','MAT']).values.T)
    # delayed(process_row_DAYCENT)(i,j,30,30,5.5) for i,j in zip(CLM_site_data.sel(var='Tsoil').values,CLM_site_data.sel(var='SMP').values)
)

MIMICS_site_age_dist = np.stack(site_age_dist).squeeze()
cum_MIMICS_site_age_dist = np.cumsum(MIMICS_site_age_dist,axis=1).T/np.sum(MIMICS_site_age_dist,axis=1).T
ts_MIMICS = np.arange(0,5000,1/12)

# %%
CLM_tropical_site_data

# %% [markdown]
# ### CESM, IPSL, MRI

# %%
model_names = {'CESM':'CESM1','IPSL':'IPSL-CM5A-LR','MRI':'MRI-ESM1'}
file_names = {'CESM':'areacella_fx_CESM1-BGC_1pctCO2_r0i0p0','IPSL':'areacella_fx_IPSL-CM5A-LR_historicalNat_r0i0p0','MRI':'areacella_fx_MRI-ESM1_esmControl_r0i0p0'}

def parse_he_data(model='CESM'):
    
    grid = xr.open_dataset(f'../../data/CMIP5/{model_names[model]}/{file_names[model]}.nc')['areacella']
    params_raster = xr.zeros_like(grid).T
    params_raster = params_raster.where(params_raster!=0)
    params_df = pd.read_csv(f'../../Persistence/CodeData/He/{model}/compartmentalParameters.txt',sep=' ')
    ds = []
    for par in params_df.columns[1:]:
        x = params_raster.values.flatten()
        x[params_df['id']] = params_df[par]
        x2 = x.reshape(params_raster.shape)
        tmp_ds = params_raster.copy()
        tmp_ds.values = np.fliplr(x2)
        ds.append(tmp_ds.T)

    ds = xr.concat(ds,dim='parameter')
    ds['parameter'] = params_df.columns[1:]
    ds.name = model
    ds['lon'] = ds['lon']-180
    return ds



# %%
mod = parse_he_data('CESM')

# %%
mod.sel(parameter='tau3').plot(vmin=0,vmax=2000,cmap='jet')

# %%
tropical_site_params = []
for model in ['CESM','IPSL','MRI']:
    mod = parse_he_data(model=model)
    extrapolated =  xr.concat([mod.sel(parameter=p).rio.write_nodata(np.nan).rio.set_spatial_dims('lon','lat').rio.write_crs(4326).rio.interpolate_na() for p in mod.parameter],dim='parameter')

    unique_tropical_sites = tropical_sites.drop_duplicates(subset=['Latitude', 'Longitude'])

    tropical_site_params.append(xr.concat([extrapolated.sel(lat=tropical_sites.iloc[i]['Latitude'],lon=tropical_sites.iloc[i]['Longitude'],method='ffill') for i in range(unique_tropical_sites.shape[0])],dim='site'))

tropical_site_params = xr.concat(tropical_site_params,dim='model')
tropical_site_params['model'] = ['CESM','IPSL','MRI']


# %% [markdown]
# #### Calculate age dist from models

# %%
ages1 = np.arange(0,20_000,0.1)
ages2 = np.arange(0,100_000,0.1)

# Define a function to process each row
def process_row(params_row,model):
    
    tropical_site_A, tropical_site_u = get_A_u(params_row, model)
    return calc_age_dist(tropical_site_A, tropical_site_u, ages2)

def process_row_uncorrected(params_row):
    tropical_site_A, tropical_site_u = get_A_u(params_row, model=model,correct=False)
    return calc_age_dist(tropical_site_A, tropical_site_u, ages1)

# Number of parallel jobs (adjust based on your system's capabilities)
n_jobs = -1  # Use all available cores


model_site_age_dist = []
model_site_age_dist_uncorrected = []
for model in tqdm(['CESM','IPSL','MRI']):    
    # Run the process in parallel with a progress bar
    site_age_dist = Parallel(n_jobs=n_jobs)(  
        delayed(process_row)(tropical_site_params.sel(model=model).values[i, :],model) for i in range(tropical_site_params['site'].size)
    )
    
    # run the same function but with the argument correct=False
    site_age_dist_uncorrected = Parallel(n_jobs=n_jobs)(
        delayed(process_row_uncorrected)(tropical_site_params.sel(model=model).values[i, :]) for i in range(tropical_site_params['site'].size)
    )
    
    model_site_age_dist.append(np.stack(site_age_dist).squeeze())
    model_site_age_dist_uncorrected.append(np.stack(site_age_dist_uncorrected).squeeze())

# %%
model_site_age_dist = np.stack(model_site_age_dist)
model_site_age_dist_uncorrected = np.stack(model_site_age_dist_uncorrected)

cum_model_site_age_dist = (np.cumsum(model_site_age_dist,axis=2).T/np.sum(model_site_age_dist,axis=2).T)
cum_model_site_age_dist = xr.DataArray(cum_model_site_age_dist,dims=['age','site','model'],coords={'model':['CESM','IPSL','MRI'],'site':unique_tropical_sites.index,'age':ages2})

cum_model_site_age_dist_uncorrected = (np.cumsum(model_site_age_dist_uncorrected,axis=2).T/np.sum(model_site_age_dist_uncorrected,axis=2).T)
cum_model_site_age_dist_uncorrected = xr.DataArray(cum_model_site_age_dist_uncorrected,dims=['age','site','model'],coords={'model':['CESM','IPSL','MRI'],'site':unique_tropical_sites.index,'age':ages1})


# %%
with open(f'../../results/analyze_balesdent/CESM_IPSL_MRI_site_age_dist_{datetime.now().strftime("%Y%m%d")}.pkl','wb') as f:
    # save CLM_site_age_dist amd cum_CLM_site_age_dist
    pickle.dump([ages1, ages2,model_site_age_dist,model_site_age_dist_uncorrected,cum_model_site_age_dist,cum_model_site_age_dist_uncorrected],f)

# %%

# take diagonal values


# %%
j = 5
plt.semilogx(ages1,cum_model_site_age_dist_uncorrected.sel(model='CESM')[:,j])
tropical_site_A, tropical_site_u = get_A_u(tropical_site_params.sel(model=model).values[j, :], model='CESM',correct=False)

for i in -1/np.diag(tropical_site_A):
    plt.axvline(i)

# %% [markdown]
# ## Plot results

# %%
# load data
date = "20241101"
date_DAYCENT = "20241104"
date_ROTH = "20241104"
date_CLM5 = '20241206'
ts_Roth,RothC_site_age_dist,cum_RothC_site_age_dist = pickle.load(open(f'../../results/analyze_balesdent/RothC_site_age_dist_{date_ROTH}.pkl','rb'))
ts_Yasso,Yasso_site_age_dist,cum_Yasso_site_age_dist = pickle.load(open(f'../../results/analyze_balesdent/Yasso_site_age_dist_{date}.pkl','rb'))
ts_DAYCENT,DAYCENT_site_age_dist,cum_DAYCENT_site_age_dist = pickle.load(open(f'../../results/analyze_balesdent/DAYCENT_site_age_dist_{date_DAYCENT}.pkl','rb'))
CLM_site_age_dist,cum_CLM_site_age_dist = pickle.load(open(f'../../results/analyze_balesdent/CLM_site_age_dist.pkl','rb'))
ts_CLM = np.arange(0,300,1/1200)
ts_CLM5, cum_CLM5_site_age_dist = pickle.load(open(f'../../results/analyze_balesdent/CLM5_site_age_dist_20000_{date_CLM5}.pkl','rb'))
ages1, ages2,model_site_age_dist,model_site_age_dist_uncorrected,cum_model_site_age_dist,cum_model_site_age_dist_uncorrected = pickle.load(open(f'../../results/analyze_balesdent/CESM_IPSL_MRI_site_age_dist_{date}.pkl','rb'))



# %%
CLM5_results = pickle.load(open('../../results/analyze_balesdent/CLM5_age_dist_3000_0.03.pkl', 'rb'))
CLM5_results = np.flip(CLM5_results)
CLM5_results = np.cumsum(CLM5_results)/np.sum(CLM5_results)

# %%
# plt.semilogx(ts_CLM5[:-1],CLM5_mean)


# %%
import seaborn as sns
fig,ax = plt.subplots(dpi=300)
# dat = nature_site_data[nature_site_data['total_fnew'].notna()]#.plot.scatter('Duration_labeling','total_fnew',ax=ax,label='observations',c='k')
sns.regplot(data=tropical_sites, x="Duration_labeling", y="total_fnew",ax=ax,scatter_kws={'color':'k'},line_kws={'color':'k','lw':0},x_bins=[3,10,30,50,100,300,1000,3000],fit_reg=False,ci=95)
# sns.regplot(data=dat, x="Duration_labeling", y="total_fnew",ax=ax,scatter_kws={'color':'k','alpha':0.5},line_kws={'color':'k','alpha':0.5,'lw':0},x_bins=np.logspace(np.log10(2),4,10),fit_reg=False,ci=95)
# mean.plot(ax=ax,c='k',marker='o',label='observations')
# ax.fill_between(std.index,mean-std,mean+std,alpha=.3,color='k',lw=0)
# ax.fill_between(std.index,mean-1.96*std,mean+1.96*std,alpha=.1,color='k',lw=0)
# ax.set_xscale('log')
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xscale('log')

# for l,d in zip(['RothC','CENTURY','CESM','CLM4cn - needleleaf','CLM4cn - decideous','CLM4cn - tropical','ICBM','Yasso07','IPSL','MRI'],[RothC_p_age,CENTURY_p_age,CESM_p_age,CLM4cn_ndl_p_age,CLM4cn_dec_p_age,CLM4cn_trop_p_age,ICBM_p_age,Yasso07_p_age,IPSL_p_age,MRI_p_age]):
    # plt.plot(ages,np.cumsum(d)/d.sum(),label=l)
# stocks from https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GB005678
#NPP from https://esajournals-onlinelibrary-wiley-com.ezproxy.weizmann.ac.il/doi/10.1890/08-0588.1
# ps = [2886/50,4830]
#age tropics from Shi et al.
# age_tropics = 93/(93+102+103+107)*390+102/(93+102+103+107)*2790 + 103/(93+102+103+107)*510 + 107/(93+102+103+107)*2860
# age_tropics_high = 93/(93+102+103+107)*770+102/(93+102+103+107)*4310 + 103/(93+102+103+107)*1620 + 107/(93+102+103+107)*7960
# # ps_tropics = [(93+102+103+107)/34.1,15_000]
# ps_tropics = [30,7500]
# ps = ps_tropics
# # ts, p_age = generate_age_dist(*ps)
# ax.plot(ts,np.cumsum(p_age)/p_age.sum(),lw=3,c='Indianred',label=f'disordered kinetics model:\n $\\tau$={ps[0]:.2g}; age={ps[1]:.4g}')

plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
ax.set(ylim=(0,1.01),xlabel='Age (years)',ylabel='Cummulative fraction of soil carbon')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='RothC')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='CENTRUY')

# k2 = np.stack(k).squeeze()
# k3 = np.cumsum(k2,axis=1).T/np.sum(k2,axis=1).T
# plt.plot(ages,k3.mean(axis=1),label='CESM',c='k')
# plt.fill_between(ages,np.percentile(k3,0.025,axis=1),np.percentile(k3,97.5,axis=1),alpha=.3,color='k',lw=0)
# plt.plot(ages,cum_model_site_age_dist.mean(axis=1))
# pd.DataFrame(cum_model_site_age_dist.mean(axis=1),index=ages,columns=['CESM','IPSL','MRI']).plot(ax=ax)

# mat_data = pd.read_csv('../../data/age_distribution/pas_tropics_mathematica_20241101_1000000.csv',header=None)
# ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'disordered kinetics model')
mat_data = pd.read_csv('../../data/age_distribution/shortpas_tropics_mathematica_20241101_0.1_200_000.csv',header=None)
ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'disordered kinetics\nmodel')

mat_data = pd.read_csv('../../data/age_distribution/pas_tropics_powerlaw_mathematica_20241104_100000_0.1.csv',header=None)
ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Royalblue',label=f'power law model')


# CLM_se = cum_CLM_site_age_dist.std(axis=1)/np.sqrt(cum_CLM_site_age_dist.shape[1]-1)
# CLM_mean = cum_CLM_site_age_dist.mean(axis=1)
# ax.plot(ts_CLM,CLM_mean,label='CLM',c='C06')
# ax.fill_between(ts_CLM,CLM_mean+1.96*CLM_se,CLM_mean-1.96*CLM_se,alpha=.3,color='C06',lw=0)
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

# cum_model_site_age_dist.mean(dim='site').plot.line(x='age',hue='model',ax=ax)
for model,c in zip(['CESM','IPSL','MRI'],['C00','C01','C02']):
    d = cum_model_site_age_dist.sel(model=model)
    ax.plot(ages2,d.mean(dim='site'),label=model,c=c)
    se = d.std(dim='site')/np.sqrt(d.shape[1]-1)
    ax.fill_between(ages2,d.mean(dim='site')+1.96*se,d.mean(dim='site')-1.96*se,alpha=.3,color=c,lw=0)

    # d = cum_model_site_age_dist_uncorrected.sel(model=model)
    # ax.plot(ages1,d.mean(dim='site'),label=model+' uncorrected',c=c,ls='--')
    # se = d.std(dim='site')/np.sqrt(d.shape[1]-1)
    # ax.fill_between(ages1,d.mean(dim='site')+1.96*se,d.mean(dim='site')-1.96*se,alpha=.3,color=c,lw=0)


    # ax.fill_between(ages, d.mean(dim='site')+d.std(dim='site'), d.mean(dim='site')-d.std(dim='site'),alpha=.3,color=c,lw=0)
    # plt.fill_between(ages,np.percentile(d.values,0.025,axis=1),np.percentile(d.values,97.5,axis=1),alpha=.3,color=c,lw=0)
# sns.regplot(data=cum_model_site_age_dist.to_dataframe(name='frac').reset_index(), x="age", y="frac",ax=ax,scatter_kws={'color':'k'},line_kws={'color':'k'},fit_reg=False,ci=95)
# plt.plot()
ax.set(xlim=[0.1,100_000])
plt.legend()

# %%
import seaborn as sns
fig,ax = plt.subplots(dpi=300)
# dat = nature_site_data[nature_site_data['total_fnew'].notna()]#.plot.scatter('Duration_labeling','total_fnew',ax=ax,label='observations',c='k')
sns.regplot(data=tropical_sites, x="Duration_labeling", y="total_fnew",ax=ax,scatter_kws={'color':'k'},line_kws={'color':'k','lw':0},x_bins=[3,10,30,50,100,300,1000,3000],fit_reg=False,ci=95)
# sns.regplot(data=dat, x="Duration_labeling", y="total_fnew",ax=ax,scatter_kws={'color':'k','alpha':0.5},line_kws={'color':'k','alpha':0.5,'lw':0},x_bins=np.logspace(np.log10(2),4,10),fit_reg=False,ci=95)
# mean.plot(ax=ax,c='k',marker='o',label='observations')
# ax.fill_between(std.index,mean-std,mean+std,alpha=.3,color='k',lw=0)
# ax.fill_between(std.index,mean-1.96*std,mean+1.96*std,alpha=.1,color='k',lw=0)
# ax.set_xscale('log')
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xscale('log')

# for l,d in zip(['RothC','CENTURY','CESM','CLM4cn - needleleaf','CLM4cn - decideous','CLM4cn - tropical','ICBM','Yasso07','IPSL','MRI'],[RothC_p_age,CENTURY_p_age,CESM_p_age,CLM4cn_ndl_p_age,CLM4cn_dec_p_age,CLM4cn_trop_p_age,ICBM_p_age,Yasso07_p_age,IPSL_p_age,MRI_p_age]):
    # plt.plot(ages,np.cumsum(d)/d.sum(),label=l)
# stocks from https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GB005678
#NPP from https://esajournals-onlinelibrary-wiley-com.ezproxy.weizmann.ac.il/doi/10.1890/08-0588.1
# ps = [2886/50,4830]
#age tropics from Shi et al.
# age_tropics = 93/(93+102+103+107)*390+102/(93+102+103+107)*2790 + 103/(93+102+103+107)*510 + 107/(93+102+103+107)*2860
# age_tropics_high = 93/(93+102+103+107)*770+102/(93+102+103+107)*4310 + 103/(93+102+103+107)*1620 + 107/(93+102+103+107)*7960
# # ps_tropics = [(93+102+103+107)/34.1,15_000]
# ps_tropics = [30,7500]
# ps = ps_tropics
# # ts, p_age = generate_age_dist(*ps)
# ax.plot(ts,np.cumsum(p_age)/p_age.sum(),lw=3,c='Indianred',label=f'disordered kinetics model:\n $\\tau$={ps[0]:.2g}; age={ps[1]:.4g}')

plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
ax.set(ylim=(0,1.01),xlabel='Age (years)',ylabel='Cummulative fraction of soil carbon')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='RothC')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='CENTRUY')

# k2 = np.stack(k).squeeze()
# k3 = np.cumsum(k2,axis=1).T/np.sum(k2,axis=1).T
# plt.plot(ages,k3.mean(axis=1),label='CESM',c='k')
# plt.fill_between(ages,np.percentile(k3,0.025,axis=1),np.percentile(k3,97.5,axis=1),alpha=.3,color='k',lw=0)
# plt.plot(ages,cum_model_site_age_dist.mean(axis=1))
# pd.DataFrame(cum_model_site_age_dist.mean(axis=1),index=ages,columns=['CESM','IPSL','MRI']).plot(ax=ax)

# mat_data = pd.read_csv('../../data/age_distribution/pas_tropics_mathematica_20241101_1000000.csv',header=None)
# ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'disordered kinetics model')
mat_data = pd.read_csv('../../data/age_distribution/shortpas_tropics_mathematica_20241101_0.1_200_000.csv',header=None)
ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'disordered kinetics model')

mat_data = pd.read_csv('../../data/age_distribution/pas_tropics_powerlaw_mathematica_20241104_100000_0.1.csv',header=None)
ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Royalblue',label=f'power law model')


# CLM_se = cum_CLM_site_age_dist.std(axis=1)/np.sqrt(cum_CLM_site_age_dist.shape[1]-1)
# CLM_mean = cum_CLM_site_age_dist.mean(axis=1)
# ax.plot(ts_CLM,CLM_mean,label='CLM',c='C06')
# ax.fill_between(ts_CLM,CLM_mean+1.96*CLM_se,CLM_mean-1.96*CLM_se,alpha=.3,color='C06',lw=0)
# N_CLM5 = (~np.isnan(cum_CLM5_site_age_dist.sum(axis=1))).sum()
# CLM5_se = np.nanstd(cum_CLM5_site_age_dist,axis=0)/np.sqrt(N_CLM5-1)
# CLM5_mean = np.nanmean(cum_CLM5_site_age_dist,axis=0)
# ax.plot(ts_CLM5[:-1],CLM5_mean,label='CLM5',c='C06')
# ax.fill_between(ts_CLM5[:-1],CLM5_mean+1.96*CLM5_se,CLM5_mean-1.96*CLM5_se,alpha=.3,color='C06',lw=0)




# DAYCENT_se = cum_DAYCENT_site_age_dist.std(axis=1)/np.sqrt(cum_DAYCENT_site_age_dist.shape[1]-1)
# DAYCENT_mean = cum_DAYCENT_site_age_dist.mean(axis=1)
# ax.plot(ts_DAYCENT,DAYCENT_mean,label='DAYCENT',c='C05')
# ax.fill_between(ts_DAYCENT,DAYCENT_mean+1.96*DAYCENT_se,DAYCENT_mean-1.96*DAYCENT_se,alpha=.3,color='C05',lw=0)


# Yasso_se = cum_Yasso_site_age_dist.std(axis=1)/np.sqrt(cum_Yasso_site_age_dist.shape[1]-1)
# Yasso_mean = cum_Yasso_site_age_dist.mean(axis=1)
# ax.plot(ts_Yasso,Yasso_mean,label='Yasso',c='C04')
# ax.fill_between(ts_Yasso,Yasso_mean+1.96*Yasso_se,Yasso_mean-1.96*Yasso_se,alpha=.3,color='C04',lw=0)

# roth_se = cum_RothC_site_age_dist.std(axis=1)/np.sqrt(cum_RothC_site_age_dist.shape[1]-1)
# roth_mean = cum_RothC_site_age_dist.mean(axis=1)
# ax.plot(ts_Roth,roth_mean,label='RothC',c='C03')
# ax.fill_between(ts_Roth,roth_mean+1.96*roth_se,roth_mean-1.96*roth_se,alpha=.3,color='C03',lw=0)

# plt.legend()

# # cum_model_site_age_dist.mean(dim='site').plot.line(x='age',hue='model',ax=ax)
# for model,c in zip(['CESM','IPSL','MRI'],['C00','C01','C02']):
#     d = cum_model_site_age_dist.sel(model=model)
#     ax.plot(ages2,d.mean(dim='site'),label=model,c=c)
#     se = d.std(dim='site')/np.sqrt(d.shape[1]-1)
#     ax.fill_between(ages2,d.mean(dim='site')+1.96*se,d.mean(dim='site')-1.96*se,alpha=.3,color=c,lw=0)

    # d = cum_model_site_age_dist_uncorrected.sel(model=model)
    # ax.plot(ages1,d.mean(dim='site'),label=model+' uncorrected',c=c,ls='--')
    # se = d.std(dim='site')/np.sqrt(d.shape[1]-1)
    # ax.fill_between(ages1,d.mean(dim='site')+1.96*se,d.mean(dim='site')-1.96*se,alpha=.3,color=c,lw=0)


    # ax.fill_between(ages, d.mean(dim='site')+d.std(dim='site'), d.mean(dim='site')-d.std(dim='site'),alpha=.3,color=c,lw=0)
    # plt.fill_between(ages,np.percentile(d.values,0.025,axis=1),np.percentile(d.values,97.5,axis=1),alpha=.3,color=c,lw=0)
# sns.regplot(data=cum_model_site_age_dist.to_dataframe(name='frac').reset_index(), x="age", y="frac",ax=ax,scatter_kws={'color':'k'},line_kws={'color':'k'},fit_reg=False,ci=95)
# plt.plot()
plt.legend()
ax.set(xlim=[0.1,100_000])


# %% [markdown]
# For IES grant

# %%
import seaborn as sns
fig,ax = plt.subplots(dpi=300,constrained_layout=True,figsize=(8,4))
sns.set_context('talk')
sns.regplot(data=tropical_sites, x="Duration_labeling", y="total_fnew",ax=ax,scatter_kws={'color':'k'},line_kws={'color':'k','lw':0},x_bins=[3,10,30,50,100,300,1000,3000],fit_reg=False,ci=95,label='observations')

ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xscale('log')

plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
ax.set(ylim=(0,1.01),xlabel='age (years)',ylabel='cumulative fraction\nof soil carbon')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='RothC')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='CENTRUY')

# k2 = np.stack(k).squeeze()
# k3 = np.cumsum(k2,axis=1).T/np.sum(k2,axis=1).T
# plt.plot(ages,k3.mean(axis=1),label='CESM',c='k')
# plt.fill_between(ages,np.percentile(k3,0.025,axis=1),np.percentile(k3,97.5,axis=1),alpha=.3,color='k',lw=0)
# plt.plot(ages,cum_model_site_age_dist.mean(axis=1))
# pd.DataFrame(cum_model_site_age_dist.mean(axis=1),index=ages,columns=['CESM','IPSL','MRI']).plot(ax=ax)

# mat_data = pd.read_csv('../../data/age_distribution/pas_tropics_mathematica_20241101_1000000.csv',header=None)
# ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'disordered kinetics model')
# mat_data = pd.read_csv('../../data/age_distribution/shortpas_tropics_mathematica_20241101_0.1_200_000.csv',header=None)
# ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'our model')

# N_CLM5 = (~np.isnan(cum_CLM5_site_age_dist.sum(axis=1))).sum()
# CLM5_se = np.nanstd(cum_CLM5_site_age_dist,axis=0)/np.sqrt(N_CLM5-1)
# CLM5_mean = np.nanmean(cum_CLM5_site_age_dist,axis=0)
# lw =0.5
# ax.plot(ts_CLM5[:-1],CLM5_mean,label='CLM5',c='C06',lw=lw)
# ax.fill_between(ts_CLM5[:-1],CLM5_mean+1.96*CLM5_se,CLM5_mean-1.96*CLM5_se,alpha=.3,color='C06',lw=0)

# DAYCENT_se = cum_DAYCENT_site_age_dist.std(axis=1)/np.sqrt(cum_DAYCENT_site_age_dist.shape[1]-1)
# DAYCENT_mean = cum_DAYCENT_site_age_dist.mean(axis=1)
# ax.plot(ts_DAYCENT,DAYCENT_mean,label='DAYCENT',c='C05',lw=lw)
# ax.fill_between(ts_DAYCENT,DAYCENT_mean+1.96*DAYCENT_se,DAYCENT_mean-1.96*DAYCENT_se,alpha=.3,color='C05',lw=0)


# Yasso_se = cum_Yasso_site_age_dist.std(axis=1)/np.sqrt(cum_Yasso_site_age_dist.shape[1]-1)
# Yasso_mean = cum_Yasso_site_age_dist.mean(axis=1)
# ax.plot(ts_Yasso,Yasso_mean,label='Yasso',c='C04',lw=lw)
# ax.fill_between(ts_Yasso,Yasso_mean+1.96*Yasso_se,Yasso_mean-1.96*Yasso_se,alpha=.3,color='C04',lw=0)

# roth_se = cum_RothC_site_age_dist.std(axis=1)/np.sqrt(cum_RothC_site_age_dist.shape[1]-1)
# roth_mean = cum_RothC_site_age_dist.mean(axis=1)
# ax.plot(ts_Roth,roth_mean,label='RothC',c='C03',lw=lw)
# ax.fill_between(ts_Roth,roth_mean+1.96*roth_se,roth_mean-1.96*roth_se,alpha=.3,color='C03',lw=0)

# plt.legend()

# cum_model_site_age_dist.mean(dim='site').plot.line(x='age',hue='model',ax=ax)
# for model,c in zip(['CESM','IPSL','MRI'],['C00','C01','C02']):
#     d = cum_model_site_age_dist.sel(model=model)
#     ax.plot(ages2,d.mean(dim='site'),label=model,c=c,lw=lw)
#     se = d.std(dim='site')/np.sqrt(d.shape[1]-1)
#     ax.fill_between(ages2,d.mean(dim='site')+1.96*se,d.mean(dim='site')-1.96*se,alpha=.3,color=c,lw=0)

ax.set(xlim=[0.1,100_000])
plt.legend(bbox_to_anchor=(1.05, 1))
# plt.savefig(f'../../results/analyze_balesdent/IES_grant_2025_{datetime.now().strftime("%Y%m%d")}.svg',dpi=300)
# plt.savefig(f'../../results/analyze_balesdent/IES_grant_2025_{datetime.now().strftime("%Y%m%d")}.png',dpi=300)

# %%
import seaborn as sns
fig,ax = plt.subplots(dpi=300,constrained_layout=True,figsize=(8,4))
sns.set_context('talk')
sns.regplot(data=tropical_sites, x="Duration_labeling", y="total_fnew",ax=ax,scatter_kws={'color':'k'},line_kws={'color':'k','lw':0},x_bins=[3,10,30,50,100,300,1000,3000],fit_reg=False,ci=95,label='observations')

ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xscale('log')

plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
ax.set(ylim=(0,1.01),xlabel='age (years)',ylabel='cumulative fraction\nof soil carbon')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='RothC')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='CENTRUY')

# k2 = np.stack(k).squeeze()
# k3 = np.cumsum(k2,axis=1).T/np.sum(k2,axis=1).T
# plt.plot(ages,k3.mean(axis=1),label='CESM',c='k')
# plt.fill_between(ages,np.percentile(k3,0.025,axis=1),np.percentile(k3,97.5,axis=1),alpha=.3,color='k',lw=0)
# plt.plot(ages,cum_model_site_age_dist.mean(axis=1))
# pd.DataFrame(cum_model_site_age_dist.mean(axis=1),index=ages,columns=['CESM','IPSL','MRI']).plot(ax=ax)

# mat_data = pd.read_csv('../../data/age_distribution/pas_tropics_mathematica_20241101_1000000.csv',header=None)
# ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'disordered kinetics model')
mat_data = pd.read_csv('../../data/age_distribution/shortpas_tropics_mathematica_20241101_0.1_200_000.csv',header=None)
ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'our model')

# N_CLM5 = (~np.isnan(cum_CLM5_site_age_dist.sum(axis=1))).sum()
# CLM5_se = np.nanstd(cum_CLM5_site_age_dist,axis=0)/np.sqrt(N_CLM5-1)
# CLM5_mean = np.nanmean(cum_CLM5_site_age_dist,axis=0)
# lw =0.5
# ax.plot(ts_CLM5[:-1],CLM5_mean,label='CLM5',c='C06',lw=lw)
# ax.fill_between(ts_CLM5[:-1],CLM5_mean+1.96*CLM5_se,CLM5_mean-1.96*CLM5_se,alpha=.3,color='C06',lw=0)

# DAYCENT_se = cum_DAYCENT_site_age_dist.std(axis=1)/np.sqrt(cum_DAYCENT_site_age_dist.shape[1]-1)
# DAYCENT_mean = cum_DAYCENT_site_age_dist.mean(axis=1)
# ax.plot(ts_DAYCENT,DAYCENT_mean,label='DAYCENT',c='C05',lw=lw)
# ax.fill_between(ts_DAYCENT,DAYCENT_mean+1.96*DAYCENT_se,DAYCENT_mean-1.96*DAYCENT_se,alpha=.3,color='C05',lw=0)


# Yasso_se = cum_Yasso_site_age_dist.std(axis=1)/np.sqrt(cum_Yasso_site_age_dist.shape[1]-1)
# Yasso_mean = cum_Yasso_site_age_dist.mean(axis=1)
# ax.plot(ts_Yasso,Yasso_mean,label='Yasso',c='C04',lw=lw)
# ax.fill_between(ts_Yasso,Yasso_mean+1.96*Yasso_se,Yasso_mean-1.96*Yasso_se,alpha=.3,color='C04',lw=0)

# roth_se = cum_RothC_site_age_dist.std(axis=1)/np.sqrt(cum_RothC_site_age_dist.shape[1]-1)
# roth_mean = cum_RothC_site_age_dist.mean(axis=1)
# ax.plot(ts_Roth,roth_mean,label='RothC',c='C03',lw=lw)
# ax.fill_between(ts_Roth,roth_mean+1.96*roth_se,roth_mean-1.96*roth_se,alpha=.3,color='C03',lw=0)

# plt.legend()

# cum_model_site_age_dist.mean(dim='site').plot.line(x='age',hue='model',ax=ax)
# for model,c in zip(['CESM','IPSL','MRI'],['C00','C01','C02']):
#     d = cum_model_site_age_dist.sel(model=model)
#     ax.plot(ages2,d.mean(dim='site'),label=model,c=c,lw=lw)
#     se = d.std(dim='site')/np.sqrt(d.shape[1]-1)
#     ax.fill_between(ages2,d.mean(dim='site')+1.96*se,d.mean(dim='site')-1.96*se,alpha=.3,color=c,lw=0)

ax.set(xlim=[0.1,100_000])
plt.legend(bbox_to_anchor=(1.05, 1))
# plt.savefig(f'../../results/analyze_balesdent/IES_grant_2025_{datetime.now().strftime("%Y%m%d")}.svg',dpi=300)
# plt.savefig(f'../../results/analyze_balesdent/IES_grant_2025_{datetime.now().strftime("%Y%m%d")}.png',dpi=300)

# %%
import seaborn as sns
fig,ax = plt.subplots(dpi=300,constrained_layout=True,figsize=(8,4))
sns.set_context('talk')
sns.regplot(data=tropical_sites, x="Duration_labeling", y="total_fnew",ax=ax,scatter_kws={'color':'k'},line_kws={'color':'k','lw':0},x_bins=[3,10,30,50,100,300,1000,3000],fit_reg=False,ci=95,label='observations')

ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xscale('log')

plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
ax.set(ylim=(0,1.01),xlabel='age (years)',ylabel='cumulative fraction\nof soil carbon')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='RothC')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='CENTRUY')

# k2 = np.stack(k).squeeze()
# k3 = np.cumsum(k2,axis=1).T/np.sum(k2,axis=1).T
# plt.plot(ages,k3.mean(axis=1),label='CESM',c='k')
# plt.fill_between(ages,np.percentile(k3,0.025,axis=1),np.percentile(k3,97.5,axis=1),alpha=.3,color='k',lw=0)
# plt.plot(ages,cum_model_site_age_dist.mean(axis=1))
# pd.DataFrame(cum_model_site_age_dist.mean(axis=1),index=ages,columns=['CESM','IPSL','MRI']).plot(ax=ax)

# mat_data = pd.read_csv('../../data/age_distribution/pas_tropics_mathematica_20241101_1000000.csv',header=None)
# ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'disordered kinetics model')
mat_data = pd.read_csv('../../data/age_distribution/shortpas_tropics_mathematica_20241101_0.1_200_000.csv',header=None)
ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'our model')

N_CLM5 = (~np.isnan(cum_CLM5_site_age_dist.sum(axis=1))).sum()
CLM5_se = np.nanstd(cum_CLM5_site_age_dist,axis=0)/np.sqrt(N_CLM5-1)
CLM5_mean = np.nanmean(cum_CLM5_site_age_dist,axis=0)
lw =0.5
ax.plot(ts_CLM5[:-1],CLM5_mean,label='CLM5',c='C06',lw=lw)
ax.fill_between(ts_CLM5[:-1],CLM5_mean+1.96*CLM5_se,CLM5_mean-1.96*CLM5_se,alpha=.3,color='C06',lw=0)

DAYCENT_se = cum_DAYCENT_site_age_dist.std(axis=1)/np.sqrt(cum_DAYCENT_site_age_dist.shape[1]-1)
DAYCENT_mean = cum_DAYCENT_site_age_dist.mean(axis=1)
ax.plot(ts_DAYCENT,DAYCENT_mean,label='DAYCENT',c='C05',lw=lw)
ax.fill_between(ts_DAYCENT,DAYCENT_mean+1.96*DAYCENT_se,DAYCENT_mean-1.96*DAYCENT_se,alpha=.3,color='C05',lw=0)


Yasso_se = cum_Yasso_site_age_dist.std(axis=1)/np.sqrt(cum_Yasso_site_age_dist.shape[1]-1)
Yasso_mean = cum_Yasso_site_age_dist.mean(axis=1)
ax.plot(ts_Yasso,Yasso_mean,label='Yasso',c='C04',lw=lw)
ax.fill_between(ts_Yasso,Yasso_mean+1.96*Yasso_se,Yasso_mean-1.96*Yasso_se,alpha=.3,color='C04',lw=0)

roth_se = cum_RothC_site_age_dist.std(axis=1)/np.sqrt(cum_RothC_site_age_dist.shape[1]-1)
roth_mean = cum_RothC_site_age_dist.mean(axis=1)
ax.plot(ts_Roth,roth_mean,label='RothC',c='C03',lw=lw)
ax.fill_between(ts_Roth,roth_mean+1.96*roth_se,roth_mean-1.96*roth_se,alpha=.3,color='C03',lw=0)

plt.legend()

# cum_model_site_age_dist.mean(dim='site').plot.line(x='age',hue='model',ax=ax)
for model,c in zip(['CESM','IPSL','MRI'],['C00','C01','C02']):
    d = cum_model_site_age_dist.sel(model=model)
    ax.plot(ages2,d.mean(dim='site'),label=model,c=c,lw=lw)
    se = d.std(dim='site')/np.sqrt(d.shape[1]-1)
    ax.fill_between(ages2,d.mean(dim='site')+1.96*se,d.mean(dim='site')-1.96*se,alpha=.3,color=c,lw=0)

ax.set(xlim=[0.1,100_000])
plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(f'../../results/analyze_balesdent/IES_grant_2025_{datetime.now().strftime("%Y%m%d")}.svg',dpi=300)
plt.savefig(f'../../results/analyze_balesdent/IES_grant_2025_{datetime.now().strftime("%Y%m%d")}.png',dpi=300)

# %%
import seaborn as sns
fig,ax = plt.subplots(dpi=300)
# dat = nature_site_data[nature_site_data['total_fnew'].notna()]#.plot.scatter('Duration_labeling','total_fnew',ax=ax,label='observations',c='k')
sns.regplot(data=tropical_sites, x="Duration_labeling", y="total_fnew",ax=ax,scatter_kws={'color':'k'},line_kws={'color':'k','lw':0},x_bins=[3,10,30,50,100,300,1000,3000],fit_reg=False,ci=95)
# sns.regplot(data=dat, x="Duration_labeling", y="total_fnew",ax=ax,scatter_kws={'color':'k','alpha':0.5},line_kws={'color':'k','alpha':0.5,'lw':0},x_bins=np.logspace(np.log10(2),4,10),fit_reg=False,ci=95)
# mean.plot(ax=ax,c='k',marker='o',label='observations')
# ax.fill_between(std.index,mean-std,mean+std,alpha=.3,color='k',lw=0)
# ax.fill_between(std.index,mean-1.96*std,mean+1.96*std,alpha=.1,color='k',lw=0)
# ax.set_xscale('log')
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_xscale('log')

# for l,d in zip(['RothC','CENTURY','CESM','CLM4cn - needleleaf','CLM4cn - decideous','CLM4cn - tropical','ICBM','Yasso07','IPSL','MRI'],[RothC_p_age,CENTURY_p_age,CESM_p_age,CLM4cn_ndl_p_age,CLM4cn_dec_p_age,CLM4cn_trop_p_age,ICBM_p_age,Yasso07_p_age,IPSL_p_age,MRI_p_age]):
    # plt.plot(ages,np.cumsum(d)/d.sum(),label=l)
# stocks from https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GB005678
#NPP from https://esajournals-onlinelibrary-wiley-com.ezproxy.weizmann.ac.il/doi/10.1890/08-0588.1
# ps = [2886/50,4830]
#age tropics from Shi et al.
# age_tropics = 93/(93+102+103+107)*390+102/(93+102+103+107)*2790 + 103/(93+102+103+107)*510 + 107/(93+102+103+107)*2860
# age_tropics_high = 93/(93+102+103+107)*770+102/(93+102+103+107)*4310 + 103/(93+102+103+107)*1620 + 107/(93+102+103+107)*7960
# # ps_tropics = [(93+102+103+107)/34.1,15_000]
# ps_tropics = [30,7500]
# ps = ps_tropics
# # ts, p_age = generate_age_dist(*ps)
# ax.plot(ts,np.cumsum(p_age)/p_age.sum(),lw=3,c='Indianred',label=f'disordered kinetics model:\n $\\tau$={ps[0]:.2g}; age={ps[1]:.4g}')

# plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
ax.set(ylim=(0,1.01),xlabel='Age (years)',ylabel='Cummulative fraction of soil carbon')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='RothC')
# plt.plot(ages,np.cumsum(RothC_p_age)/RothC_p_age.sum(),c='k',label='CENTRUY')

# k2 = np.stack(k).squeeze()
# k3 = np.cumsum(k2,axis=1).T/np.sum(k2,axis=1).T
# plt.plot(ages,k3.mean(axis=1),label='CESM',c='k')
# plt.fill_between(ages,np.percentile(k3,0.025,axis=1),np.percentile(k3,97.5,axis=1),alpha=.3,color='k',lw=0)
# plt.plot(ages,cum_model_site_age_dist.mean(axis=1))
# pd.DataFrame(cum_model_site_age_dist.mean(axis=1),index=ages,columns=['CESM','IPSL','MRI']).plot(ax=ax)

# mat_data = pd.read_csv('../../data/age_distribution/pas_tropics_mathematica_20241101_1000000.csv',header=None)
# # ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'disordered kinetics model')
# mat_data = pd.read_csv('../../data/age_distribution/shortpas_tropics_mathematica_20241101_0.1_200_000.csv',header=None)
# ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Indianred',label=f'disordered kinetics model')

# mat_data = pd.read_csv('../../data/age_distribution/pas_tropics_powerlaw_mathematica_20241104_100000_0.1.csv',header=None)
# ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Royalblue',label=f'power law model')


# CLM_se = cum_CLM_site_age_dist.std(axis=1)/np.sqrt(cum_CLM_site_age_dist.shape[1]-1)
# CLM_mean = cum_CLM_site_age_dist.mean(axis=1)
# ax.plot(ts_CLM,CLM_mean,label='CLM',c='C06')
# ax.fill_between(ts_CLM,CLM_mean+1.96*CLM_se,CLM_mean-1.96*CLM_se,alpha=.3,color='C06',lw=0)
# N_CLM5 = (~np.isnan(cum_CLM5_site_age_dist.sum(axis=1))).sum()
# CLM5_se = np.nanstd(cum_CLM5_site_age_dist,axis=0)/np.sqrt(N_CLM5-1)
# CLM5_mean = np.nanmean(cum_CLM5_site_age_dist,axis=0)
# ax.plot(ts_CLM5[:-1],CLM5_mean,label='CLM5',c='C06')
# ax.fill_between(ts_CLM5[:-1],CLM5_mean+1.96*CLM5_se,CLM5_mean-1.96*CLM5_se,alpha=.3,color='C06',lw=0)




# DAYCENT_se = cum_DAYCENT_site_age_dist.std(axis=1)/np.sqrt(cum_DAYCENT_site_age_dist.shape[1]-1)
# DAYCENT_mean = cum_DAYCENT_site_age_dist.mean(axis=1)
# ax.plot(ts_DAYCENT,DAYCENT_mean,label='DAYCENT',c='C05')
# ax.fill_between(ts_DAYCENT,DAYCENT_mean+1.96*DAYCENT_se,DAYCENT_mean-1.96*DAYCENT_se,alpha=.3,color='C05',lw=0)


# Yasso_se = cum_Yasso_site_age_dist.std(axis=1)/np.sqrt(cum_Yasso_site_age_dist.shape[1]-1)
# Yasso_mean = cum_Yasso_site_age_dist.mean(axis=1)
# ax.plot(ts_Yasso,Yasso_mean,label='Yasso',c='C04')
# ax.fill_between(ts_Yasso,Yasso_mean+1.96*Yasso_se,Yasso_mean-1.96*Yasso_se,alpha=.3,color='C04',lw=0)

# roth_se = cum_RothC_site_age_dist.std(axis=1)/np.sqrt(cum_RothC_site_age_dist.shape[1]-1)
# roth_mean = cum_RothC_site_age_dist.mean(axis=1)
# ax.plot(ts_Roth,roth_mean,label='RothC',c='C03')
# ax.fill_between(ts_Roth,roth_mean+1.96*roth_se,roth_mean-1.96*roth_se,alpha=.3,color='C03',lw=0)

# plt.legend()

# # cum_model_site_age_dist.mean(dim='site').plot.line(x='age',hue='model',ax=ax)
# for model,c in zip(['CESM','IPSL','MRI'],['C00','C01','C02']):
#     d = cum_model_site_age_dist.sel(model=model)
#     ax.plot(ages2,d.mean(dim='site'),label=model,c=c)
#     se = d.std(dim='site')/np.sqrt(d.shape[1]-1)
#     ax.fill_between(ages2,d.mean(dim='site')+1.96*se,d.mean(dim='site')-1.96*se,alpha=.3,color=c,lw=0)

    # d = cum_model_site_age_dist_uncorrected.sel(model=model)
    # ax.plot(ages1,d.mean(dim='site'),label=model+' uncorrected',c=c,ls='--')
    # se = d.std(dim='site')/np.sqrt(d.shape[1]-1)
    # ax.fill_between(ages1,d.mean(dim='site')+1.96*se,d.mean(dim='site')-1.96*se,alpha=.3,color=c,lw=0)


    # ax.fill_between(ages, d.mean(dim='site')+d.std(dim='site'), d.mean(dim='site')-d.std(dim='site'),alpha=.3,color=c,lw=0)
    # plt.fill_between(ages,np.percentile(d.values,0.025,axis=1),np.percentile(d.values,97.5,axis=1),alpha=.3,color=c,lw=0)
# sns.regplot(data=cum_model_site_age_dist.to_dataframe(name='frac').reset_index(), x="age", y="frac",ax=ax,scatter_kws={'color':'k'},line_kws={'color':'k'},fit_reg=False,ci=95)
# plt.plot()
# plt.legend('')
ax.set(xlim=[0.1,100_000])


# %% [markdown]
# ## Build a BIC metric

# %%
np.argwhere(loglikelihoods[:,0] > 1e10)
tropical_sites.iloc[36][['Duration_labeling','total_fnew']] 
row= tropical_sites.iloc[36]
t = np.argwhere(ts==row['Duration_labeling'])
cdf[t]

norm(loc=cdf[t].mean(),scale=cdf[t].std()).pdf(row['total_fnew'])

# %%


# %%
# import functions for calcualting podf of normal distribution
from scipy.stats import norm

def parse_mathematica(f):
    df = pd.read_csv(f,header=None)
    t = df[0]
    cdf = np.cumsum(df[1])/df[1].sum()
    return t,cdf

ts_lognorm, lognorm_cdf = parse_mathematica('../../data/age_distribution/shortpas_tropics_mathematica_20241101_0.1_200_000.csv')
ts_power, power_cdf = parse_mathematica('../../data/age_distribution/pas_tropics_powerlaw_mathematica_20241104_100000_0.1.csv')

# tropical_sites[['Duration_labeling','total_fnew']]



# ax.plot(mat_data[0],np.cumsum(mat_data[1])/mat_data[1].sum(),lw=3,c='Royalblue',label=f'power law model')
names = ['RotC','Yasso','DAYCENT','CLM','log-normal','power-law','CESM_uncorrected','IPSL_uncorrected','MRI_uncorrected','CESM','IPSL','MRI']
cdfs = [cum_RothC_site_age_dist,cum_Yasso_site_age_dist,cum_DAYCENT_site_age_dist,cum_CLM_site_age_dist,lognorm_cdf,power_cdf,cum_model_site_age_dist_uncorrected.sel(model='CESM').values,cum_model_site_age_dist_uncorrected.sel(model='IPSL').values,cum_model_site_age_dist_uncorrected.sel(model='MRI').values,cum_model_site_age_dist.sel(model='CESM').values,cum_model_site_age_dist.sel(model='IPSL').values,cum_model_site_age_dist.sel(model='MRI').values]
tss = [ts_Roth,ts_Yasso,ts_DAYCENT,ts_CLM,ts_lognorm,ts_power,ages1,ages1,ages1,ages2,ages2,ages2]
k = tropical_sites.shape[0]
m = [6,]
loglikelihoods = np.zeros((k,len(tss)))
for i,(ts,cdf) in enumerate(zip(tss,cdfs)):
    for j,row in tropical_sites.reset_index().iterrows():
        t = np.argwhere(ts==row['Duration_labeling'])
        # print(j,k,row['Duration_labeling'])
        # check the pdf of the 'total_fnew' assuming a normal distribution with mean and std from cdf[k]
        # print(-np.log(norm(loc=cdf[t].mean(),scale=cdf[t].std()).pdf(row['total_fnew'])))
        loglikelihoods[j,i] = -np.log(norm(loc=cdf[k].mean(),scale=cdf[k].std()).pdf(row['total_fnew']))
    
pd.DataFrame(loglikelihoods,columns=names).sum()

# %%


# %%
cum_RothC_site_age_dist

# %%

cum_RothC_site_age_dist
cum_Yasso_site_age_dist
cum_DAYCENT_site_age_dist
cum_CLM_site_age_dist


cum_model_site_age_dist
cum_model_site_age_dist_uncorrected



# %% [markdown]
# ## Test

# %%
from scipy.integrate import  dblquad,nquad
import cubepy as cp
age = 75_000
to = 300
sig = np.sqrt(np.log(age/to))
mu  =-np.log(np.sqrt(to**3/age))
# mu
lognorm_pdf = lambda x: (1/(x*sig*np.sqrt(2*np.pi))) * np.exp(-((np.log(x)-mu)**2)/(2*sig**2))
f = lambda a,x: a * lognorm_pdf(x) * np.exp(-a*x) / np.exp(-mu + (sig**2)/2)

nquad(f,[[1e-3,np.inf],[1e-3,np.inf]],opts={'limit':10000})
# cp.integrate(f,low=[0,0],high=[np.inf,np.inf], itermax=100000)


# %%
# f(0,0)
a_s, x_s = np.meshgrid(np.linspace(0,1e-15,1_000),np.linspace(0,1e-15,1_000))

ps = f(a_s,x_s)

# %%


# %%
np.nanmax(ps)

# %%
import seaborn as sns
sns.heatmap(ps)

# %%



