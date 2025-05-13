import numpy as np
from os import path
import pandas as pd
import xarray as xr
import rioxarray as rio
import requests
import zipfile
import os

model_names = {'CESM':'CESM1','IPSL':'IPSL-CM5A-LR','MRI':'MRI-ESM1'}
file_names = {'CESM':'areacella_fx_CESM1-BGC_1pctCO2_r0i0p0','IPSL':'areacella_fx_IPSL-CM5A-LR_historicalNat_r0i0p0','MRI':'areacella_fx_MRI-ESM1_esmControl_r0i0p0'}

def download_he_data():
    """
    Download the HE data from the given URL and save it to the specified file path.
    """
    url = "https://git.bgc-jena.mpg.de/csierra/Persistence/-/archive/master/Persistence-master.zip"
    
    if not path.exists('data/he_2016/'):

        # Download the file
        response = requests.get(url)
        with open('he_2016.zip', 'wb') as f:
            f.write(response.content)

        # Unzip the file
        with zipfile.ZipFile('he_2016.zip', 'r') as zip_ref:
            zip_ref.extractall('data/he_2016/')

        # Remove the zip file
        os.remove('he_2016.zip')
        os.remove('data/he_2016/Persistence-master/CodeData/WorldGrids/')
    else:
        print("Data already downloaded.")
        return


def parse_he_data(model='CESM'):
    
    grid = xr.open_dataset(f'data/CMIP5/{model_names[model]}/{file_names[model]}.nc')['areacella']
    params_raster = xr.zeros_like(grid).T
    params_raster = params_raster.where(params_raster!=0)
    params_df = pd.read_csv(f'data/he_2016/Persistence-master/CodeData/He/{model}/compartmentalParameters.txt',sep=' ')
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

def get_RCM_A_u(params,model='CESM',tau_fac = {'CESM':3.7, 'IPSL':14, 'MRI':13},rs_fac = {'CESM':0.34, 'IPSL':0.07, 'MRI':0.34}, correct= True):
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

download_he_data()

def interpolate_ds(ds: xr.DataArray,p:str) -> xr.DataArray:
    """
    Interpolates the data to fill in missing values.

    Parameters
    ----------
    ds : xr.DataArray
        The data to be interpolated.
    p : str
        The parameter to be used for interpolation.

    Returns
    -------
    xr.DataArray
        The interpolated data.
    """

    return ds.sel(parameter=p)\
            .rio.write_nodata(np.nan)\
            .rio.set_spatial_dims('lon','lat')\
            .rio.write_crs(4326)\
            .rio.interpolate_na()

for model in ['CESM','IPSL','MRI']:
    mod = parse_he_data(model=model)
    extrapolated = xr.concat([interpolate_ds(mod,p) for p in mod.parameter],dim='parameter')
    extrapolated.to_netcdf(f'results/process_he_2016/{model}_params.nc')