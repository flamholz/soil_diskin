
#%% load libraries
import pandas as pd
import numpy as np
import os
import xarray as xr
import rioxarray as rio
import zipfile
import geemap
import ee
ee.Authenticate()  # Authenticate Earth Engine
ee.Initialize(project='diskin')

from soil_diskin.utils import download_file

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""
All scripts to be run from project root directory.

This script downloads and processes the 14C and NPP data for tropical sites,
which is used to calculate soil carbon turnover times. The processed data is
saved to a CSV file in the results folder.
"""

# # get data if not already downloaded, otherwise load it
# # get the 14C data from [Shi et al., 2020](https://zenodo.org/records/3823612)
c14_folder = "data/shi_2020/"
c14_data_filename = "global_delta_14C.nc"
# c14_url = "https://zenodo.org/records/3823612/files/global_delta_14C.nc"
# download_file(c14_url, c14_folder, c14_data_filename)

# # get the GPP data from [Kang et al. 2023](https://zenodo.org/records/8212707)
# GPP_data_filename = "ST_CFE-Hybrid_NT.zip"
# GPP_url = "https://zenodo.org/records/8212707/files/ST_CFE-Hybrid_NT.zip?download=1"
# download_file(GPP_url, GPP_folder, "ST_CFE-Hybrid_NT.zip")

# # unzip the GPP data
# if not os.path.exists(os.path.join(GPP_folder, "ST_CFE-Hybrid_NT")):
#     with zipfile.ZipFile(os.path.join(GPP_folder, "ST_CFE-Hybrid_NT.zip"), 'r') as zip_ref:
#         zip_ref.extractall(GPP_folder)
# else:
#     print("GPP data already unzipped")

#%% Load relevant data

# load the 14C data, add a CRS, and set the nodata value
c14_data = rio.open_rasterio(
    os.path.join(
        c14_folder,c14_data_filename),
        masked=True).rio.write_crs(
            "EPSG:4326", inplace=True).rio.write_nodata(np.nan)

# extrapolate the 14C data to fill in NaN values
c14_data_extrapolated = c14_data.rio.interpolate_na(method='nearest')

# # load the GPP data
# GPP = xr.open_mfdataset(GPP_folder+'ST_CFE-Hybrid_NT/CEDAR-GPP_v01_ST_CFE-Hybrid_NT_*.nc')

# # scale by the scale factor and take use the variable "GPP_mean" as the GPP data, and convert to gC/m2/year
# GPP = (GPP * GPP.SCALE_FACTOR)['GPP_mean'] * 365.25

# # calcualte the mean GPP over the time period, and set the nodata value and CRS
# GPP = GPP.where(GPP>0).mean(dim='time')\
#                 .rio.write_crs("EPSG:4326", inplace=True)\
#                 .rio.write_nodata(np.nan,inplace=True)

# load the tropical sites data
balesdent_fname = "results/processed_balesdent_2018.csv"
tropical_sites = pd.read_csv(balesdent_fname)
unique_tropical_coords = tropical_sites.drop_duplicates(subset=['Longitude','Latitude'])
site_C_weights = unique_tropical_coords.filter(regex='^weight_\d+').values

# use the coordinates in the unique_tropical_coords to create a list of ee.Geometry.Point objects
# unique_tropical_sites_ee = unique_tropical_coords[['Longitude','Latitude']].rename(columns={'Longitude':'x','Latitude':'y'})
unique_tropical_coords['geometry'] = unique_tropical_coords.apply(lambda row: ee.Geometry.Point([row['Longitude'], row['Latitude']]), axis=1)
# create a list of ee.Feature objects from the unique_tropical_sites
unique_tropical_sites_ee = ee.FeatureCollection(unique_tropical_coords.apply(lambda row: ee.Feature(row['geometry']), axis=1).tolist())

# load geemap data for npp 
MOD17A3HGF = ee.ImageCollection("MODIS/061/MOD17A3HGF")  

#%% get 14C and NPP data for the tropical sites

# site_14C = find_nearest(unique_tropical_coords[['Longitude','Latitude']],c14_data_extrapolated)
# site_GPP = find_nearest(unique_tropical_coords[['Longitude','Latitude']],GPP)

def extract_tropical_sites(ds):
    return xr.concat([ds.sel(y=row['Latitude'],x=row['Longitude'],method='nearest') for i,row in unique_tropical_coords.iterrows()],dim='site')

# extract 1m integrated 14C signal
site_1m_14C = np.nansum(extract_tropical_sites(c14_data_extrapolated).values.reshape((-1,10,10)).mean(axis=2) * site_C_weights, axis=1)

# get the NPP data for the tropical sites
# get the NPP data for the tropical sites
# scale factor for NPP from https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD17A3HGF#bands
# NPP is in kgC/m2/year, so we multiply by 1000 to convert to gC/m2/year
site_npp_data = MOD17A3HGF.select('Npp').mean().multiply(0.0001*1e3).reduceRegions( 
                        collection=unique_tropical_sites_ee,
                        reducer=ee.Reducer.mean(),
                        scale=10_000)
# Note: we use 10 km scale because the accuracy of the coordinates in the
# original Balesdent et al. (2018) dataset is not very high

site_npp_data = geemap.ee_to_df(site_npp_data)
site_npp_data.columns = ['NPP']

# merge SOC, 14C and NPP data
merged_site_data = unique_tropical_coords[['Latitude','Longitude']].reset_index()\
                            .merge(site_npp_data, left_index=True, right_index=True, how='left')\
                            .merge(pd.DataFrame(site_1m_14C, columns=['14C']), left_index=True, right_index=True, how='left')

merged_site_data = tropical_sites.merge(merged_site_data, on=['Latitude','Longitude'], how='left')
merged_site_data = merged_site_data[['Latitude','Longitude','14C','NPP','Ctotal_0-100estim']]

merged_site_data['fm'] = merged_site_data['14C'] / 1e3 + 1; # fm is the fraction of modern carbon, so we convert 14C from per mil to fraction
merged_site_data['turnover'] = merged_site_data['Ctotal_0-100estim'] * 1e3 / merged_site_data['NPP'] # turnover is SOC/NPP, so we convert NPP from kgC/m2/year to gC/m2/year

merged_site_data.to_csv('results/tropical_sites_14C_turnover.csv', index=False)
