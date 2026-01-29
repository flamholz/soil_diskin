
#%% load libraries
import pandas as pd
import numpy as np
import os
import xarray as xr
import rioxarray as rio
import geemap
import ee
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

ee.Authenticate()  # Authenticate Earth Engine
ee.Initialize(project=config['earth_engine']['project'])

from soil_diskin.utils import download_file

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""
All scripts to be run from project root directory.

This script downloads and processes the 14C and NPP data for tropical sites,
which is used to calculate soil carbon turnover times. The processed data is
saved to a CSV file in the results folder.
"""

c14_folder = "data/shi_2020/"
c14_data_filename = "global_delta_14C.nc"

#%% Load relevant data

# load the 14C data, add a CRS, and set the nodata value
c14_data = rio.open_rasterio(
    os.path.join(
        c14_folder,c14_data_filename),
        masked=True).rio.write_crs(
            "EPSG:4326", inplace=True).rio.write_nodata(np.nan)

# extrapolate the 14C data to fill in NaN values
c14_data_extrapolated = c14_data.rio.interpolate_na(method='nearest')

# load the tropical sites data
balesdent_fname = "results/processed_balesdent_2018.csv"
all_sites = pd.read_csv(balesdent_fname)
unique_coords = all_sites.drop_duplicates(subset=['Longitude','Latitude'])
site_C_weights = unique_coords.filter(regex='^weight_\d+').values

# use the coordinates in the unique_coords to create a list of ee.Geometry.Point objects
unique_coords.loc[:,'geometry'] = unique_coords.apply(lambda row: ee.Geometry.Point([row['Longitude'], row['Latitude']]), axis=1)
# create a list of ee.Feature objects from the unique_coords
unique_sites_ee = ee.FeatureCollection(unique_coords.apply(lambda row: ee.Feature(row['geometry']), axis=1).tolist())

# load geemap data for npp 
MOD17A3HGF = ee.ImageCollection("MODIS/061/MOD17A3HGF")  

#%% get 14C and NPP data for the tropical sites

def extract_sites(ds):
    return xr.concat([ds.sel(y=row['Latitude'], x=row['Longitude'], method='nearest')
                      for _, row in unique_coords.iterrows()], dim='site')

extracted_c14 = extract_sites(c14_data_extrapolated).values.reshape(
    (-1,10,10)).mean(axis=2)
# extract 1m integrated 14C signal
site_1m_14C = np.nansum(extracted_c14 * site_C_weights, axis=1)

# get the NPP data for the sites
# scale factor for NPP from https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD17A3HGF#bands
# NPP is in kgC/m2/year, so we multiply by 1000 to convert to gC/m2/year
site_npp_data = MOD17A3HGF.select('Npp').mean().multiply(0.0001*1e3).reduceRegions( 
                        collection=unique_sites_ee,
                        reducer=ee.Reducer.mean(),
                        scale=10_000)
# Note: we use 10 km scale because the accuracy of the coordinates in the
# original Balesdent et al. (2018) dataset is not very high

site_npp_data = geemap.ee_to_df(site_npp_data)
site_npp_data.columns = ['NPP']

# merge SOC, 14C and NPP data
single_col_14C_df = pd.DataFrame(site_1m_14C, columns=['14C'])
merged_site_data = unique_coords[['Latitude','Longitude']].reset_index().merge(
    site_npp_data, left_index=True, right_index=True, how='left').merge(
        single_col_14C_df, left_index=True, right_index=True, how='left')

# Merge with the original all_sites dataframe and retain only the
# relevant columns for output. Need to make sure we retain all sites 
# so that our output corresponds to the Balesdent et al. (2018) data
# for downstream analysis. 
merged_site_data = all_sites.merge(
    merged_site_data, on=['Latitude','Longitude'], how='left')
cols2keep = ['Latitude','Longitude','14C','NPP','Ctotal_0-100estim','Ctotal_0-100estim_q05',
             'Ctotal_0-100estim_q95']
merged_site_data = merged_site_data[cols2keep]

# fm is the fraction of modern carbon. to get this value we convert
# 14C from per mil to fraction.
merged_site_data['fm'] = merged_site_data['14C'] / 1e3 + 1; 

# turnover is SOC/NPP, so we convert NPP from kgC/m2/year to gC/m2/year
merged_site_data['turnover'] = merged_site_data['Ctotal_0-100estim'] * 1e3 / merged_site_data['NPP'] 
merged_site_data['turnover_q05'] = merged_site_data['Ctotal_0-100estim_q05'] * 1e3 / merged_site_data['NPP'] 
merged_site_data['turnover_q95'] = merged_site_data['Ctotal_0-100estim_q95'] * 1e3 / merged_site_data['NPP']

merged_site_data.to_csv('results/all_sites_14C_turnover.csv', index=False)
