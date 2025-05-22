#%% 00 - load libraries
import os
import pandas as pd
import numpy as np
import requests

#%% 01 - Download raw data from [Baledant et al. 2018](https://www.nature.com/articles/s41586-018-0328-3) if not already downloaded

folder = "../data/balesdant_2018/"
raw_data_filename = "41586_2018_328_MOESM3_ESM.xlsx"
site_metadata_filename = "Soil_natural_13C_labelling_profiles_Database.xlsx"
raw_data_url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0328-3/MediaObjects/41586_2018_328_MOESM3_ESM.xlsx"


# Create the folder if it doesn't exist
if not os.path.exists(os.path.join(folder,raw_data_filename)):
    os.makedirs(folder, exist_ok=True)
    
    # Download the file
    response = requests.get(raw_data_url)
    with open(os.path.join(folder,raw_data_filename), 'wb') as f:
        f.write(response.content)
else:
    print("Data already downloaded")

#%% 02 - Load and process the data

# Load the raw data
raw_data = pd.read_excel(os.path.join(folder,raw_data_filename),skiprows=7)

# Take only sites with MAT>17, PANN>1000, P/PET>0.8
tropical_sites = raw_data.query('MAT_C>17.0 & PANN_mm>1000.0 & `P to PET ratio` >0.8')

# Take only the columns with Ctotal_ and end with a digit but can have other characters in between, 
# and calculate the difference between the columns as an estimate of the density of C in each layer
C_dens = tropical_sites.filter(regex='^Ctotal_\d+.*\d$').diff(axis=1).iloc[:,1:]

# Calculate the weighting for each layer as a fraction of the carbon density in the layer out 
# of the total carbon in the top 1 meter of the specific site. For sites with no C density
# data, we take the mean C density of all sites in each layer divided by the mean C density in
# the top 1 meter of all sites.
mean_C_weights = (C_dens.mean() / C_dens.mean().sum()).values
site_C_weights = (C_dens.div(C_dens.sum(axis=1), axis=0)).values
site_C_weights[np.nansum(site_C_weights,axis=1) == 0,:] = mean_C_weights

# Filter columns that start with f_ and then a number between 1 and 3 digits to get the fraction of new C in each layer
f_data = tropical_sites.filter(regex='^f_\d{1,3}$')

# Calculate mean fraction of new C across layers for each site by using the weights. We first take a rolling
# mean of the fraction of new C across layers, because the fraction of new C is calculated in a depth point, 
# and the C content is calculated for a depth interval. 
layer_f_data = f_data.rolling(2, axis=1).mean().values[:, 1:]
tropical_sites.loc[:, 'total_fnew'] = np.nansum(layer_f_data * site_C_weights, axis=1)

# Calculate data for unique sites
final_data = tropical_sites.groupby(['Latitude','Longitude','Duration_labeling'])['total_fnew'].mean().reset_index() 

# Save the data
final_data.to_csv(os.path.join('../results/processed_balesdant_2018.csv'), index=False)