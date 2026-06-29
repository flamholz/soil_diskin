"""
Fetch environmental covariates for the lognormal-fit sites and save to CSV.

For each site (lat/lon) used in the lognormal calibration we sample:
  * WorldClim v1 bioclimatic variables (bio01-bio19) from `WORLDCLIM/V1/BIO`
  * SoilGrids edaphic factors (bulk density, pH, clay, CEC) from
    `projects/soilgrids-isric/*_mean`, aggregated as a 0-30 cm
    thickness-weighted mean of the 0-5, 5-15 and 15-30 cm layers.

The result is written to results/03_calibrate_models/site_env_covariates.csv so
that the correlation analysis is reproducible without re-querying Earth Engine.

Run from the project root.
"""
#%% libraries
import os
import numpy as np
import pandas as pd
import ee
import geemap
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

ee.Initialize(project=config['earth_engine']['project'])

#%% load the sites used in the lognormal fit
ln = pd.read_csv('results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv')
sites = ln.drop_duplicates(subset=['Longitude', 'Latitude'])[
    ['Latitude', 'Longitude']].reset_index(drop=True)
print(f"{len(sites)} unique sites")

# build an ee.FeatureCollection, tagging each feature with its row index so we
# can join the results back deterministically
features = [
    ee.Feature(ee.Geometry.Point([row['Longitude'], row['Latitude']]),
               {'site_id': int(i)})
    for i, row in sites.iterrows()
]
sites_ee = ee.FeatureCollection(features)


def sample_image(image, scale):
    """reduceRegions mean of an image over the sites, returned as a DataFrame
    indexed by site_id."""
    fc = image.reduceRegions(collection=sites_ee,
                             reducer=ee.Reducer.mean(),
                             scale=scale)
    df = geemap.ee_to_df(fc)
    return df.set_index('site_id').sort_index()


#%% WorldClim bioclim. Native res ~1 km; we sample at 1 km.
# bio01..bio19 with their conventional scale factors (temps are *10 in V1).
worldclim = ee.Image('WORLDCLIM/V1/BIO')
wc_df = sample_image(worldclim, scale=1000)
# WorldClim V1 stores temperatures (bio01,2,4,5,6,7,8,9,10,11) scaled by 10.
temp_bands = ['bio01', 'bio02', 'bio04', 'bio05', 'bio06',
              'bio07', 'bio08', 'bio09', 'bio10', 'bio11']
for b in temp_bands:
    if b in wc_df:
        wc_df[b] = wc_df[b] / 10.0
print("worldclim bands:", list(wc_df.columns))

#%% SoilGrids edaphic factors. 0-30cm thickness-weighted mean.
soilgrids_props = {
    'bdod': 0.01,   # cg/cm3   -> g/cm3
    'phh2o': 0.1,   # pH*10    -> pH
    'clay': 0.1,    # g/kg     -> %
    'cec': 0.1,     # mmol(c)/kg -> cmol(c)/kg
}
depths = ['0-5cm', '5-15cm', '15-30cm']
thick = np.array([5.0, 10.0, 15.0])

soil_frames = {}
for prop, scale_factor in soilgrids_props.items():
    img = ee.Image(f'projects/soilgrids-isric/{prop}_mean')
    bands = [f'{prop}_{d}_mean' for d in depths]
    df = sample_image(img.select(bands), scale=250)
    # thickness-weighted 0-30cm mean, then apply unit scaling
    vals = df[bands].to_numpy()
    weighted = (vals * thick).sum(axis=1) / thick.sum()
    soil_frames[f'{prop}_0-30cm'] = pd.Series(
        weighted * scale_factor, index=df.index)
    print(f"sampled soilgrids {prop}")

soil_df = pd.DataFrame(soil_frames)

#%% combine and save
env = sites.join(wc_df).join(soil_df)
out_path = 'results/03_calibrate_models/site_env_covariates.csv'
env.to_csv(out_path, index=False)
print(f"wrote {out_path} with shape {env.shape}")
print(env.head())
