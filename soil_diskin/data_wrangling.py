import pandas as pd
import numpy as np
import xarray as xr


model_names = {'CESM':'CESM1','IPSL':'IPSL-CM5A-LR','MRI':'MRI-ESM1'}
file_names = {'CESM':'areacella_fx_CESM1-BGC_1pctCO2_r0i0p0',
              'IPSL':'areacella_fx_IPSL-CM5A-LR_historicalNat_r0i0p0',
              'MRI':'areacella_fx_MRI-ESM1_esmControl_r0i0p0'}


def parse_he_data(model='CESM', file_names=None) -> xr.DataArray:
    """
    Reads the He et al. 2016 parameter values into an xarray DataArray.
    """
    grid = xr.open_dataset(f'data/CMIP5/{model_names[model]}/{file_names[model]}.nc')['areacella']
    params_raster = xr.zeros_like(grid).T
    params_raster = params_raster.where(params_raster!=0)
    
    params_fname = f'data/he_2016/Persistence-master/CodeData/He/{model}/compartmentalParameters.txt'
    params_df = pd.read_csv(params_fname, sep=' ')
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


def process_balesdent_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the raw Balesdent et al. 2018 data.

    Args:
        raw_data (pd.DataFrame): The raw data loaded from the Excel file.

    Returns:
        pd.DataFrame: The processed data.
    """
    # The Balesdent 2018 dataset has columns with names like Ctotal_0-10,
    # Ctotal_0-20, C_total_0-30, etc. that give the total carbon content in 
    # a depth range of the soil column. We first use these to calculate the
    # density of carbon in each 10 cm layer.
    #
    # Take only the columns with Ctotal_ and end with a digit but can have
    # other characters in between, and calculate the difference between the
    # columns as an estimate of the density of C in each layer
    end_depths = list(range(10, 110, 10))
    cols_of_interest = [f'Ctotal_0-{d}' for d in end_depths]
    C_dens = raw_data[cols_of_interest].diff(axis=1).iloc[:, 1:]

    # Calculate the weighting for each layer as a fraction of the carbon
    # density in the layer out of the total carbon in the top 1 meter of
    # the specific site. 
    mean_C_weights = (C_dens.mean() / C_dens.mean().sum()).values
    site_C_weights = (C_dens.div(C_dens.sum(axis=1), axis=0)).values

    # For sites with no C density data, we take the mean C density of all
    # sites in each layer divided by the mean C density in the top 1 meter
    # of all sites.
    site_C_weights[np.nansum(site_C_weights,axis=1) == 0,:] = mean_C_weights

    # Add the weights to the raw dataframe, with the layer defined by its 
    # upper depth limit, e.g., weight_0 is the weight for the 0-10 cm layer
    weight_columns = [f'weight_{d}' for d in end_depths[:-1]]
    site_C_weights_df = pd.DataFrame(site_C_weights, columns=weight_columns,
                                     index=raw_data.index)
    all_sites = pd.merge(raw_data, site_C_weights_df, left_index=True, right_index=True)

    # Restrict to columns of the form f_10, f_20, ..., f_100. These columns
    # give the fraction of new C in the soil column at a specific depth after
    # a certain duration of labeling.
    f_columns_of_interest = [f'f_{d}' for d in end_depths]
    f_data = all_sites[f_columns_of_interest]

    # Calculate mean fraction of new C across layers for each site by using
    # the weights. We first take a rolling mean of the fraction of new C
    # across layers, because the fraction of new C is calculated for a
    # specific depth while the C content is calculated for a depth interval. 
    layer_f_data = f_data.T.rolling(2).mean().T.values[:, 1:]
    all_sites.loc[:, 'total_fnew'] = np.nansum(layer_f_data * site_C_weights, axis=1)

    # There are 11 sites with entirely missing C density data. 
    # We could potentiall fill in from SoilGrids or other global datasets.
    # For now we omit them. 
    all_Ctotal_cols = cols_of_interest + ['Ctotal_0-100estim']
    all_sites = all_sites[~(all_sites[all_Ctotal_cols].isna().all(axis=1))]

    # Note: in the real data, Ctotal_0-100estim is present when there 
    # is some site level C density data, and absent when there is none.

    # Calculate data for unique sites 
    group_cols = ['Latitude','Longitude','Duration_labeling']
    output_cols = ['total_fnew'] + weight_columns + ['Ctotal_0-100estim']
    return all_sites.groupby(group_cols)[output_cols].mean().reset_index()


def point_in_polygon(x, y, polygon_coords):
    """Check if point (x,y) is inside polygon using ray casting algorithm"""
    n = len(polygon_coords)
    inside = False
    
    p1x, p1y = polygon_coords[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_coords[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def distance_to_polygon(x, y, polygon_coords):
    """Calculate minimum distance from point to polygon boundary"""
    min_dist = float('inf')
    
    for i in range(len(polygon_coords)):
        p1 = polygon_coords[i]
        p2 = polygon_coords[(i + 1) % len(polygon_coords)]
        
        # Distance from point to line segment
        A = x - p1[0]
        B = y - p1[1]
        C = p2[0] - p1[0]
        D = p2[1] - p1[1]
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq != 0:
            param = dot / len_sq
        else:
            param = -1
        
        if param < 0:
            xx, yy = p1[0], p1[1]
        elif param > 1:
            xx, yy = p2[0], p2[1]
        else:
            xx = p1[0] + param * C
            yy = p1[1] + param * D
        
        dx = x - xx
        dy = y - yy
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = min(min_dist, dist)
    
    return min_dist

def assign_biome_numpy(row, biome_data):
    """Assign biome using numpy-only approach"""
    x, y = row['MAT_C'], row['PANN_mm']/10  # Convert precipitation to cm
    
    # Check if point is inside any polygon
    for biome, group in biome_data.groupby('biome'):
        coords = group[['x', 'y']].values
        if point_in_polygon(x, y, coords):
            return biome
    
    # If not inside any polygon, find closest
    min_distance = float('inf')
    closest_biome = None
    
    for biome, group in biome_data.groupby('biome'):
        coords = group[['x', 'y']].values
        distance = distance_to_polygon(x, y, coords)
        if distance < min_distance:
            min_distance = distance
            closest_biome = biome
    
    return closest_biome