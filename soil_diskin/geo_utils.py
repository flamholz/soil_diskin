from sklearn.neighbors import KDTree
from typing import Union
import numpy as np
import pandas as pd
import xarray as xr


def find_nearest(coords: Union[np.ndarray,pd.DataFrame],
                 ds: xr.DataArray,tolerance: float = 1) -> np.ndarray:
    '''
    Find the nearest pixel in a dataset to a set of coordinates

    Parameters
    ----------
    coords : Union[np.ndarray,pd.DataFrame]
        Coordinates to find the nearest pixel to
    ds : xr.DataArray
        Dataset to find the nearest pixel in
    tolerance : float, optional
        Tolerance for the distance to be considered a match. The default is 1.

    Returns
    -------
    np.ndarray
        Array of the nearest pixel values

    '''
    if type(coords) == pd.DataFrame:
        coords = coords.values
    
    # Convert the DataArray to a DataFrame and drop NaN values
    df = ds.to_dataframe().dropna()[ds.name]

    # build a KDTree from the DataFrame index
    # TODO: allow users to pass a pre-built KDTree to avoid rebuilding it every time
    kdt = KDTree(np.stack(df.index.to_numpy()),leaf_size=30, metric='euclidean')
    
    # query the KDTree for the nearest pixel
    distance,ind = kdt.query(coords,k=1)
    
    # get the values of the nearest pixel
    res = df.values[ind]

    # set the values to NaN if the distance is greater than the tolerance
    res[distance>tolerance] = np.nan

    return res