import requests
import os
import numpy as np
import xarray as xr
import rioxarray as rio
from typing import Union


def download_file(url, folder, filename):
    """
    Download a file from a given URL and save it to a specified folder with a specified filename.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    folder : str
        The folder where the file will be saved.
    filename : str
        The name of the file to save.

    Returns
    -------
    None
    """
    
    # Create the folder if it doesn't exist
    if not os.path.exists(os.path.join(folder,filename)):
        os.makedirs(folder, exist_ok=True)
        
        # Download the file
        response = requests.get(url)
        with open(os.path.join(folder,filename), 'wb') as f:
            f.write(response.content)
    else:
        print("Data already downloaded")


# define function to calculate surface area of each pixel
def calc_pixel_area(raster:xr.DataArray) -> xr.DataArray:
    '''
    Calculate the area of each pixel in a raster

    Parameters:
    raster (xarray.DataArray): raster to calculate pixel area for

    Returns:
    xarray.DataArray: raster with pixel area as values
    '''

    # get the resolution of the raster
    res = raster.rio.resolution()

    l1 = np.radians(raster['y']- np.abs(res[1])/2)
    l2 = np.radians(raster['y']+ np.abs(res[1])/2)
    dx = np.radians(np.abs(res[0]))    
    _R = 6371e3  # Radius of earth in m. Use 3956e3 for miles

    # calculate the area of each pixel
    area = _R**2 * dx * (np.sin(l2) - np.sin(l1))

    # create a new xarray with the pixel area as values
    result = ((raster-raster+1)*area)

    # set the nodata value    
    if raster.rio.nodata is None:
        result.rio.set_nodata(np.nan,inplace=True)
    else:
        result.rio.set_nodata(raster.rio.nodata,inplace=True)
    
    return result
