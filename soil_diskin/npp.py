"""Site NPP from MODIS MOD17A3HGF via Google Earth Engine.

The only code in the project that queries Earth Engine. Used by
notebooks/02a_fetch_npp.py and notebooks/experimental/calibrate_lognormal_israd.py.
Requires Earth Engine credentials (`earthengine authenticate`) and the project
named under earth_engine.project in config.yaml.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = 'MODIS/061/MOD17A3HGF'
BAND = 'Npp'
SCALE_TO_G_M2 = 0.0001*1e3  # band units of 0.0001 kg C/m²/yr -> g C/m²/yr
FILL_MIN = 32700  # values at or above this are fill codes (water, urban, barren, ...)
FOOTPRINT_M = 10_000


def initialize(config_path=ROOT/'config.yaml'):
    """Authenticate with Earth Engine and initialize it with the configured project."""
    import ee
    config = yaml.safe_load(Path(config_path).read_text())
    ee.Authenticate()
    ee.Initialize(project=config['earth_engine']['project'])


def annual_images(years=None, mask_fill=False):
    """The annual NPP images, optionally restricted to `years` (first, last) and with fill values masked."""
    import ee
    annual = ee.ImageCollection(PRODUCT).select(BAND)
    if years is not None:
        annual = annual.filter(ee.Filter.calendarRange(years[0], years[1], 'year'))
    if mask_fill:
        annual = annual.map(lambda image: image.updateMask(image.lt(FILL_MIN)))
    return annual


def image_ids(years=None):
    """IDs of the annual images that would be averaged (e.g. '2001_01_01'), for provenance."""
    return annual_images(years).aggregate_array('system:index').getInfo()


def fetch_modis_npp(locations, years=None, mask_fill=False, sd=False, scale=FOOTPRINT_M):
    """NPP at each location in g C/m²/yr: the mean of the annual images, averaged over `scale` meters.

    Each point is tagged with its row number and the results are matched back on
    it, rather than relying on the order Earth Engine returns them in. Call
    initialize() first.

    Args:
        locations: table with latitude and longitude columns.
        years: (first, last) calendar years to average, or None for all images.
        mask_fill: mask fill codes before averaging. Without it, a fill pixel
            inside the footprint is averaged in as ~3277 g C/m²/yr.
        sd: also return NPP_sd, the standard deviation across annual images
            (interannual variability, not the error of the product).
        scale: footprint, in meters, over which each location is averaged.

    Returns:
        `locations` with an NPP column (NaN where no valid pixel is in the
        footprint), and NPP_sd if requested.
    """
    import ee
    import geemap
    points = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {'row': i})
        for i, (lat, lon) in enumerate(locations[['latitude', 'longitude']].to_numpy(float))])
    annual = annual_images(years, mask_fill)
    if sd:
        stats = annual.reduce(ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True))
        stats, columns = stats.multiply(SCALE_TO_G_M2), {'Npp_mean': 'NPP', 'Npp_stdDev': 'NPP_sd'}
    else:
        stats, columns = annual.mean().multiply(SCALE_TO_G_M2), {'mean': 'NPP'}
    table = geemap.ee_to_df(stats.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=scale))
    table = table.set_index('row').reindex(range(len(locations)))
    # A property is absent when no point has a valid pixel; anything else is unexpected.
    if any(source not in table for source in columns) and len(table.columns):
        raise ValueError(f'unexpected Earth Engine columns: {list(table.columns)}')
    out = locations.copy()
    for source, name in columns.items():
        out[name] = table[source].to_numpy(float) if source in table else np.nan
    return out
