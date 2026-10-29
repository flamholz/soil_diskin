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
# Annual images averaged. Pinned so that new releases do not change NPP; 2001-2025
# were all the images available when this was set (2026-10-29).
YEARS = (2001, 2025)
# NPP at a location is the mean of the native (~463 m, sinusoidal) pixels whose
# centres lie within RADIUS_M of it. An earlier version read the point at a 10 km
# scale, which takes a pixel of Earth Engine's image pyramid; which pyramid level
# was used depended on the other points in the same request, so a site's NPP could
# change (by up to ~12%) when sites were added or removed. The disk is reproducible
# across requests up to the order of summation (~1e-15 relative), which rounding
# to DECIMALS removes.
RADIUS_M = 5_000
DECIMALS = 6  # g C/m²/yr


def initialize(config_path=ROOT/'config.yaml'):
    """Authenticate with Earth Engine and initialize it with the configured project."""
    import ee
    config = yaml.safe_load(Path(config_path).read_text())
    ee.Authenticate()
    ee.Initialize(project=config['earth_engine']['project'])


def annual_images(years=YEARS):
    """The annual NPP images of `years` (first, last; None for all), with fill codes masked.

    The mask is defensive: Earth Engine's copy of MOD17A3HGF already marks fill codes
    as no-data (checked 2026-10-29: no value >= FILL_MIN within 5 km of any
    Balesdent site, and none over the ocean), so it changes nothing today. Without
    it, a fill code reaching an average would count as ~3277 g C/m²/yr.
    """
    import ee
    annual = ee.ImageCollection(PRODUCT).select(BAND)
    if years is not None:
        annual = annual.filter(ee.Filter.calendarRange(years[0], years[1], 'year'))
    return annual.map(lambda image: image.updateMask(image.lt(FILL_MIN)))


def image_ids(years=YEARS):
    """IDs of the annual images that would be averaged (e.g. '2001_01_01'), for provenance."""
    return annual_images(years).aggregate_array('system:index').getInfo()


def fetch_modis_npp(locations, years=YEARS, sd=False, radius=RADIUS_M):
    """NPP at each location in g C/m²/yr: the mean over a disk of the mean of the annual images.

    Each pixel's NPP is averaged over the annual images of `years` (fill codes
    masked), and the native-resolution pixels whose centres lie within `radius`
    meters of the location are averaged (see RADIUS_M). Each disk is tagged with
    its row number and the results are matched back on it, rather than relying on
    the order Earth Engine returns them in. Call initialize() first.

    Args:
        locations: table with latitude and longitude columns.
        years: (first, last) calendar years to average, or None for all images.
        sd: also return NPP_sd, the disk mean of each pixel's standard deviation
            across annual images (interannual variability, not the error of the product).
        radius: disk radius in meters.

    Returns:
        `locations` with NPP (rounded to DECIMALS; NaN if the disk has no valid
        pixel), NPP_sd if requested, n_valid_pixels and valid_share (the share of
        the disk's pixels with data; small near coasts, water and cities).
    """
    import ee
    import geemap
    disks = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(lon), float(lat)]).buffer(radius), {'row': i})
        for i, (lat, lon) in enumerate(locations[['latitude', 'longitude']].to_numpy(float))])
    annual = annual_images(years)
    native = annual.first().projection()
    if sd:
        stats = annual.reduce(ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)).rename(['NPP', 'NPP_sd'])
    else:
        stats = annual.mean().rename('NPP')
    # The constant band counts every pixel of the disk, with data or not.
    image = stats.multiply(SCALE_TO_G_M2).addBands(ee.Image.constant(1).rename('pixel'))
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
    table = geemap.ee_to_df(image.reduceRegions(collection=disks, reducer=reducer, crs=native,
                                                scale=native.nominalScale()))
    table = table.set_index('row').reindex(range(len(locations)))
    columns = {'NPP_mean': 'NPP', **({'NPP_sd_mean': 'NPP_sd'} if sd else {}), 'NPP_count': 'n_valid_pixels'}
    unexpected = set(table.columns) - set(columns) - {'NPP_sd_count', 'pixel_mean', 'pixel_count'}
    if unexpected:
        raise ValueError(f'unexpected Earth Engine columns: {sorted(unexpected)}')
    out = locations.copy()
    for source, name in columns.items():  # NPP_mean is absent when no disk has data
        out[name] = table[source].to_numpy(float) if source in table else np.nan
    for name in ['NPP', 'NPP_sd']:
        if name in out:
            out[name] = out[name].round(DECIMALS)
    out['n_valid_pixels'] = out.n_valid_pixels.fillna(0).astype(int)
    out['valid_share'] = out.n_valid_pixels/table['pixel_count'].to_numpy(float)
    return out
