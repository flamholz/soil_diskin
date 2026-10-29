"""Per-site inputs for step 02: soil radiocarbon, NPP and its allocation with depth.

Shared by the bulk (notebooks/02_get_turnover_14C.py) and depth-resolved
(notebooks/02c_get_layer_turnover_14C.py) preparation scripts:
  * sample_radiocarbon and radiocarbon_at: Delta14C profiles from the global map
    of Shi et al. (2020), checked against its published checksum by verify_shi;
  * read_npp and npp_at: site NPP from the table written by
    notebooks/02a_fetch_npp.py;
  * InputAllocation: how NPP is distributed over the top 100 cm of soil.

Tables passed in here use lowercase `latitude` and `longitude` columns.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr

from soil_diskin.utils import file_digest

COORDS = ['latitude', 'longitude']
SHI_MD5 = '645aa8d54cbc36cb329c29bcffc4352b'  # Published delta-14C file; metadata says 'year'.
SHI_URL = 'https://zenodo.org/records/3823612'
# NPP is matched to sites on coordinates rounded to this many decimals (~0.1 mm),
# so that last-bit differences between CSV parsers cannot break the match.
KEY_DECIMALS = 9


@dataclass(frozen=True)
class InputAllocation:
    """How the carbon input (NPP) is distributed over the top 100 cm of soil.

    A share `surface_fraction` of the input enters uniformly in 0-10 cm, and the
    rest is distributed exponentially with depth, with e-folding depth
    `input_depth_cm`, normalized over 0-100 cm. Only `soil_npp_fraction` of NPP
    enters the soil at all. Used by the depth-resolved preparation only.

    Attributes:
        input_depth_cm: e-folding depth of the exponential part, in cm (> 0).
        surface_fraction: share of the input that enters in 0-10 cm, in [0, 1).
        soil_npp_fraction: share of NPP that enters the soil, in (0, 1].
    """
    input_depth_cm: float = 30.
    surface_fraction: float = 0.
    soil_npp_fraction: float = 1.

    def __post_init__(self):
        if not np.isfinite(self.input_depth_cm) or self.input_depth_cm <= 0:
            raise ValueError('input_depth_cm must be finite and positive')
        if not 0 <= self.surface_fraction < 1:
            raise ValueError('surface_fraction must be in [0, 1)')
        if not 0 < self.soil_npp_fraction <= 1:
            raise ValueError('soil_npp_fraction must be in (0, 1]')

    def layer_input(self, npp, z_top_cm, z_bottom_cm):
        """Carbon input to the layer [z_top_cm, z_bottom_cm), in the units of `npp`.

        Integrates the input profile over the layer. The profile is normalized over
        0-100 cm whether or not the other layers of the site were reported, so the
        inputs to a site's reported layers sum to less than NPP when some depths are
        missing. Accepts scalars or broadcastable arrays; rows with NPP <= 0 or NaN,
        or bounds outside 0-100 cm, get NaN.
        """
        npp, top, bottom = np.broadcast_arrays(np.asarray(npp, float),
                                              np.asarray(z_top_cm, float), np.asarray(z_bottom_cm, float))
        valid = np.isfinite(npp) & (npp > 0) & (top >= 0) & (bottom <= 100) & (top < bottom)
        inputs = np.full(npp.shape, np.nan)
        top, bottom = top[valid], bottom[valid]
        h = self.input_depth_cm
        weights = np.exp(-top/h)*-np.expm1(-(bottom-top)/h)/-np.expm1(-100/h)
        weights *= 1-self.surface_fraction
        weights += self.surface_fraction*np.maximum(0, np.minimum(bottom, 10)-top)/10
        inputs[valid] = npp[valid]*self.soil_npp_fraction*weights
        return inputs.item() if inputs.ndim == 0 else inputs


# Baseline: half of NPP in 0–10 cm, half Jackson global (beta = 0.966) over 0–100 cm.
DEFAULT_ALLOCATION = InputAllocation(-1/np.log(.966), .5)


def verify_shi(path):
    """Raise unless the file at `path` is the published Shi et al. (2020) map; return its MD5."""
    checksum = file_digest(path, 'md5')
    if checksum != SHI_MD5:
        raise ValueError(f'Shi file does not match the published delta-14C checksum at {SHI_URL}')
    return checksum


def valid_coordinates(table):
    """True for rows with finite latitude in [-90, 90] and longitude in [-180, 180]."""
    lat, lon = table[COORDS].to_numpy(float).T
    return np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)


def sample_radiocarbon(shi, sites):
    """Delta14C (per mil) at each site and one-cm depth level from the Shi et al. map.

    Gaps in the map (NaN cells) are first filled with the nearest value at the same
    depth, and each site then takes the nearest grid cell. Sites with invalid
    coordinates get NaN.

    Args:
        shi: the Shi et al. (2020) dataset, with a `temp` variable (Delta14C,
            despite the 'year' units in its metadata) on level (0..99 cm), lat, lon.
        sites: table with `latitude` and `longitude` columns, one row per site.

    Returns:
        delta: array (n_sites, 100) of Delta14C; column z is the depth [z, z+1) cm.
        filled_depths: boolean array of the same shape, True where the value came
            from filling a gap rather than from the site's own grid cell.
    """
    if set(shi['temp'].dims) != {'level', 'lat', 'lon'}:
        raise ValueError('Shi temp must have level, lat, and lon dimensions')
    if not np.array_equal(np.sort(shi.level.values), np.arange(100)):
        raise ValueError('Shi levels must be the published 0..99 one-cm depth coordinates')
    shi = shi.sortby('level')
    for dim in ('lat', 'lon'):
        values = shi[dim].values
        if not np.isfinite(values).all() or len(np.unique(values)) != len(values):
            raise ValueError(f'Shi {dim} coordinates must be finite and unique')
        shi = shi.sortby(dim, ascending=(dim == 'lon'))  # Preserve rasterio's nearest-fill tie order.
    lat, lon = sites[COORDS].to_numpy(float).T
    valid = valid_coordinates(sites)
    delta = np.full((len(sites), 100), np.nan)
    filled_depths = np.zeros(delta.shape, dtype=bool)
    if valid.any():
        raster = shi['temp'].rename({'lat': 'y', 'lon': 'x'}).transpose('level', 'y', 'x')
        raster = raster.rio.write_crs('EPSG:4326').rio.write_nodata(np.nan)
        coords = {'y': xr.DataArray(lat[valid], dims='site'), 'x': xr.DataArray(lon[valid], dims='site')}
        native = raster.sel(coords, method='nearest').transpose('site', 'level').values
        sampled = raster.rio.interpolate_na(method='nearest').sel(coords, method='nearest').transpose('site', 'level').values
        delta = delta.astype(sampled.dtype, copy=False)  # Preserve filled-raster precision for bulk means.
        delta[valid] = sampled
        filled_depths[valid] = np.isnan(native) & np.isfinite(sampled)
    return delta, filled_depths


def radiocarbon_at(data, shi):
    """Delta14C profile and gap-filled flags (see sample_radiocarbon) for each row of `data`.

    The map is sampled once per location, since rows can share one (e.g. rows for
    different labeling durations, or the layers of one profile).
    """
    sites = data[COORDS].drop_duplicates().reset_index(drop=True)
    delta, filled = sample_radiocarbon(shi, sites)
    rows = data[COORDS].merge(sites.assign(site_index=sites.index), on=COORDS, how='left',
                              validate='many_to_one').site_index.to_numpy(int)
    return delta[rows], filled[rows]


def read_npp(path):
    """The NPP table of notebooks/02a_fetch_npp.py: latitude, longitude, NPP (g C/m²/yr)."""
    return pd.read_csv(path, float_precision='round_trip')


def location_keys(table):
    """Coordinates rounded to KEY_DECIMALS, for matching locations across tables."""
    return pd.DataFrame(table[COORDS].to_numpy(float).round(KEY_DECIMALS), columns=COORDS)


def npp_at(data, npp):
    """NPP (g C/m²/yr) for each row of `data`, from the table returned by read_npp.

    Rows with invalid coordinates get NaN. Raises if a row with valid coordinates
    has no location in the table, which means the table is out of date.
    """
    table = location_keys(npp).assign(NPP=npp.NPP.to_numpy())
    matched = location_keys(data).merge(table, on=COORDS, how='left', validate='many_to_one', indicator=True)
    valid = valid_coordinates(data)
    unknown = valid & matched._merge.ne('both').to_numpy()
    if unknown.any():
        missing = data.loc[unknown, COORDS].drop_duplicates()
        raise ValueError(f'{len(missing)} locations have no NPP; rerun 02a_fetch_npp.py:\n{missing}')
    return np.where(valid, matched.NPP.to_numpy(), np.nan)


def source_entry(path):
    """Resolved path and SHA-256 of an input file, for the provenance JSON of step 02."""
    return {'path': str(Path(path).resolve()), 'sha256': file_digest(path)}


def shi_provenance(md5):
    """Provenance fields describing the Shi et al. map and how it is sampled."""
    return {'shi_published_md5_verified': md5,
            'shi_metadata_units_override': 'published delta-14C in per mil; file says year',
            'radiocarbon_spatial_filling': 'rioxarray nearest at each one-cm depth before site selection; matches original 02_get_turnover_14C.py',
            'radiocarbon_reference_year': 2000, 'reference_year_basis': 'inherited model convention; absent from NetCDF metadata'}


def npp_provenance(path):
    """The NPP table's path and checksum, with the query settings from its JSON (see 02a_fetch_npp.py)."""
    meta_path = Path(path).with_suffix('.json')
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    keep = ['product', 'band', 'years', 'image_ids', 'mask_fill', 'footprint_m', 'statistic', 'units', 'retrieved_utc']
    return {**source_entry(path), **{key: meta[key] for key in keep if key in meta}}
