"""Per-site inputs for step 02: soil radiocarbon, NPP and its allocation with depth.

Shared by the bulk (notebooks/02_get_turnover_14C.py) and depth-resolved
(notebooks/02c_get_layer_turnover_14C.py) preparation scripts:
  * sample_radiocarbon and radiocarbon_at: Delta14C profiles from the global map
    of Shi et al. (2020), and the grid cell each profile comes from (its
    published checksum is verified on download, in the Snakefile);
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


def valid_coordinates(table):
    """True for rows with finite latitude in [-90, 90] and longitude in [-180, 180]."""
    lat, lon = table[COORDS].to_numpy(float).T
    return np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)


def sample_radiocarbon(shi, sites):
    """Delta14C (per mil) at each site and one-cm depth level from the Shi et al. map.

    Gaps in the map (NaN cells) are first filled with the value of the nearest
    non-empty cell at the same depth, and each site then takes its nearest grid
    cell. Both "nearest"s are measured in degrees of latitude and longitude, not
    in km (rioxarray's interpolate_na). Sites with invalid coordinates get NaN.

    Args:
        shi: the Shi et al. (2020) dataset, with a `temp` variable (Delta14C,
            despite the 'year' units in its metadata) on level (0..99 cm), lat, lon.
        sites: table with `latitude` and `longitude` columns, one row per site.

    Returns:
        delta: array (n_sites, 100) of Delta14C; column z is the depth [z, z+1) cm.
        filled_depths: boolean array of the same shape, True where the value came
            from filling a gap rather than from the site's own grid cell.
        source: array (n_sites, 2) with the latitude and longitude of the grid
            cell the site's values come from: its own cell, or the cell its gaps
            were filled from. Raises if a site's values come from more than one
            cell (gaps filled from different cells at different depths).
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
    source = np.full((len(sites), 2), np.nan)
    if valid.any():
        raster = shi['temp'].rename({'lat': 'y', 'lon': 'x'}).transpose('level', 'y', 'x')
        raster = raster.rio.write_crs('EPSG:4326').rio.write_nodata(np.nan)
        coords = {'y': xr.DataArray(lat[valid], dims='site'), 'x': xr.DataArray(lon[valid], dims='site')}
        native = raster.sel(coords, method='nearest').transpose('site', 'level').values
        sampled = raster.rio.interpolate_na(method='nearest').sel(coords, method='nearest').transpose('site', 'level').values
        delta = delta.astype(sampled.dtype, copy=False)  # Preserve filled-raster precision for bulk means.
        delta[valid] = sampled
        filled_depths[valid] = np.isnan(native) & np.isfinite(sampled)
        source[valid] = _source_cells(raster, coords)
    return delta, filled_depths, source


def _source_cells(raster, coords):
    """Latitude and longitude of the cell each site's values come from (see sample_radiocarbon).

    Fills a raster of cell numbers, empty where the map is, exactly as the map is
    filled, so every cell holds the number of the cell its values come from.
    """
    ny, nx = raster.sizes['y'], raster.sizes['x']
    numbers = xr.DataArray(np.arange(ny*nx, dtype=float).reshape(ny, nx), dims=('y', 'x'),
                           coords={'y': raster.y, 'x': raster.x})
    numbers = numbers.where(raster.notnull()).transpose('level', 'y', 'x')
    numbers = numbers.rio.write_crs('EPSG:4326').rio.write_nodata(np.nan)
    cells = numbers.rio.interpolate_na(method='nearest').sel(coords, method='nearest').transpose('site', 'level').values
    source = np.full((cells.shape[0], 2), np.nan)
    for i, site_cells in enumerate(cells):
        unique = np.unique(site_cells[np.isfinite(site_cells)]).astype(int)
        if len(unique) > 1:
            raise ValueError(f'site {i} takes radiocarbon from {len(unique)} different cells at different depths')
        if len(unique):
            row, column = divmod(unique[0], nx)
            source[i] = raster.y.values[row], raster.x.values[column]
    return source


def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2
    return 2*6371.0*np.arcsin(np.sqrt(a))


def radiocarbon_at(data, shi):
    """Delta14C profile, gap-filled flags and source cell (see sample_radiocarbon) for each row of `data`.

    The map is sampled once per location, since rows can share one (e.g. rows for
    different labeling durations, or the layers of one profile).

    Returns:
        delta, filled_depths: as in sample_radiocarbon, one row per row of `data`.
        source: table with radiocarbon_source_latitude, radiocarbon_source_longitude
            and radiocarbon_source_km, the distance from the row's location to the
            source cell's centre; indexed like `data`.
    """
    sites = data[COORDS].drop_duplicates().reset_index(drop=True)
    delta, filled, source = sample_radiocarbon(shi, sites)
    rows = data[COORDS].merge(sites.assign(site_index=sites.index), on=COORDS, how='left',
                              validate='many_to_one').site_index.to_numpy(int)
    lat, lon = data[COORDS].to_numpy(float).T
    source = pd.DataFrame({'radiocarbon_source_latitude': source[rows, 0],
                           'radiocarbon_source_longitude': source[rows, 1]}, index=data.index)
    source['radiocarbon_source_km'] = distance_km(lat, lon, source.radiocarbon_source_latitude,
                                                  source.radiocarbon_source_longitude)
    return delta[rows], filled[rows], source


def read_npp(path):
    """The NPP table of notebooks/02a_fetch_npp.py: latitude, longitude, NPP (g C/m²/yr)."""
    return pd.read_csv(path, float_precision='round_trip')


def location_keys(table):
    """Coordinates rounded to KEY_DECIMALS, for matching locations across tables."""
    return pd.DataFrame(table[COORDS].to_numpy(float).round(KEY_DECIMALS), columns=COORDS)


def npp_at(data, npp, column='NPP'):
    """NPP (g C/m²/yr) for each row of `data`, from the table returned by read_npp.

    `column` selects another column of the table, e.g. NPP_sd.

    Rows with invalid coordinates get NaN. Raises if a row with valid coordinates
    has no location in the table, which means the table is out of date.
    """
    table = location_keys(npp).assign(NPP=npp[column].to_numpy())
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


def shi_provenance():
    """Provenance fields describing the Shi et al. map and how it is sampled."""
    return {'shi_metadata_units_override': 'published delta-14C in per mil; file says year',
            'radiocarbon_spatial_filling': 'rioxarray nearest at each one-cm depth before site selection; matches original 02_get_turnover_14C.py',
            'radiocarbon_reference_year': 2000, 'reference_year_basis': 'inherited model convention; absent from NetCDF metadata'}


def npp_provenance(path):
    """The NPP table's path and checksum, with the query settings from its JSON (see 02a_fetch_npp.py)."""
    meta_path = Path(path).with_suffix('.json')
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    keep = ['product', 'band', 'years', 'image_ids', 'mask_fill', 'radius_m', 'decimals', 'statistic', 'units', 'retrieved_utc']
    return {**source_entry(path), **{key: meta[key] for key in keep if key in meta}}
