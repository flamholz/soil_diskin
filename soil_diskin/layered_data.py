"""Profile-preserving inputs for complete or partially observed soil columns."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 -- registers the .rio accessor used by the original analysis
import xarray as xr

from .layered_lognormal import DZ, N_LAYERS

# Published checksum distinguishes delta-14C from the similarly named age file;
# the original delta-14C NetCDF incorrectly labels its variable units as 'year'.
# https://zenodo.org/records/3823612 (v1, checked 2026-09-23).
SHI_MD5 = '645aa8d54cbc36cb329c29bcffc4352b'
NPP_COORD_DECIMALS = 10  # Match CSV/Excel coordinate roundoff, not neighboring sites.


@dataclass
class PreparedProfiles:
    profiles: pd.DataFrame
    excluded: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def file_digest(path: str | Path, algorithm: str = 'sha256') -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def prepare_profiles(raw: pd.DataFrame, shi: xr.Dataset,
                     npp: pd.DataFrame, *, allow_partial: bool = False) -> PreparedProfiles:
    """Return usable layers and exclusions, preserving the original depth indices.

    ``raw`` is the Balesdent Profiles sheet after its seven introductory rows;
    ``shi.temp`` is delta-14C in per mil at levels 0..99 (one cm increments);
    ``npp.NPP`` is in g C/m²/yr. Shi spatial gaps are filled at each one-cm
    depth by nearest neighbor, matching 02_get_turnover_14C.py lines 36–67.
    Missing stocks and NPP are not filled.
    By default a profile needs all ten layers. With allow_partial, each layer
    needs stock and radiocarbon plus site NPP. A blank exclusion layer means
    the whole profile was excluded; otherwise only that layer was excluded.
    """
    if raw.Internal_profile_ID.isna().any() or raw.Internal_profile_ID.duplicated().any():
        raise ValueError('Internal_profile_ID must be nonmissing and unique')
    raw = raw.reset_index(drop=True)
    if set(shi['temp'].dims) != {'level', 'lat', 'lon'}:
        raise ValueError('Shi temp must have level, lat, and lon dimensions')
    if not np.array_equal(np.sort(shi.level.values), np.arange(100)):
        raise ValueError('Shi levels must be the published 0..99 one-cm depth coordinates')
    shi = shi.sortby('level')
    for dim in ('lat', 'lon'):
        values = shi[dim].values
        if not np.isfinite(values).all() or len(np.unique(values)) != len(values):
            raise ValueError(f'Shi {dim} coordinates must be finite and unique')
        # Match rasterio's north-to-south rows, including nearest-fill tie order.
        shi = shi.sortby(dim, ascending=(dim == 'lon'))
    coords = ['Latitude', 'Longitude']
    npp_values = npp[coords+['NPP']].copy()
    npp_values[coords] = npp_values[coords].apply(pd.to_numeric, errors='coerce').round(NPP_COORD_DECIMALS)
    npp_values = npp_values.drop_duplicates()
    if npp_values.duplicated(coords).any():
        raise ValueError('conflicting cached NPP values at the same coordinates')
    join_coords = raw[coords].apply(pd.to_numeric, errors='coerce').round(NPP_COORD_DECIMALS)
    joined = join_coords.merge(npp_values, on=coords, how='left', validate='many_to_one')
    inputs = pd.to_numeric(joined.NPP, errors='coerce').to_numpy(float)/1000
    lat = pd.to_numeric(raw.Latitude, errors='coerce').to_numpy(float)
    lon = pd.to_numeric(raw.Longitude, errors='coerce').to_numpy(float)
    valid_coords = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    delta = np.full((len(raw), N_LAYERS), np.nan)
    spatially_filled = np.zeros((len(raw), N_LAYERS), dtype=bool)
    if np.any(valid_coords):
        raster = shi['temp'].rename({'lat': 'y', 'lon': 'x'}).transpose('level', 'y', 'x')
        raster = raster.rio.write_crs('EPSG:4326').rio.write_nodata(np.nan)
        sample_coords = {'y': xr.DataArray(lat[valid_coords], dims='profile'),
                         'x': xr.DataArray(lon[valid_coords], dims='profile')}
        native = raster.sel(sample_coords, method='nearest').transpose('profile', 'level').values
        filled = raster.rio.interpolate_na(method='nearest')
        sampled = filled.sel(sample_coords, method='nearest').transpose('profile', 'level').values
        spatially_filled[valid_coords] = (np.isnan(native) & np.isfinite(sampled)).reshape(-1, 10, 10).any(axis=2)
        # Same arithmetic mean and float precision as original line 66; no depth interpolation.
        delta[valid_coords] = sampled.reshape(-1, 10, 10).mean(axis=2)
    fm = 1+delta/1000
    stocks = raw[[f'Ctotal_0-{z}' for z in range(0, 101, 10)]].apply(
        pd.to_numeric, errors='coerce').diff(axis=1).iloc[:, 1:].to_numpy(float)
    new_columns = ['Cnew_0_0']+[f'Cnew_0-{z}' for z in range(10, 101, 10)]
    new_stocks = raw.reindex(columns=new_columns).apply(
        pd.to_numeric, errors='coerce').diff(axis=1).iloc[:, 1:].to_numpy(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        new_fraction = new_stocks/stocks
    evaluation_valid = np.asarray(np.isfinite(new_fraction) & (new_fraction >= 0)
                                  & (new_fraction <= 1), dtype=bool)
    new_fraction[~evaluation_valid] = np.nan
    durations = pd.to_numeric(raw.get('Duration_labeling', pd.Series(np.nan, index=raw.index)),
                              errors='coerce').to_numpy(float)
    stock_valid = np.isfinite(stocks) & (stocks > 0)
    radio_valid = np.asarray(np.isfinite(fm), dtype=bool)
    usable_layers = stock_valid & radio_valid
    good_stock = stock_valid.all(axis=1)
    good_radio = np.asarray(radio_valid.all(axis=1), dtype=bool)
    good_npp = np.isfinite(inputs) & (inputs > 0)
    complete = valid_coords & good_stock & good_radio & good_npp
    eligible = valid_coords & good_npp & usable_layers.any(axis=1) if allow_partial else complete
    records: list[dict] = []
    exclusions: list[dict] = []
    for row in range(len(raw)):
        profile_id = str(raw.Internal_profile_ID.iloc[row])
        if not eligible[row]:
            reasons = [reason for valid, reason in [
                (valid_coords[row], 'invalid coordinates'),
                (good_stock[row], 'incomplete or nonpositive layer stocks'),
                (good_radio[row], 'incomplete radiocarbon after spatial filling'),
                (good_npp[row], 'missing or nonpositive NPP')] if not valid]
            exclusions.append({'profile_id': profile_id, 'reason': '; '.join(reasons)})
            continue
        for layer in range(N_LAYERS):
            if not usable_layers[row, layer]:
                reasons = [reason for valid, reason in [
                    (stock_valid[row, layer], 'missing or nonpositive layer stock'),
                    (radio_valid[row, layer], 'incomplete radiocarbon after spatial filling')] if not valid]
                exclusions.append({'profile_id': profile_id, 'layer': layer, 'reason': '; '.join(reasons)})
                continue
            records.append({'profile_id': profile_id, 'latitude': lat[row], 'longitude': lon[row],
                            'layer': layer, 'z_top_cm': layer*DZ, 'z_bottom_cm': (layer+1)*DZ,
                            'stock_kg_m2': stocks[row, layer], 'fm_obs': fm[row, layer],
                            'radiocarbon_spatially_filled': bool(spatially_filled[row, layer]),
                            'npp_kg_m2_yr': inputs[row], 'duration_years': durations[row],
                            'fnew_obs': new_fraction[row, layer],
                            'fnew_observation_valid': bool(evaluation_valid[row, layer])})
    columns = ['profile_id', 'latitude', 'longitude', 'layer', 'z_top_cm', 'z_bottom_cm',
               'stock_kg_m2', 'fm_obs', 'npp_kg_m2_yr', 'duration_years', 'fnew_obs',
               'fnew_observation_valid', 'radiocarbon_spatially_filled']
    metadata = {'input_profile_count': len(raw), 'eligible_profile_count': int(eligible.sum()),
                'allow_partial': allow_partial, 'eligible_layer_count': len(records),
                'complete_profile_count': int(complete.sum()),
                'partial_profile_count': int((eligible & ~complete).sum()),
                'radiocarbon_spatial_filling': 'rioxarray nearest at each one-cm depth before site selection; matches 02_get_turnover_14C.py',
                'radiocarbon_spatially_filled_raw_layer_count': int(spatially_filled.sum()),
                'radiocarbon_spatially_filled_layer_count': sum(r['radiocarbon_spatially_filled'] for r in records),
                'radiocarbon_depth_aggregation': 'ordinary mean of ten one-cm values after spatial filling; no depth interpolation',
                'radiocarbon_reference_year': 2000,
                'reference_year_basis': 'inherited model convention; absent from NetCDF metadata',
                'npp_conversion': 'cached g C/m²/yr divided by 1000 to kg C/m²/yr',
                'npp_coordinate_matching_decimals': NPP_COORD_DECIMALS,
                'npp_imputed': False,
                'fnew_observation': 'difference of cumulative Cnew divided by layer Ctotal'}
    return PreparedProfiles(pd.DataFrame(records, columns=columns),
                            pd.DataFrame(exclusions, columns=['profile_id', 'layer', 'reason']), metadata)


def load_profiles(balesdent_path: str | Path = 'data/balesdent_2018/balesdent_2018_raw.xlsx',
                  shi_path: str | Path = 'data/shi_2020/global_delta_14C.nc',
                  npp_path: str | Path = 'results/all_sites_14C_turnover.csv', *,
                  allow_partial: bool = False) -> PreparedProfiles:
    """Read existing local inputs and verify the published Shi product identity."""
    checksum = file_digest(shi_path, 'md5')
    if checksum != SHI_MD5:
        raise ValueError('Shi file does not match the published delta-14C checksum at '
                         'https://zenodo.org/records/3823612; use prepare_profiles for custom data')
    raw = pd.read_excel(balesdent_path, sheet_name='Profiles', skiprows=7)
    npp = pd.read_csv(npp_path)
    with xr.open_dataset(shi_path) as dataset:
        result = prepare_profiles(raw, dataset, npp, allow_partial=allow_partial)
    result.metadata['sources'] = {
        name: {'path': str(Path(path).resolve()), 'sha256': file_digest(path)}
        for name, path in [('balesdent', balesdent_path), ('shi', shi_path), ('npp', npp_path)]}
    result.metadata['shi_published_md5_verified'] = checksum
    result.metadata['shi_metadata_units_override'] = 'published delta-14C in per mil; file says year'
    return result
