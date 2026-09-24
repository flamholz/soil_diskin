"""Prepare a reusable depth-resolved turnover table; select its usable layers."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr

from .layered_lognormal import InputAllocation, N_LAYERS

SHI_MD5 = '645aa8d54cbc36cb329c29bcffc4352b'  # Published delta-14C file, despite its 'year' units.
NPP_COORD_DECIMALS = 10  # CSV/Excel roundoff, not neighboring sites.


@dataclass
class PreparedProfiles:
    profiles: pd.DataFrame
    excluded: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def file_digest(path: str | Path, algorithm: str = 'sha256') -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def allocate_inputs(layers: pd.DataFrame, distribution=InputAllocation()) -> pd.DataFrame:
    """Call distribution with metadata columns, then calculate stock/input turnover.

    The callable receives only coordinates, depth bounds, land use, vegetation,
    and site NPP. Return one input per row (kg C/m²/year), in the same order.
    Missing layers retain their depths; never normalize over retained rows.
    """
    columns = ['latitude', 'longitude', 'z_top_cm', 'z_bottom_cm',
               'land_use', 'vegetation', 'npp_kg_m2_yr']
    arguments = layers.reindex(columns=columns).copy()
    arguments['npp_kg_m2_yr'] = arguments.npp_kg_m2_yr.where(np.isfinite(arguments.npp_kg_m2_yr) & arguments.npp_kg_m2_yr.gt(0))
    valid_depth = (layers.z_top_cm.ge(0) & layers.z_bottom_cm.le(100)
                   & layers.z_bottom_cm.gt(layers.z_top_cm))
    selected = arguments.loc[valid_depth]
    values = np.asarray(distribution(**selected.to_dict('series')), dtype=float)
    if values.shape != (len(selected),) or np.isinf(values).any() or (values < 0).any():
        raise ValueError('NPP distribution must return one nonnegative input per row; missing values may be NaN')
    inputs = np.full(len(layers), np.nan)
    inputs[valid_depth] = values
    settings = asdict(distribution) if isinstance(distribution, InputAllocation) else {
        'input_depth_cm': np.nan, 'surface_fraction': np.nan, 'soil_npp_fraction': np.nan}
    result = layers.assign(**settings, npp_distribution=getattr(distribution, '__name__', type(distribution).__name__),
                           input_kg_m2_yr=inputs)
    result['implied_turnover_years'] = result.stock_kg_m2/result.input_kg_m2_yr.where(result.input_kg_m2_yr.gt(0))
    return result


def prepare_turnover(layers: pd.DataFrame, shi: xr.Dataset, npp: pd.DataFrame,
                     distribution=InputAllocation()) -> pd.DataFrame:
    """Attach cached site NPP and original-notebook Shi targets, then allocate inputs."""
    sites = layers[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
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
    coords = ['latitude', 'longitude']
    npp = npp.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    npp_values = npp[coords+['NPP']].copy()
    npp_values[coords] = npp_values[coords].apply(pd.to_numeric, errors='coerce').round(NPP_COORD_DECIMALS)
    npp_values = npp_values.drop_duplicates()
    if npp_values.duplicated(coords).any():
        raise ValueError('conflicting cached NPP values at the same coordinates')
    join_coords = sites[coords].apply(pd.to_numeric, errors='coerce').round(NPP_COORD_DECIMALS)
    joined = join_coords.merge(npp_values, on=coords, how='left', validate='many_to_one')
    inputs = pd.to_numeric(joined.NPP, errors='coerce').to_numpy(float)/1000
    lat, lon = sites[coords].to_numpy(float).T
    valid_coords = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    delta = np.full((len(sites), 100), np.nan)
    fm = np.full((len(sites), N_LAYERS), np.nan)
    filled_depths = np.zeros((len(sites), 100), dtype=bool)
    if np.any(valid_coords):
        raster = shi['temp'].rename({'lat': 'y', 'lon': 'x'}).transpose('level', 'y', 'x')
        raster = raster.rio.write_crs('EPSG:4326').rio.write_nodata(np.nan)
        sample_coords = {'y': xr.DataArray(lat[valid_coords], dims='profile'),
                         'x': xr.DataArray(lon[valid_coords], dims='profile')}
        native = raster.sel(sample_coords, method='nearest').transpose('profile', 'level').values
        filled = raster.rio.interpolate_na(method='nearest')
        sampled = filled.sel(sample_coords, method='nearest').transpose('profile', 'level').values
        filled_depths[valid_coords] = np.isnan(native) & np.isfinite(sampled)
        delta[valid_coords] = sampled
        # Preserve the filled raster's arithmetic precision in the original path.
        fm[valid_coords] = 1+sampled.reshape(-1, N_LAYERS, 10).mean(axis=2)/1000
    sites['site_index'] = np.arange(len(sites))
    result = layers.merge(sites, on=['latitude', 'longitude'], how='left', validate='many_to_one')
    row, layer = result.site_index.to_numpy(int), result.layer.to_numpy(int)
    result['npp_kg_m2_yr'] = inputs[row]
    if 'zmid_cm' in result:
        # Evaluate at the reported depth; fractional depths interpolate adjacent cm.
        depths = result.zmid_cm.to_numpy(float)
        result['fm_obs'] = [1+np.interp(z, shi.level, delta[i], left=np.nan, right=np.nan)/1000
                            for i, z in zip(row, depths)]
        result['radiocarbon_spatially_filled'] = [
            bool(filled_depths[i, int(np.floor(z)):int(np.ceil(z))+1].any()) if 0 <= z <= 99 else False
            for i, z in zip(row, depths)]
    else:
        # Same arithmetic mean and float precision as original line 66.
        result['fm_obs'] = fm[row, layer]
        result['radiocarbon_spatially_filled'] = filled_depths.reshape(-1, N_LAYERS, 10).any(axis=2)[row, layer]
    return allocate_inputs(result.drop(columns='site_index'), distribution)


def prepare_profiles(layers: pd.DataFrame, *, allow_partial: bool = False) -> PreparedProfiles:
    """Select fit-ready rows; missing f_new never excludes calibration data."""
    sampled = 'zmid_cm' in layers
    if (layers.profile_id.isna().any() or layers.duplicated(['profile_id', 'layer']).any()
            or not (layers.layer.ge(0) & layers.layer.mod(1).eq(0)).all()):
        raise ValueError('expected unique profile/layer keys with nonnegative integer layer indices')
    if not sampled and (not layers.layer.isin(range(10)).all()
                         or not layers.z_top_cm.eq(layers.layer*10).all()
                         or not layers.z_bottom_cm.eq((layers.layer+1)*10).all()):
        raise ValueError('expected original 10-cm depth bounds unless zmid_cm is supplied')
    if layers.groupby('profile_id').npp_kg_m2_yr.nunique(dropna=False).ne(1).any():
        raise ValueError('inconsistent site NPP')
    good_npp = np.isfinite(layers.npp_kg_m2_yr) & layers.npp_kg_m2_yr.gt(0)
    valid = pd.DataFrame({
        'invalid coordinates': layers.latitude.between(-90, 90) & layers.longitude.between(-180, 180),
        'incomplete or nonpositive layer stocks': np.isfinite(layers.stock_kg_m2) & layers.stock_kg_m2.gt(0),
        'incomplete radiocarbon after spatial filling': np.isfinite(layers.fm_obs),
        'missing or nonpositive NPP': good_npp,
        'missing or nonpositive layer input': ~good_npp | (np.isfinite(layers.input_kg_m2_yr) & layers.input_kg_m2_yr.gt(0))})
    if sampled:
        valid['interval outside 0–100 cm or invalid thickness'] = (
            layers.z_top_cm.ge(0) & layers.z_bottom_cm.le(100) & layers.z_bottom_cm.gt(layers.z_top_cm))
        valid['zmid outside interval or Shi depth range (0–99 cm)'] = (
            layers.zmid_cm.between(layers.z_top_cm, layers.z_bottom_cm) & layers.zmid_cm.between(0, 99))
    usable = valid.all(axis=1)
    grouped = valid.groupby(layers.profile_id, sort=False).all()
    coverage = layers.groupby('profile_id').size().ge(1) if sampled else layers.groupby('profile_id').size().eq(10)
    complete = grouped.all(axis=1) & coverage
    eligible = usable.groupby(layers.profile_id, sort=False).any() if allow_partial else complete
    keep = layers.profile_id.map(eligible) & usable
    profiles = layers[keep].copy()
    excluded_profiles = grouped.loc[~eligible].copy()
    excluded_profiles['incomplete layer coverage'] = coverage.reindex(excluded_profiles.index)
    excluded = _exclusions(excluded_profiles.index, np.nan, excluded_profiles)
    missing = layers.profile_id.map(eligible) & ~usable
    layer_checks = valid[missing].rename(columns={'incomplete or nonpositive layer stocks': 'missing or nonpositive layer stock'})
    excluded = pd.concat([excluded, _exclusions(layers.loc[missing, 'profile_id'], layers.loc[missing, 'layer'], layer_checks)], ignore_index=True)
    metadata = {'layer_source': 'Layers sheet' if sampled else 'Profiles sheet, 10-cm increments',
                'input_profile_count': len(eligible), 'eligible_profile_count': int(eligible.sum()),
                'allow_partial': allow_partial, 'eligible_layer_count': len(profiles),
                'complete_profile_count': int(complete.sum()), 'partial_profile_count': int((eligible & ~complete).sum()),
                'radiocarbon_spatially_filled_raw_layer_count': int(layers.radiocarbon_spatially_filled.sum()),
                'radiocarbon_spatially_filled_layer_count': int(profiles.radiocarbon_spatially_filled.sum())}
    return PreparedProfiles(profiles, excluded, metadata)


def _exclusions(ids, layers, validity: pd.DataFrame) -> pd.DataFrame:
    reasons = ['; '.join(name for name, valid in row.items() if not valid) for row in validity.to_dict('records')]
    return pd.DataFrame({'profile_id': np.asarray(ids), 'layer': np.asarray(layers), 'reason': reasons})


def save_depth_turnover(processed_path, shi_path, npp_path, output_path,
                        distribution=InputAllocation()) -> None:
    """Preparation only: write every layer plus source fingerprints; no model fitting."""
    output = Path(output_path)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'use a new output path: {output}')
    checksum = file_digest(shi_path, 'md5')
    if checksum != SHI_MD5:
        raise ValueError('Shi file does not match the published delta-14C checksum at https://zenodo.org/records/3823612')
    layers = pd.read_csv(processed_path, dtype={'profile_id': str}, float_precision='round_trip')
    with xr.open_dataset(shi_path) as shi:
        result = prepare_turnover(layers, shi, pd.read_csv(npp_path), distribution)
    metadata = {'sources': {name: {'path': str(Path(path).resolve()), 'sha256': file_digest(path)}
                           for name, path in [('processed_layers', processed_path), ('shi', shi_path), ('npp', npp_path)]},
                'shi_published_md5_verified': checksum, 'shi_metadata_units_override': 'published delta-14C in per mil; file says year',
                'radiocarbon_spatial_filling': 'rioxarray nearest at each one-cm depth before site selection; matches original 02_get_turnover_14C.py',
                'radiocarbon_depth_aggregation': ('value at reported zmid; linear interpolation between adjacent one-cm levels'
                    if 'zmid_cm' in result else 'ordinary mean of ten one-cm values after spatial filling; no depth interpolation'),
                'radiocarbon_reference_year': 2000, 'reference_year_basis': 'inherited model convention; absent from NetCDF metadata',
                'npp_conversion': 'cached g C/m²/yr divided by 1000 to kg C/m²/yr',
                'npp_coordinate_matching_decimals': NPP_COORD_DECIMALS, 'npp_imputed': False,
                'fnew_observation': ('Layers sheet ratio_newCtoC; stocks from Cstock in kg C/m²'
                    if 'zmid_cm' in result else 'difference of cumulative Cnew divided by layer Ctotal')}
    output.parent.mkdir(parents=True, exist_ok=True)
    preprocessing_metadata = Path(processed_path).with_suffix('.json')
    if preprocessing_metadata.exists():
        metadata['preprocessing'] = json.loads(preprocessing_metadata.read_text())
    result.to_csv(output, index=False)
    metadata['table_sha256'] = file_digest(output)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')


def load_profiles(path: str | Path = 'results/all_sites_14C_turnover_depth.csv', *,
                  allow_partial: bool = False) -> PreparedProfiles:
    """Read the prepared CSV; fitting never reloads Excel or the global Shi raster."""
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text()) if path.with_suffix('.json').exists() else {}
    digest = file_digest(path)
    if metadata.get('table_sha256', digest) != digest:
        raise ValueError('prepared table changed without updating its metadata')
    result = prepare_profiles(pd.read_csv(path, dtype={'profile_id': str}, float_precision='round_trip'), allow_partial=allow_partial)
    result.metadata = {**metadata, **result.metadata, 'prepared_table': {'path': str(path.resolve()), 'sha256': digest}}
    return result
