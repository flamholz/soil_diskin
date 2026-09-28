"""Prepare bulk or sampled-layer radiocarbon, NPP inputs, and stock/input turnover."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr

from soil_diskin.utils import file_digest
import ee
import geemap
import ssl
import yaml

SHI_MD5 = '645aa8d54cbc36cb329c29bcffc4352b'  # Published delta-14C file; metadata says 'year'.

@dataclass(frozen=True)
class InputAllocation:
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
        """Integrate NPP over a layer in cm; return kg C/m²/year (NaN for invalid rows).

        Accept scalars or arrays. Normalize over 0–100 cm, including missing layers.
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


def sample_radiocarbon(shi, sites):
    """Sample each one-cm depth after the original nearest-neighbor spatial filling."""
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
    lat, lon = sites[['latitude', 'longitude']].to_numpy(float).T
    valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
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


def prepare_turnover(data, shi, distribution=DEFAULT_ALLOCATION):
    """Fetch site NPP and prepare layer turnover; distribution=None selects bulk."""
    coords = ['latitude', 'longitude']
    names = {'Latitude': 'latitude', 'Longitude': 'longitude'}
    data = data.rename(columns=names)
    sites = data.drop_duplicates(coords).reset_index(drop=True)
    valid = sites.latitude.between(-90, 90) & sites.longitude.between(-180, 180)
    sites['NPP'] = np.nan
    sites.loc[valid, 'NPP'] = fetch_npp(sites.loc[valid, coords].rename(columns={v: k for k, v in names.items()}))
    delta, filled = sample_radiocarbon(shi, sites)
    result = data.merge(sites[coords+['NPP']].assign(site_index=sites.index), on=coords, how='left', validate='many_to_one')
    rows = result.pop('site_index').to_numpy(int)
    result['npp_kg_m2_yr'] = pd.to_numeric(result.NPP, errors='coerce')/1000
    if distribution is None:
        weights = sites.filter(regex=r'^weight_\d+').to_numpy()
        result['14C'] = np.nansum(delta.reshape(-1, 10, 10).mean(axis=2)*weights, axis=1)[rows]
        result['fm'] = result['14C']/1000+1
        suffixes = ['', '_q05', '_q95']
        stocks = ['Ctotal_0-100estim'+suffix for suffix in suffixes]
        for stock, suffix in zip(stocks, suffixes):
            result['turnover'+suffix] = result[stock]/result.npp_kg_m2_yr
        columns = coords+['14C', 'NPP', *stocks, 'fm']+['turnover'+suffix for suffix in suffixes]
        return result[columns].rename(columns={v: k for k, v in names.items()})

    top, bottom = result[['z_top_cm', 'z_bottom_cm']].to_numpy(float).T
    depth = np.arange(100)  # Level z represents [z, z+1) cm, as in the bulk means.
    overlap = np.maximum(0, np.minimum(bottom[:, None], depth+1)-np.maximum(top[:, None], depth))
    supported = (top >= 0) & (bottom <= 100) & (bottom > top)
    thickness = np.where(supported, bottom-top, np.nan)
    # Missing contributing depths invalidate the mean; missing depths elsewhere do not.
    result['14C'] = (np.where(overlap > 0, delta[rows], 0)*overlap).sum(axis=1)/thickness
    result['fm'] = result['14C']/1000+1
    result['radiocarbon_spatially_filled'] = (filled[rows] & (overlap > 0)).any(axis=1) & supported
    result = result.assign(**asdict(distribution), npp_distribution='InputAllocation')
    result['input_kg_m2_yr'] = distribution.layer_input(result.npp_kg_m2_yr, result.z_top_cm, result.z_bottom_cm)
    for suffix in ['', '_q05', '_q95']:
        if 'stock_kg_m2'+suffix in result:
            result['turnover'+suffix] = result['stock_kg_m2'+suffix]/result.input_kg_m2_yr.where(result.input_kg_m2_yr.gt(0))
    return result.drop(columns='NPP')


def fetch_npp(sites):
    """Original MODIS mean at 10 km resolution, returned in g C/m²/year."""
    config = yaml.safe_load(Path('config.yaml').read_text())
    ee.Authenticate()
    ee.Initialize(project=config['earth_engine']['project'])
    ssl._create_default_https_context = ssl._create_unverified_context
    points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([row.Longitude, row.Latitude]))
                                  for row in sites.itertuples()])
    npp = ee.ImageCollection('MODIS/061/MOD17A3HGF').select('Npp').mean().multiply(0.0001*1e3)
    result = geemap.ee_to_df(npp.reduceRegions(collection=points, reducer=ee.Reducer.mean(), scale=10_000))
    result.columns = ['NPP']
    return result.NPP.to_numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth-resolved', action='store_true')
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('--shi', default='data/shi_2020/global_delta_14C.nc')
    parser.add_argument('--input-depth', type=float, default=DEFAULT_ALLOCATION.input_depth_cm)
    parser.add_argument('--surface-fraction', type=float, default=DEFAULT_ALLOCATION.surface_fraction)
    parser.add_argument('--soil-npp-fraction', type=float, default=1.)
    args = parser.parse_args(argv)
    suffix = '_sampled' if args.depth_resolved else ''
    source = Path(args.input or f'results/processed_balesdent_2018{suffix}.csv')
    output = Path(args.output or f'results/all_sites_14C_turnover{suffix}.csv')
    if args.depth_resolved:
        if output.exists() or output.with_suffix('.json').exists():
            raise FileExistsError(f'use a new output path: {output}')
        checksum = file_digest(args.shi, 'md5')
        if checksum != SHI_MD5:
            raise ValueError('Shi file does not match the published delta-14C checksum at https://zenodo.org/records/3823612')
    data = pd.read_csv(source, dtype={'profile_id': str}, float_precision='round_trip' if args.depth_resolved else None)
    distribution = InputAllocation(args.input_depth, args.surface_fraction, args.soil_npp_fraction) if args.depth_resolved else None
    with xr.open_dataset(args.shi) as shi:
        result = prepare_turnover(data, shi, distribution=distribution)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    if args.depth_resolved:
        metadata = {'sources': {name: {'path': str(Path(path).resolve()), 'sha256': file_digest(path)}
                               for name, path in [('processed_layers', source), ('shi', args.shi)]},
                    'shi_published_md5_verified': checksum, 'shi_metadata_units_override': 'published delta-14C in per mil; file says year',
                    'radiocarbon_spatial_filling': 'rioxarray nearest at each one-cm depth before site selection; matches original 02_get_turnover_14C.py',
                    'radiocarbon_depth_aggregation': 'thickness-weighted mean over reported bounds; one-cm bins [level, level+1); no depth extrapolation',
                    'radiocarbon_reference_year': 2000, 'reference_year_basis': 'inherited model convention; absent from NetCDF metadata',
                    'npp_source': 'Earth Engine MODIS/061/MOD17A3HGF Npp; temporal mean and spatial mean at 10 km',
                    'npp_retrieved_utc': datetime.now(timezone.utc).isoformat(),
                    'npp_conversion': 'g C/m²/yr divided by 1000 to kg C/m²/yr', 'npp_imputed': False,
                    'input_allocation': asdict(distribution),
                    'fnew_observation': 'Layers sheet ratio_newCtoC; stocks from Cstock in kg C/m²',
                    'preparation_source_sha256': file_digest(__file__), 'table_sha256': file_digest(output)}
        if source.with_suffix('.json').exists():
            metadata['preprocessing'] = json.loads(source.with_suffix('.json').read_text())
        output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved turnover to {output}')


if __name__ == '__main__':
    main()
