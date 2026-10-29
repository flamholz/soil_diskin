"""Step 02: soil radiocarbon, carbon inputs and turnover times for the Balesdent sites.

For each site, this script combines
  * the soil carbon stock, from script 01;
  * the carbon input, taken to be the site's NPP from MODIS (MOD17A3HGF, the mean
    of all annual images, averaged over 10 km around the site), fetched from
    Earth Engine on every run with the project named in config.yaml; and
  * the soil radiocarbon, from the global Delta14C map of Shi et al. (2020)
    (data/shi_2020/global_delta_14C.nc), which has one-cm levels from 0 to 100 cm.
    Gaps in the map are filled with the nearest value at the same depth.

The turnover time is stock / input. Radiocarbon is reported as Delta14C in per mil
(column `14C`) and as R = Delta14C/1000 + 1 (column `fm_obs`), the convention of
data/14C_atm_annot.csv and of the calibration scripts.

There are two modes:
  * bulk (default): one row per row of results/processed_balesdent_2018.csv,
    for the 0-100 cm soil column. Delta14C is averaged over 10-cm bins and weighted
    by the row's share of carbon in each bin (weight_0 ... weight_90 from script
    01). The turnover time uses the whole NPP and the 0-100 cm stock, with its
    q05/q95 estimates. Writes results/all_sites_14C_turnover.csv.
  * depth-resolved (--depth-resolved): one row per reported layer of
    results/processed_balesdent_2018_sampled.csv. Delta14C is averaged over the
    layer's depth range, and the layer's input is its share of NPP under an
    InputAllocation of inputs with depth. Writes
    results/all_sites_14C_turnover_sampled.csv and a JSON file recording the
    sources and settings, and refuses to overwrite either.

Run from the project root directory:
    python notebooks/02_get_turnover_14C.py [--depth-resolved] [-i INPUT] [-o OUTPUT]
"""
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
    """How the carbon input (NPP) is distributed over the top 100 cm of soil.

    A share `surface_fraction` of the input enters uniformly in 0-10 cm, and the
    rest is distributed exponentially with depth, with e-folding depth
    `input_depth_cm`, normalized over 0-100 cm. Only `soil_npp_fraction` of NPP
    enters the soil at all. Used by the depth-resolved mode only.

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
    """Add radiocarbon, NPP and turnover times to the processed Balesdent table.

    NPP and the radiocarbon profile are looked up once per location (rows of
    `data` can share a location, e.g. with different labeling durations), and
    every row then gets its location's values.

    Args:
        data: processed Balesdent table from script 01, with Latitude and
            Longitude. Bulk mode needs weight_0 ... weight_90 and
            Ctotal_0-100estim (and its _q05/_q95); depth-resolved mode needs
            z_top_cm, z_bottom_cm and stock_kg_m2 (and its _q05/_q95, if present).
        shi: the Shi et al. (2020) Delta14C dataset (see sample_radiocarbon).
        distribution: InputAllocation for depth-resolved mode, or None for bulk.

    Returns:
        Bulk: one row per row of `data` with Latitude, Longitude, 14C, NPP
        (g C/m²/yr), the stocks, fm_obs and turnover/_q05/_q95 (years).
        Depth-resolved: `data` plus 14C, fm_obs, radiocarbon_spatially_filled,
        npp_kg_m2_yr, the allocation settings, input_kg_m2_yr and
        turnover/_q05/_q95.
    """
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
        # Weight the 10-cm bin means of each row's location by the row's own carbon
        # weights: rows at one location share the radiocarbon profile, but their
        # weights can differ. Bins below a profile's deepest measured layer have no
        # weight and are skipped (script 01 normalizes the weights over the rest).
        bins = delta.reshape(-1, 10, 10).mean(axis=2)
        weights = result.filter(regex=r'^weight_\d+').to_numpy()
        result['14C'] = np.nansum(bins[rows]*weights, axis=1)
        result['fm_obs'] = result['14C']/1000+1
        suffixes = ['', '_q05', '_q95']
        stocks = ['Ctotal_0-100estim'+suffix for suffix in suffixes]
        for stock, suffix in zip(stocks, suffixes):
            result['turnover'+suffix] = result[stock]/result.npp_kg_m2_yr
        columns = coords+['14C', 'NPP', *stocks, 'fm_obs']+['turnover'+suffix for suffix in suffixes]
        return result[columns].rename(columns={v: k for k, v in names.items()})

    top, bottom = result[['z_top_cm', 'z_bottom_cm']].to_numpy(float).T
    depth = np.arange(100)  # Level z represents [z, z+1) cm, as in the bulk means.
    overlap = np.maximum(0, np.minimum(bottom[:, None], depth+1)-np.maximum(top[:, None], depth))
    supported = (top >= 0) & (bottom <= 100) & (bottom > top)
    thickness = np.where(supported, bottom-top, np.nan)
    # Missing contributing depths invalidate the mean; missing depths elsewhere do not.
    result['14C'] = (np.where(overlap > 0, delta[rows], 0)*overlap).sum(axis=1)/thickness
    result['fm_obs'] = result['14C']/1000+1
    result['radiocarbon_spatially_filled'] = (filled[rows] & (overlap > 0)).any(axis=1) & supported
    result = result.assign(**asdict(distribution), npp_distribution='InputAllocation')
    result['input_kg_m2_yr'] = distribution.layer_input(result.npp_kg_m2_yr, result.z_top_cm, result.z_bottom_cm)
    for suffix in ['', '_q05', '_q95']:
        if 'stock_kg_m2'+suffix in result:
            result['turnover'+suffix] = result['stock_kg_m2'+suffix]/result.input_kg_m2_yr.where(result.input_kg_m2_yr.gt(0))
    return result.drop(columns='NPP')


def fetch_npp(sites):
    """NPP at each site in g C/m²/yr, from MODIS via Earth Engine.

    Takes the mean of all annual images of MODIS/061/MOD17A3HGF (band Npp, scaled
    from 0.0001 kg C/m²/yr to g C/m²/yr) and averages it over 10 km around each
    site. Authenticates with Earth Engine and uses the project named in
    config.yaml (read from the working directory).

    Args:
        sites: table with Latitude and Longitude columns.

    Returns:
        Array of NPP, in the order of `sites`.
    """
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
    """Read the processed sites, add radiocarbon, NPP and turnover, and write the table.

    In depth-resolved mode, also checks the Shi file against its published
    checksum and writes a JSON file of sources and settings beside the table.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--depth-resolved', action='store_true',
                        help='one row per reported layer instead of one per site (bulk 0-100 cm)')
    parser.add_argument('-i', '--input',
                        help='processed sites from script 01 (default: results/processed_balesdent_2018[_sampled].csv)')
    parser.add_argument('-o', '--output',
                        help='output table (default: results/all_sites_14C_turnover[_sampled].csv)')
    parser.add_argument('--shi', default='data/shi_2020/global_delta_14C.nc', help='Shi et al. (2020) Delta14C map')
    parser.add_argument('--input-depth', type=float, default=DEFAULT_ALLOCATION.input_depth_cm,
                        help='depth-resolved: e-folding depth of the input profile (cm)')
    parser.add_argument('--surface-fraction', type=float, default=DEFAULT_ALLOCATION.surface_fraction,
                        help='depth-resolved: share of the input entering in 0-10 cm')
    parser.add_argument('--soil-npp-fraction', type=float, default=1.,
                        help='depth-resolved: share of NPP entering the soil')
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
