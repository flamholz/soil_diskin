"""Retrieve SoilGrids WCS values and fill bulk or sampled-layer carbon stocks."""
from datetime import datetime, timezone
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from rasterio.errors import RasterioError
from rasterio.io import MemoryFile
import yaml

from .utils import file_digest


_DEFAULT_WCS_CONFIG = {
    "base_url": "https://maps.isric.org/mapserv", "map": "/map/soc.map",
    "buffer_m": 125, "projection_crs": "ESRI:54052", "geographic_crs": "EPSG:4326",
    "format": "image/tiff", "timeout": 60, "subset_axes": ["long", "lat"],
    "soc_prefix": "soc", "bdod_prefix": "bdod",
    "depths": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"],
    "stock_depths": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
    "thickness_cm": {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15, "30-60cm": 30, "60-100cm": 40},
    "stats": {"mean": "mean", "q05": "Q0.05", "q95": "Q0.95"},
    "soc_scale": 0.1,  # dg/kg to g/kg
    "bdod_scale": 0.01,  # cg/cm³ to g/cm³
}
_WCS_CONFIG = None


def _load_wcs_config():
    global _WCS_CONFIG
    if _WCS_CONFIG is None:
        path = Path('config.yaml')
        overrides = (yaml.safe_load(path.read_text()) or {}).get('wcs', {}) if path.exists() else {}
        cfg = dict(_DEFAULT_WCS_CONFIG)
        for key, value in overrides.items():
            cfg[key] = {**cfg[key], **value} if isinstance(value, dict) and isinstance(cfg.get(key), dict) else value
        _WCS_CONFIG = cfg
    return _WCS_CONFIG


def _compute_subset(lat, lon, buffer_m=None):
    """Projected bounding box around a geographic point, in metres."""
    cfg = _load_wcs_config()
    half = float(buffer_m or cfg['buffer_m'])
    point = gpd.points_from_xy([lon], [lat], crs=cfg['geographic_crs']).to_crs(cfg['projection_crs'])
    x, y = float(point.x[0]), float(point.y[0])
    return x-half, y-half, x+half, y+half


def _fetch_coverage_value(coverage_id, subset, stat_type, scale=1.):
    cfg = _load_wcs_config()
    x0, y0, x1, y1 = subset
    x_axis, y_axis = cfg['subset_axes']
    params = {'SERVICE': 'WCS', 'VERSION': '2.0.1', 'REQUEST': 'GetCoverage',
              'COVERAGEID': coverage_id, 'FORMAT': cfg['format'],
              'SUBSET': [f'{x_axis}({x0},{x1})', f'{y_axis}({y0},{y1})']}
    if cfg.get('map'):
        params['map'] = f"/map/{cfg[stat_type+'_prefix']}.map"
    response = requests.get(cfg['base_url'], params=params, timeout=cfg['timeout'])
    response.raise_for_status()
    with MemoryFile(response.content) as raster, raster.open() as dataset:
        values = dataset.read(masked=True).astype(float).filled(np.nan)
    valid = values[~np.isnan(values)] * scale
    return float(valid.mean()) if valid.size else None


def get_stats_at_point(lat, lon, buffer_m=None, depths=None, stats=None, stat_type=None):
    """Return {depth: {statistic: value}}; nodata is None, network errors propagate."""
    cfg = _load_wcs_config()
    subset = _compute_subset(lat, lon, buffer_m)
    depths = list(cfg['depths'] if depths is None else depths)
    stats = list(cfg['stats'] if stats is None else stats)
    result = {}
    for depth in depths:
        result[depth] = {}
        for stat in stats:
            if stat not in cfg['stats']:
                continue
            coverage = f"{cfg[stat_type+'_prefix']}_{depth}_{cfg['stats'][stat]}"
            result[depth][stat] = _fetch_coverage_value(coverage, subset, stat_type, float(cfg[stat_type+'_scale']))
    return result


def backfill_missing_soc(data: pd.DataFrame, cache_path=None, *, use_bulk_density=True,
                         calc_uncertainty=True) -> tuple[pd.DataFrame, dict]:
    """Fill missing bulk or layer stocks, sharing retrieval and depth integration.

    Layer tables supply z_top_cm/z_bottom_cm; bulk tables are single 0–100 cm
    intervals. Both use the site's 0–30 cm OCS q05/mean and q95/mean ratios
    as relative stock uncertainty, assumed constant with depth.
    Reported stocks remain unchanged. Failed layer rows stay for exclusion logs;
    failed bulk rows are dropped, as in the original preprocessing pipeline.
    """
    layered = 'z_top_cm' in data
    names = {} if layered else {'Latitude': 'latitude', 'Longitude': 'longitude', 'C_data_source': 'stock_source',
                                **{'Ctotal_0-100estim'+s: 'stock_kg_m2'+s for s in ['', '_q05', '_q95']}}
    result = data.rename(columns=names).copy()
    requested = result.stock_kg_m2.isna()
    if not layered:
        if not requested.any():
            return data.copy(), {'n_missing': 0, 'n_filled': 0, 'n_failed': 0}
        result = result.assign(z_top_cm=0., z_bottom_cm=100.)
    cfg = _load_wcs_config()
    path = Path(cache_path) if cache_path is not None else None
    cache = json.loads(path.read_text()) if path and path.exists() else {'wcs_config': cfg, 'points': {}}
    if cache['wcs_config'] != cfg:
        raise ValueError('SoilGrids cache settings differ; use a new cache path')
    if 'stock_source' not in result:
        result['stock_source'] = np.where(result.stock_kg_m2.notna(), 'Balesdent Layers' if layered else 'Balesdent et al. 2018', 'missing')
    if 'stock_kg_m2_reported' not in result:
        result['stock_kg_m2_reported'] = result.stock_kg_m2.where(result.stock_source.ne('SoilGrids backfill'))
    qcols = ['stock_kg_m2_q05', 'stock_kg_m2_q95']
    for column in qcols + ['stock_fill_error', 'stock_uncertainty_error']:
        if column not in result:
            result[column] = np.nan if column in qcols else ''
        elif column not in qcols:
            result[column] = result[column].astype(object)
    valid_depth = (result.z_top_cm.ge(0) & result.z_bottom_cm.le(100)
                   & result.z_bottom_cm.gt(result.z_top_cm)
                   & result.latitude.between(-90, 90) & result.longitude.between(-180, 180))
    missing = requested & valid_depth
    quantiles = calc_uncertainty and use_bulk_density
    # Refresh existing bounds too, so older uncertainty methods are replaced.
    needs_bounds = quantiles & result.stock_source.eq('SoilGrids backfill')
    depths, stats = cfg['stock_depths'], ['mean', 'q05', 'q95']
    properties = [(prop, depths, ['mean'])
                  for prop in (['soc', 'bdod'] if use_bulk_density else ['soc'])]
    if quantiles:
        properties.append(('ocs', cfg['depths_ocs'][:1], stats))
    bounds = np.array([depth.removesuffix('cm').split('-') for depth in depths], dtype=float)
    filled = 0
    for (lat, lon), group in result.loc[missing | (needs_bounds & valid_depth)].groupby(['latitude', 'longitude'], sort=False):
        point = cache['points'].setdefault(f'{lat:.10f},{lon:.10f}', {'latitude': lat, 'longitude': lon})
        errors = {}
        for prop, prop_depths, prop_stats in properties:
            values = point.setdefault(prop, {})
            for stat in prop_stats:
                needed = [depth for depth in prop_depths if stat not in values.get(depth, {})]
                if not needed:
                    continue
                print(f'SoilGrids: {lat:.6f}, {lon:.6f}: {prop} {stat}', flush=True)
                try:
                    fetched = get_stats_at_point(lat, lon, depths=needed, stats=[stat], stat_type=prop)
                    for depth in needed:
                        values.setdefault(depth, {}).update(fetched[depth])
                    point['retrieved_utc'] = datetime.now(timezone.utc).isoformat()
                except (requests.RequestException, RasterioError) as error:
                    errors['stock_fill_error' if stat == 'mean' and prop != 'ocs' else 'stock_uncertainty_error'] = str(error)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=2, allow_nan=False)+'\n')
        soc, bdod = [np.array([[point.get(prop, {}).get(depth, {}).get(stat) for stat in stats]
                               for depth in depths], dtype=float) for prop in ['soc', 'bdod']]
        if not use_bulk_density:
            bdod[:] = 1.3
        for index, row in group.iterrows():
            overlap = np.maximum(0, np.minimum(row.z_bottom_cm, bounds[:, 1])
                                 - np.maximum(row.z_top_cm, bounds[:, 0]))
            used = overlap > 0
            valid = (np.isfinite(soc[used]*bdod[used]) & (soc[used] >= 0) & (bdod[used] > 0)).all(axis=0)
            valid &= np.isclose(overlap.sum(), row.z_bottom_cm-row.z_top_cm)
            stock = overlap[used] @ (0.01*soc[used]*bdod[used])
            stock[~valid] = np.nan
            if quantiles:
                ocs = np.array([point['ocs'].get(cfg['depths_ocs'][0], {}).get(s) for s in stats], dtype=float)
                mean_stock = stock[0] if missing.loc[index] else row.stock_kg_m2
                stock[1:] = mean_stock*(ocs[1:]/ocs[0]) if np.isfinite(ocs[0]) and ocs[0] > 0 else np.nan
                ordered = 0 <= ocs[1] <= ocs[2]
            if missing.loc[index]:
                if np.isfinite(stock[0]) and (stock[0] > 0 if layered else stock[0] >= 0):
                    result.loc[index, ['stock_kg_m2', 'stock_source', 'stock_fill_error']] = [stock[0], 'SoilGrids backfill', '']
                    filled += 1
                else:
                    result.loc[index, 'stock_fill_error'] = errors.get('stock_fill_error', 'Missing/nonpositive SoilGrids stock in a required depth band')
            if quantiles and result.loc[index, 'stock_source'] == 'SoilGrids backfill':
                available = np.isfinite(stock[1:]).all() and ordered
                result.loc[index, qcols] = stock[1:] if available else np.nan
                result.loc[index, 'stock_uncertainty_error'] = '' if available else errors.get(
                    'stock_uncertainty_error', 'Missing/invalid SoilGrids OCS quantiles')
    if not layered:
        locations = result.loc[requested, ['latitude', 'longitude']]
        n_locations = len(locations.drop_duplicates())
        n_filled = len(locations.loc[result.loc[requested, 'stock_kg_m2'].notna()].drop_duplicates())
        columns = list(dict.fromkeys([*data.columns, 'C_data_source'] + (['Ctotal_0-100estim_q05', 'Ctotal_0-100estim_q95'] if quantiles else [])))
        result = result.loc[result.stock_kg_m2.notna()].rename(columns={v: k for k, v in names.items()})
        return result[columns], {'n_missing': int(requested.sum()), 'n_filled': n_filled,
                                 'n_failed': n_locations-n_filled, 'fill_rate': n_filled/n_locations}
    n_bounds = int((result.stock_source.eq('SoilGrids backfill') & result[qcols].notna().all(axis=1)).sum())
    metadata = {'requested_layers': int(missing.sum()), 'filled_layers': filled,
                'unfilled_layers': int(missing.sum())-filled,
                'method': 'sum(overlap_cm * SOC_g_kg * bulk_density_g_cm3 * 0.01); no coarse-fragment correction',
                'uncertainty_method': 'Mean stock scaled by site 0–30 cm OCS q05/mean and q95/mean; relative uncertainty assumed constant with depth',
                'uncertainty_layers': n_bounds, 'uncertainty_propagated': bool(n_bounds),
                'cache_path': str(path.resolve()) if path else None,
                'cache_sha256': file_digest(path) if path and path.exists() else None}
    return result, metadata
