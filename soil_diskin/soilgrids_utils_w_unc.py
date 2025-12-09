"""
SoilGrids utilities using the ISRIC WCS endpoint.

This module replaces the previous Google Earth Engine implementation with direct
WCS requests so we can retrieve means and quantiles for buffered point queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml
# from pyproj import CRS, Transformer 
import geopandas as gpd
from rasterio.io import MemoryFile
import rioxarray
from owslib.wcs import WebCoverageService


_DEFAULT_WCS_CONFIG: Dict[str, object] = {
    "base_url": "https://maps.isric.org/mapserv",
    "map": "/map/soc.map",
    "buffer_m": 125,
    "projection_crs": "ESRI:54052",  # IGH
    "geographic_crs": "EPSG:4326",
    "format": "image/tiff",
    "timeout": 60,
    "subset_axes": ["long", "lat"],
    "soc_prefix": "soc",
    "bdod_prefix": "bdod",
    "depths": [
        "0-5cm",
        "5-15cm",
        "15-30cm",
        "30-60cm",
        "60-100cm",
        "100-200cm",
    ],
    "stock_depths": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
    "thickness_cm": {
        "0-5cm": 5,
        "5-15cm": 10,
        "15-30cm": 15,
        "30-60cm": 30,
        "60-100cm": 40,
    },
    "stats": {"mean": "mean", "q05": "Q0.05", "q95": "Q0.95"},
    "soc_scale": 0.1,  # convert SoilGrids integer dg/kg to g/kg
    "bdod_scale": 0.01,  # convert cg/cm3 to g/cm3
}


_WCS_CONFIG: Optional[Dict[str, object]] = None


def _config_path():
    """Return the absolute path to config.yaml."""
    return "config.yaml"


def _merge_dict(default: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    merged = dict(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _load_wcs_config() -> Dict[str, object]:
    global _WCS_CONFIG
    if _WCS_CONFIG is not None:
        return _WCS_CONFIG

    cfg = dict(_DEFAULT_WCS_CONFIG)
    cfg_path = _config_path()
    try:
        with open(cfg_path, "r") as f:
            file_cfg = yaml.safe_load(f) or {}
        if "wcs" in file_cfg:
            cfg = _merge_dict(cfg, file_cfg["wcs"])
    except FileNotFoundError:
        pass
    _WCS_CONFIG = cfg
    return cfg


def initialize_earth_engine():
    """Placeholder to preserve API compatibility with the previous module."""


def _get_transformers(cfg: Dict[str, object]) -> Tuple[Transformer, Transformer]:
    proj = CRS.from_user_input(cfg["projection_crs"])
    geo = CRS.from_user_input(cfg["geographic_crs"])
    forward = Transformer.from_crs(geo, proj, always_xy=True)
    inverse = Transformer.from_crs(proj, geo, always_xy=True)
    return forward, inverse


def _compute_subset(lat: float, lon: float, buffer_m: Optional[float] = None) -> Tuple[float, float, float, float]:
    cfg = _load_wcs_config()
    half = float(buffer_m or cfg["buffer_m"])
    gdf = gpd.points_from_xy([lon], [lat])
    gdf.crs = cfg['geographic_crs']
    gdf = gdf.to_crs(cfg['projection_crs'])
    # forward, inverse = _get_transformers(cfg)
    # x, y = inverse.transform(lon, lat)
    # corners = [
    #     (float(gdf.x[0]) - half, float(gdf.y[0]) - half),
    #     (float(gdf.x[0]) - half, float(gdf.y[0]) + half),
    #     (float(gdf.x[0]) + half, float(gdf.y[0]) - half),
    #     (float(gdf.x[0]) + half, float(gdf.y[0]) + half),
    # ]
    # lon_vals: List[float] = []
    # lat_vals: List[float] = []
    # for cx, cy in corners:
    #     lon_c, lat_c = inverse.transform(cx, cy)
    #     lon_vals.append(lon_c)
    #     lat_vals.append(lat_c)
    return float(gdf.x[0]) - half, float(gdf.y[0]) - half, float(gdf.x[0]) + half, float(gdf.y[0]) + half


def _build_params(coverage_id: str, subset: Tuple[float, float, float, float], cfg: Dict[str, object], stat_type: str) -> Dict[str, object]:
    lon_min, lat_min, lon_max, lat_max = subset
    lon_axis, lat_axis = cfg["subset_axes"]
    subsets = [
        f"{lon_axis}({lon_min},{lon_max})",
        f"{lat_axis}({lat_min},{lat_max})",
    ]
    params = {
        "SERVICE": "WCS",
        "VERSION": "2.0.1",
        "REQUEST": "GetCoverage",
        "COVERAGEID": coverage_id,
        "FORMAT": cfg["format"],
        "SUBSET": subsets,
    }
    if cfg.get("map"):
        params["map"] = f'/map/{cfg[stat_type+"_prefix"]}.map'#
    return params


def _fetch_coverage_value(coverage_id: str, subset: Tuple[float, float, float, float], stat_type: str, scale: float = 1.0) -> Optional[float]:
    cfg = _load_wcs_config()
    params = _build_params(coverage_id, subset, cfg, stat_type)
    response = requests.get(cfg["base_url"], params=params, timeout=cfg["timeout"])
    response.raise_for_status()

    with MemoryFile(response.content) as memfile:
        with memfile.open() as dataset:
            dataarray = rioxarray.open_rasterio(dataset, masked=True)
            values = dataarray.squeeze().values
    if np.ma.isMaskedArray(values):
        arr = values.filled(np.nan).astype(float)
    else:
        arr = np.asarray(values, dtype=float)
    arr = arr * scale
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return None
    return float(np.nanmean(valid))


def get_stats_at_point(
    lat: float,
    lon: float,
    buffer_m: Optional[float] = None,
    depths: Optional[Iterable[str]] = None,
    stats: Optional[Iterable[str]] = None,
    stat_type: str = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    cfg = _load_wcs_config()
    subset = _compute_subset(lat, lon, buffer_m)
    depth_list = list(depths) if depths is not None else list(cfg["depths"])
    stat_list = list(stats) if stats is not None else list(cfg["stats"].keys())

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for depth in depth_list:
        results[depth] = {}
        for stat_key in stat_list:
            stat_suffix = cfg["stats"].get(stat_key)
            if stat_suffix is None:
                continue
            coverage_id = f"{cfg[stat_type+'_prefix']}_{depth}_{stat_suffix}"
            value = _fetch_coverage_value(
                coverage_id,
                subset,
                stat_type=stat_type,
                scale=float(cfg[stat_type+"_scale"])
            )
            results[depth][stat_key] = value
    return results


def get_soc_at_point(
    lat: float,
    lon: float,
    buffer_m: Optional[float] = None,
    include_quantiles: bool = False,
    depths: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, object]]:
    stat_keys: Optional[List[str]] = None if include_quantiles else ["mean"]
    stats = get_stats_at_point(
        lat,
        lon,
        buffer_m=buffer_m,
        depths=depths,
        stats=stat_keys,
        stat_type='soc'
    )
    if not stats:
        return None
    if include_quantiles:
        return stats
    return {depth: values.get("mean") for depth, values in stats.items()}


# def get_bdod_at_point(
#     lat: float,
#     lon: float,
#     buffer_m: Optional[float] = None,
#     depths: Optional[Iterable[str]] = None,
# ) -> Dict[str, Optional[float]]:
#     cfg = _load_wcs_config()
#     subset = _compute_subset(lat, lon, buffer_m)
#     depth_list = list(depths) if depths is not None else list(cfg["stock_depths"])
#     results: Dict[str, Optional[float]] = {}
#     for depth in depth_list:
#         coverage_id = f"{cfg['bdod_prefix']}_{depth}_{cfg['stats']['mean']}"
#         results[depth] = _fetch_coverage_value(
#             coverage_id,
#             subset,
#             stat_type='bdod',
#             scale=float(cfg["bdod_scale"])
#         )
#     return results


def _depth_mean(value: object) -> Optional[float]:
    if isinstance(value, dict):
        return value.get("mean")
    return value  # type: ignore[return-value]


def calculate_total_soc_0_100(soc_dict: Optional[Dict[str, object]]) -> Optional[float]:
    if soc_dict is None:
        return None
    cfg = _load_wcs_config()
    thickness_map: Dict[str, float] = cfg["thickness_cm"]
    total_soc = 0.0
    for depth in cfg["stock_depths"]:
        soc_val = _depth_mean(soc_dict.get(depth)) if soc_dict else None
        if soc_val is None:
            return None
        thickness_cm = thickness_map.get(depth)
        if thickness_cm is None:
            return None
        # SOC (kg/m^2) = SOC (g/g) * bulk_density (g/cm^3) * thickness (cm) * 10
        layer_soc = (soc_val / 1000.0) * 1.3 * thickness_cm * 10
        total_soc += layer_soc
    return total_soc


def get_soc_with_bulk_density(
    lat: float,
    lon: float,
    buffer_m: Optional[float] = None,
    depths: Optional[Iterable[str]] = None,
) -> Optional[float]:
    cfg = _load_wcs_config()
    depth_list = list(depths) if depths is not None else list(cfg["stock_depths"])
    thickness_map: Dict[str, float] = cfg["thickness_cm"]

    soc_stats = get_stats_at_point(lat, lon, buffer_m=buffer_m, depths=depth_list, stats=["mean"], stat_type='soc')
    bdod_vals = get_stats_at_point(lat, lon, buffer_m=buffer_m, depths=depth_list, stats=["mean"], stat_type='bdod')

    total_soc = 0.0
    for depth in depth_list:
        soc_val = _depth_mean(soc_stats.get(depth)) if soc_stats else None
        bd_val = bdod_vals.get(depth)
        thickness_cm = thickness_map.get(depth)
        if soc_val is None or bd_val is None or thickness_cm is None:
            return None
        layer_soc = (soc_val / 1000.0) * bd_val * thickness_cm * 10
        total_soc += layer_soc
    return total_soc

def get_soc_with_bulk_density_w_uncertainty(
    lat: float,
    lon: float,
    buffer_m: Optional[float] = None,
    depths: Optional[Iterable[str]] = None,
) -> Optional[float]:
    cfg = _load_wcs_config()
    depth_list = list(depths) if depths is not None else list(cfg["stock_depths"])
    thickness_map: Dict[str, float] = cfg["thickness_cm"]

    soc_stats = get_stats_at_point(lat, lon, buffer_m=buffer_m, depths=depth_list, stat_type='soc')
    bdod_vals = get_stats_at_point(lat, lon, buffer_m=buffer_m, depths=depth_list, stat_type='bdod')

    total_soc = pd.Series(0,index=['mean', 'q05', 'q95'])
    for depth in depth_list:
        soc_val = pd.Series(soc_stats.get(depth))# if soc_stats else None
        bd_val = pd.Series(bdod_vals.get(depth))
        thickness_cm = thickness_map.get(depth)
        if soc_val is None or bd_val is None or thickness_cm is None:
            return None
        layer_soc = (soc_val / 1000.0) * bd_val * thickness_cm * 10
        total_soc += layer_soc
    return total_soc


def backfill_missing_soc(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    soc_col: str = "Ctotal_0-100estim",
    source_col: str = "C_data_source",
    use_bulk_density: bool = True,
    calc_uncertainty: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = df.copy()

    missing_mask = df[soc_col].isna()
    n_missing = int(missing_mask.sum())

    if n_missing == 0:
        return df, {"n_missing": 0, "n_filled": 0, "n_failed": 0}

    missing_locs = df[missing_mask][[lat_col, lon_col]].drop_duplicates()

    soc_lookup = {}
    n_filled = 0
    n_failed = 0

    for _, row in missing_locs.iterrows():
        lat_val, lon_val = float(row[lat_col]), float(row[lon_col])
        key = (lat_val, lon_val)
        try:
            if use_bulk_density:
                if calc_uncertainty:
                    soc_value = get_soc_with_bulk_density_w_uncertainty(lat_val, lon_val)
                else:
                    soc_value = get_soc_with_bulk_density(lat_val, lon_val)
            else:
                soc_values = get_soc_at_point(lat_val, lon_val)
                soc_value = calculate_total_soc_0_100(soc_values)
            if soc_value is not None:
                soc_lookup[key] = soc_value
                n_filled += 1
            else:
                n_failed += 1
        except Exception:
            n_failed += 1

    for idx, row in df[missing_mask].iterrows():
        key = (row[lat_col], row[lon_col])
        if key in soc_lookup:
            df.loc[idx, soc_col] = soc_lookup[key] if not calc_uncertainty else soc_lookup[key]['mean']
            df.loc[idx, source_col] = "SoilGrids backfill"
            if calc_uncertainty:
                df.loc[idx, soc_col + '_q05'] = soc_lookup[key]['q05']
                df.loc[idx, soc_col + '_q95'] = soc_lookup[key]['q95']

    stats = {
        "n_missing": n_missing,
        "n_filled": n_filled,
        "n_failed": n_failed,
        "fill_rate": n_filled / len(missing_locs) if len(missing_locs) > 0 else 0,
    }

    df = df[~df[soc_col].isna()]
    return df, stats
