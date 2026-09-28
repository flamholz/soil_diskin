"""WCS units/nodata and the bulk stock paths used by preprocessing."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from soil_diskin import soilgrids_utils_w_unc as soilgrids


@pytest.mark.parametrize('pixels,expected', [([10., 20., -9999., 30.], 2.),
                                           ([-9999.]*4, None), ([np.nan]*4, None)])
def test_wcs_request_reads_scaled_values_and_nodata(monkeypatch, pixels, expected):
    with MemoryFile() as raster:
        with raster.open(driver='GTiff', width=2, height=2, count=1, dtype='float32',
                         nodata=-9999., transform=from_origin(0, 2, 1, 1)) as dataset:
            dataset.write(np.array(pixels, dtype='float32').reshape(1, 2, 2))
        content = raster.read()
    calls = []

    def get(url, *, params, timeout):
        calls.append(params)
        assert url == soilgrids._load_wcs_config()['base_url'] and timeout == 60
        return SimpleNamespace(content=content, raise_for_status=lambda: None)

    monkeypatch.setattr(soilgrids.requests, 'get', get)
    result = soilgrids.get_stats_at_point(40., -75., buffer_m=10., depths=['0-5cm'],
                                         stats=['mean', 'q05', 'unknown'], stat_type='soc')
    assert result == {'0-5cm': {'mean': expected, 'q05': expected}}
    assert [call['COVERAGEID'] for call in calls] == ['soc_0-5cm_mean', 'soc_0-5cm_Q0.05']
    assert calls[0]['map'] == '/map/soc.map' and calls[0]['VERSION'] == '2.0.1'
    for subset in calls[0]['SUBSET']:
        lower, upper = map(float, subset.split('(')[1].rstrip(')').split(','))
        assert upper-lower == pytest.approx(20.)


@pytest.mark.parametrize('use_bulk_density', [True, False])
@pytest.mark.parametrize('uncertainty', [True, False])
def test_bulk_backfill_preserves_units_observations_and_failure_counts(monkeypatch, use_bulk_density, uncertainty):
    calls = []
    def fetch(lat, lon, buffer_m=None, *, depths, stats=None, stat_type):
        calls.append((lat, stat_type))
        if lat == 3.:
            raise requests.ConnectionError('Service unavailable')
        if stat_type == 'ocs':
            return {'0-30cm': {'mean': 100., 'q05': 20., 'q95': 180.}}
        depths = soilgrids._load_wcs_config()['depths'] if depths is None else depths
        values = [10., 20., 30., 40., 50., 60.] if stat_type == 'soc' else [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        return {depth: {'mean': None if lat == 4. and depth == '15-30cm' else value}
                for depth, value in zip(depths, values)}

    monkeypatch.setattr(soilgrids, 'get_stats_at_point', fetch)
    data = pd.DataFrame({'Latitude': [1., 1., 2., 3., 4.], 'Longitude': 0.,
                         'Ctotal_0-100estim': [np.nan, np.nan, 9., np.nan, np.nan],
                         'C_data_source': 'measured'})
    result, info = soilgrids.backfill_missing_soc(data, use_bulk_density=use_bulk_density,
                                                calc_uncertainty=uncertainty)
    mean = 55.6 if use_bulk_density else 50.7
    np.testing.assert_allclose(result['Ctotal_0-100estim'], [mean, mean, 9.])
    assert result.C_data_source.tolist() == ['SoilGrids backfill']*2+['measured']
    assert calls.count((1., 'soc')) == 1 and not any(lat == 2. for lat, _ in calls)
    assert info == {'n_missing': 4, 'n_filled': 1, 'n_failed': 2, 'fill_rate': 1/3}
    if use_bulk_density and uncertainty:
        # Original bulk method scales the whole stock by 0–30 cm OCS ratios.
        np.testing.assert_allclose(result['Ctotal_0-100estim_q05'], [11.12, 11.12, np.nan])
        np.testing.assert_allclose(result['Ctotal_0-100estim_q95'], [100.08, 100.08, np.nan])
        assert calls.count((1., 'ocs')) == 1
    else:
        assert 'Ctotal_0-100estim_q05' not in result
    _, unchanged = soilgrids.backfill_missing_soc(result)
    assert unchanged == {'n_missing': 0, 'n_filled': 0, 'n_failed': 0}


def test_bulk_and_layers_share_cached_means_and_relative_uncertainty(tmp_path, monkeypatch):
    calls = []

    def fetch(lat, lon, *, depths, stats, stat_type):
        calls.append((stat_type, stats[0]))
        if stat_type == 'ocs':
            return {depth: {s: {'mean': 100., 'q05': 20., 'q95': 180.}[s] for s in stats} for depth in depths}
        mean = [10., 20., 30., 40., 50.] if stat_type == 'soc' else [1.1, 1.2, 1.3, 1.4, 1.5]
        factors = {'mean': 1., 'q05': .5 if stat_type == 'soc' else .8,
                   'q95': 1.5 if stat_type == 'soc' else 1.2}
        return {depth: {s: value*factors[s] for s in stats} for depth, value in zip(depths, mean)}

    monkeypatch.setattr(soilgrids, 'get_stats_at_point', fetch)
    cache = tmp_path/'shared.json'
    bulk, _ = soilgrids.backfill_missing_soc(pd.DataFrame({'Latitude': [1.], 'Longitude': [2.],
                                                        'Ctotal_0-100estim': [np.nan]}), cache)
    bulk_calls = len(calls)
    layers, _ = soilgrids.backfill_missing_soc(pd.DataFrame({'latitude': [1., 1.], 'longitude': [2., 2.],
        'z_top_cm': [0., 20.], 'z_bottom_cm': [20., 100.], 'stock_kg_m2': np.nan}), cache)
    assert len(calls) == bulk_calls  # All means and OCS quantiles are already cached.
    np.testing.assert_allclose(layers.stock_kg_m2, [4.9, 50.7])
    assert layers.stock_kg_m2.sum() == pytest.approx(bulk['Ctotal_0-100estim'].iloc[0])
    assert layers.stock_kg_m2_q05.sum() == pytest.approx(11.12)
    assert bulk['Ctotal_0-100estim_q05'].iloc[0] == pytest.approx(11.12)
    assert layers.stock_kg_m2_q95.sum() == pytest.approx(bulk['Ctotal_0-100estim_q95'].iloc[0])
