"""Depth overlap, units, source preservation, and cached network reads."""
import json
import numpy as np
import pandas as pd
import pytest
import requests

from soil_diskin import soilgrids_utils_w_unc as soilgrids


def test_layer_backfill_preserves_observations_and_integrates_depth_bands(tmp_path, monkeypatch):
    calls = []

    def fetch(lat, lon, *, depths, stats, stat_type):
        calls.append((lat, lon, stat_type))
        assert stats == ['mean']
        values = [10., 20., 30., 40., 50.] if stat_type == 'soc' else [1.1, 1.2, 1.3, 1.4, 1.5]
        return {depth: {'mean': value} for depth, value in zip(depths, values)}

    monkeypatch.setattr(soilgrids, 'get_stats_at_point', fetch)
    layers = pd.DataFrame({'latitude': [1.]*6, 'longitude': [2.]*6,
        'z_top_cm': [2., 20., 2., 95., 0., 98.], 'z_bottom_cm': [20., 30., 20., 110., 0., 100.],
        'zmid_cm': [10., 25., 10., 100., 0., 99.5],
        'stock_kg_m2': [np.nan, np.nan, 123., np.nan, 0., np.nan],
        'fnew_obs': [.1, .2, .3, .4, .5, .6]})
    cache = tmp_path/'cache.json'
    filled, info = soilgrids.backfill_missing_soc(layers, cache, calc_uncertainty=False)
    # 2–20 cm overlaps 3 cm, 10 cm and 5 cm of three SoilGrids bands.
    # !!! KNOWN FAILURE !!! TODO: resolve, then delete this comment.
    # The old backfill_layer_stocks (commit cbaf876) skipped layers whose zmid_cm was
    # outside 0–99 cm, so the last row (98–100 cm, zmid 99.5) stayed NaN. Commit 4996c4e
    # merged it into backfill_missing_soc, which dropped that filter, so the row is now
    # filled (1.5). Either restore the zmid filter in the layered path of
    # backfill_missing_soc, or, if the change was intended, expect 1.5 here and
    # update the 'missing' entry in the stock_source assertion below.
    np.testing.assert_allclose(filled.stock_kg_m2, [4.68, 3.9, 123., np.nan, 0., np.nan])
    np.testing.assert_array_equal(filled.stock_kg_m2_reported, layers.stock_kg_m2)
    np.testing.assert_array_equal(filled.fnew_obs, layers.fnew_obs)
    assert filled.stock_source.tolist() == ['SoilGrids backfill']*2 + ['Balesdent Layers', 'missing', 'Balesdent Layers', 'missing']
    assert info['requested_layers'] == info['filled_layers'] == 2
    assert info['unfilled_layers'] == 0 and info['uncertainty_propagated'] is False
    assert len(calls) == 2  # One SOC and one density request per coordinate, not per layer.
    assert json.loads(cache.read_text())['points']['1.0000000000,2.0000000000']['soc']['0-5cm']['mean'] == 10.
    again, _ = soilgrids.backfill_missing_soc(layers, cache, calc_uncertainty=False)
    pd.testing.assert_frame_equal(filled, again)
    assert len(calls) == 2
    again, _ = soilgrids.backfill_missing_soc(filled, cache, calc_uncertainty=False)
    pd.testing.assert_frame_equal(filled, again)
    changed = json.loads(cache.read_text())
    changed['wcs_config']['soc_scale'] = 1.
    cache.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='cache settings differ'):
        soilgrids.backfill_missing_soc(layers, cache, calc_uncertainty=False)


def test_missing_depth_or_failed_request_never_fabricates_a_stock(tmp_path, monkeypatch):
    def fetch(lat, lon, *, depths, stats, stat_type):
        if lat == 3.:
            raise requests.ConnectionError('Unavailable service')
        return {depth: {'mean': None if depth == '5-15cm' else 10. if stat_type == 'soc' else 1.}
                for depth in depths}

    monkeypatch.setattr(soilgrids, 'get_stats_at_point', fetch)
    layers = pd.DataFrame({'latitude': [1., 1., 3.], 'longitude': [2.]*3,
        'z_top_cm': [0., 0., 0.], 'z_bottom_cm': [5., 10., 5.], 'stock_kg_m2': np.nan})
    filled, info = soilgrids.backfill_missing_soc(layers, tmp_path/'cache.json', calc_uncertainty=False)
    np.testing.assert_allclose(filled.stock_kg_m2, [.5, np.nan, np.nan])
    assert info['filled_layers'] == 1 and info['unfilled_layers'] == 2
    assert 'required depth band' in filled.stock_fill_error.iloc[1]
    assert 'Unavailable service' in filled.stock_fill_error.iloc[2]
