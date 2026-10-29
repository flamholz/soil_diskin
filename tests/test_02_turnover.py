"""Tests for the step 02 scripts: bulk (02_get_turnover_14C.py) and depth-resolved
(02c_get_layer_turnover_14C.py) turnover.

Uses a small synthetic Shi map whose Delta14C depends only on depth, and an NPP
table in place of the one script 02a fetches from Earth Engine.
"""
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from soil_diskin.site_inputs import DEFAULT_ALLOCATION

step02 = import_module('notebooks.02_get_turnover_14C')
step02c = import_module('notebooks.02c_get_layer_turnover_14C')

WEIGHTS = [f'weight_{d}' for d in range(0, 100, 10)]


def synthetic_shi():
    """2x2 map with Delta14C = -10 * level (per mil) at every cell."""
    level = np.arange(100)
    temp = np.broadcast_to(-10.*level[:, None, None], (100, 2, 2)).copy()
    return xr.Dataset({'temp': (('level', 'lat', 'lon'), temp)},
                      coords={'level': level, 'lat': [-.25, .25], 'lon': [-.25, .25]})


def expected_delta(weights):
    """Carbon-weighted mean of the 10-cm bin means, skipping bins without a weight."""
    bin_means = -10.*(np.arange(0, 100, 10) + 4.5)
    return np.nansum(bin_means*np.asarray(weights, float))


def site_row(lat, lon, weights, stock=10.):
    row = dict(Latitude=lat, Longitude=lon, **dict(zip(WEIGHTS, weights)))
    row.update({'Ctotal_0-100estim': stock, 'Ctotal_0-100estim_q05': stock/2, 'Ctotal_0-100estim_q95': stock*2})
    return row


def npp_table(data, npp=500.):
    """An NPP table, as read by site_inputs.read_npp, giving every location of `data` the same NPP."""
    coords = data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})[['latitude', 'longitude']]
    return coords.drop_duplicates().assign(NPP=npp)


def run_bulk(data, npp=500.):
    return step02.bulk_turnover(data, synthetic_shi(), npp_table(data, npp))


def test_rows_at_one_location_use_their_own_weights():
    """Rows sharing a location must each weight Delta14C by their own carbon profile.

    Regression test for a bug inherited from the original script 02 (fixed in
    f6b2459, before the bulk and depth-resolved code were split). NPP and the
    radiocarbon profile were looked up once per location, from
    `sites = data.drop_duplicates(coords)`, and the bulk Delta14C took its depth
    weights from `sites` too:

        weights = sites.filter(regex=r'^weight_\\d+').to_numpy()
        result['14C'] = np.nansum(bins*weights, axis=1)[rows]

    so every row at a location got the 14C computed with the weights of the
    location's first row. The processed Balesdent table has several rows per
    location (one per labeling duration), and at 13 of its 15 shared locations
    those rows have different carbon-depth weights. The bug gave 23 of the 99
    rows the wrong fm_obs, off by a median of 0.011 and at most 0.030. The fix
    weights each row's location profile by the row's own weights:

        result['14C'] = np.nansum(bins[rows]*weights_of_each_row, axis=1)

    Here two rows share a location, one with all its carbon in 0-20 cm and one
    with all of it in 80-100 cm. With the bug both get the shallow row's 14C.
    """
    shallow = [.5, .5] + [0.]*8           # all carbon in 0-20 cm
    deep = [0.]*8 + [.5, .5]              # all carbon in 80-100 cm
    data = pd.DataFrame([site_row(.1, .1, shallow), site_row(.1, .1, deep)])
    result = run_bulk(data)
    np.testing.assert_allclose(result['14C'], [expected_delta(shallow), expected_delta(deep)])
    np.testing.assert_allclose(result.fm_obs, result['14C']/1000 + 1)


def test_bins_without_weight_are_skipped():
    """A profile measured to 60 cm has NaN weights below it and is averaged over 0-60 cm."""
    weights = [1/6]*6 + [np.nan]*4
    result = run_bulk(pd.DataFrame([site_row(-.1, .2, weights)]))
    assert result['14C'].iloc[0] == pytest.approx(expected_delta(weights))


def test_turnover_is_stock_over_npp():
    """Turnover (yr) = stock (kg C/m²) / NPP (g C/m²/yr converted to kg), for each stock estimate."""
    result = run_bulk(pd.DataFrame([site_row(.1, .1, [.1]*10, stock=12.)]), npp=400.)
    assert result.turnover.iloc[0] == pytest.approx(12/.4)
    assert result.turnover_q05.iloc[0] == pytest.approx(6/.4)
    assert result.turnover_q95.iloc[0] == pytest.approx(24/.4)


def test_bulk_without_radiocarbon_or_positive_npp_gives_nan():
    """A row with no radiocarbon in any weighted bin gets NaN 14C (not 0, i.e. fm_obs = 1),
    and NPP <= 0 gives NaN turnover rather than an infinite or negative one."""
    data = pd.DataFrame([site_row(.1, .1, [.1]*10), site_row(-.1, -.1, [.1]*10)])
    npp = npp_table(data).assign(NPP=[500., 0.])
    shi = synthetic_shi()
    shi['temp'][:] = np.nan  # an empty map, so gap filling has nothing to fill from
    result = step02.bulk_turnover(data, shi, npp)
    assert result['14C'].isna().all() and result.fm_obs.isna().all()
    assert np.isfinite(result.turnover.iloc[0]) and np.isnan(result.turnover.iloc[1])


def layer_row(top, bottom, stock=2.):
    return dict(latitude=.1, longitude=.1, z_top_cm=top, z_bottom_cm=bottom, stock_kg_m2=stock,
                stock_kg_m2_q05=stock/2, stock_kg_m2_q95=stock*2)


def test_layer_radiocarbon_is_the_thickness_weighted_mean():
    """A layer's Delta14C is the mean over its one-cm levels [z, z+1); layers outside 0-100 cm get NaN."""
    data = pd.DataFrame([layer_row(10, 30), layer_row(0, 1), layer_row(90, 110)])
    result = step02c.layer_turnover(data, synthetic_shi(), npp_table(data))
    np.testing.assert_allclose(result['14C'][:2], [-10*19.5, 0.])
    assert np.isnan(result['14C'].iloc[2])
    assert not result.radiocarbon_spatially_filled.any()


def test_layer_turnover_is_stock_over_allocated_input():
    """The layer's input is its share of NPP under the allocation, and turnover = stock / input."""
    data = pd.DataFrame([layer_row(10, 30, stock=3.)])
    result = step02c.layer_turnover(data, synthetic_shi(), npp_table(data, npp=400.))
    expected_input = DEFAULT_ALLOCATION.layer_input(.4, 10, 30)
    assert result.npp_kg_m2_yr.iloc[0] == pytest.approx(.4)
    assert result.input_kg_m2_yr.iloc[0] == pytest.approx(expected_input)
    assert result.turnover.iloc[0] == pytest.approx(3/expected_input)
    assert result.turnover_q95.iloc[0] == pytest.approx(6/expected_input)


def test_source_cell_is_recorded_for_own_and_gap_filled_cells():
    """A site in an empty cell takes its profile from the nearest non-empty cell, and the
    tables record that cell and its distance; other sites record their own cell."""
    from soil_diskin.site_inputs import distance_km
    shi = synthetic_shi()
    shi['temp'][:, 1, 1] = np.nan  # empty cell at (0.25, 0.25)
    data = pd.DataFrame([site_row(.2, .3, [.1]*10), site_row(-.2, -.3, [.1]*10)])
    result = step02.bulk_turnover(data, shi, npp_table(data))
    assert result.radiocarbon_spatially_filled.tolist() == [True, False]
    src = result[['radiocarbon_source_latitude', 'radiocarbon_source_longitude']].to_numpy()
    assert tuple(src[1]) == (-.25, -.25)  # own cell
    assert tuple(src[0]) in {(-.25, .25), (.25, -.25)}  # an adjacent non-empty cell
    np.testing.assert_allclose(result.radiocarbon_source_km,
                               distance_km(data.Latitude, data.Longitude, src[:, 0], src[:, 1]))
    np.testing.assert_allclose(result['14C'], expected_delta([.1]*10))  # filled with the same profile


def test_profile_from_several_cells_raises():
    """If a site's gaps would be filled from different cells at different depths, a single
    source cell cannot describe it, so sampling refuses rather than picking one."""
    from soil_diskin.site_inputs import sample_radiocarbon
    level = np.arange(100)
    temp = np.full((100, 1, 3), np.nan)
    temp[:50, 0, 0] = -1.  # west cell has data only above 50 cm
    temp[50:, 0, 2] = -2.  # east cell only below
    shi = xr.Dataset({'temp': (('level', 'lat', 'lon'), temp)},
                     coords={'level': level, 'lat': [0.], 'lon': [-.5, 0., .5]})
    with pytest.raises(ValueError, match='different cells'):
        sample_radiocarbon(shi, pd.DataFrame({'latitude': [0.], 'longitude': [0.]}))
