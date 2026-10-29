"""Tests for prepare_turnover in notebooks/02_get_turnover_14C.py (bulk mode).

Uses a small synthetic Shi map whose Delta14C depends only on depth, and patches
out the Earth Engine NPP fetch.
"""
from importlib import import_module
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

step02 = import_module('notebooks.02_get_turnover_14C')

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


def run_bulk(data, npp=500.):
    n_locations = len(data[['Latitude', 'Longitude']].drop_duplicates())
    with patch.object(step02, 'fetch_npp', return_value=np.full(n_locations, npp)):
        return step02.prepare_turnover(data, synthetic_shi(), distribution=None)


def test_rows_at_one_location_use_their_own_weights():
    """Rows sharing a location must each weight Delta14C by their own carbon profile.

    Regression test for a bug inherited from the original script 02. NPP and
    the radiocarbon profile are looked up once per location, from
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
