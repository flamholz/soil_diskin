"""Tests for soil_diskin.geo_utils.find_nearest.

Moved from tests/test_02_get_turnover_14C.py, whose other tests exercised mocks
rather than script 02.
"""
import numpy as np
import pandas as pd
import xarray as xr

from soil_diskin.geo_utils import find_nearest


def grid():
    """5x5 grid of values in [-50, 50) on y in [0, 40], x in [-80, -40]."""
    rng = np.random.default_rng(0)
    return xr.DataArray(rng.random((5, 5))*100 - 50, dims=['y', 'x'],
                        coords={'y': np.linspace(0, 40, 5), 'x': np.linspace(-80, -40, 5)},
                        name='delta_14C')


def test_find_nearest_with_numpy_array():
    """Points within the tolerance get the value of their nearest pixel."""
    data = grid()
    result = find_nearest(np.array([[10.0, -50.0], [20.0, -60.0]]), data, tolerance=10.0)
    assert isinstance(result, np.ndarray)
    assert len(result) == 2
    np.testing.assert_array_equal(result.ravel(), [data.sel(y=10, x=-50).item(), data.sel(y=20, x=-60).item()])


def test_find_nearest_with_dataframe():
    """A DataFrame of coordinates works like the equivalent array."""
    data = grid()
    coords = pd.DataFrame({'lat': [10.0, 20.0], 'lon': [-50.0, -60.0]})
    np.testing.assert_array_equal(find_nearest(coords, data, tolerance=10.0),
                                  find_nearest(coords.to_numpy(), data, tolerance=10.0))


def test_find_nearest_with_strict_tolerance():
    """Points farther than the tolerance from every pixel get NaN."""
    result = find_nearest(np.array([[100.0, -200.0]]), grid(), tolerance=0.1)
    assert np.isnan(result[0])
