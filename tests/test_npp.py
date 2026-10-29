"""Tests for site NPP: matching the NPP table to sites (soil_diskin.site_inputs) and
matching Earth Engine results to locations (soil_diskin.npp, with Earth Engine mocked).
"""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from soil_diskin.npp import fetch_modis_npp
from soil_diskin.site_inputs import npp_at, read_npp

NPP = pd.DataFrame({'latitude': [-28.666666666666668, 10.], 'longitude': [153.33333333333334, 20.],
                    'NPP': [1388.5520000000001, 258.568]})


def test_read_npp_round_trips_exactly(tmp_path):
    """The NPP table survives writing and reading bit for bit."""
    NPP.to_csv(tmp_path/'npp.csv', index=False)
    pd.testing.assert_frame_equal(read_npp(tmp_path/'npp.csv'), NPP, check_exact=True)


def test_npp_at_matches_locations_despite_last_bit_differences():
    """Rows get their location's NPP even when their coordinates differ from the table's in
    the last bits, as when two CSV readers parse the same text; invalid coordinates get NaN."""
    lat = np.nextafter(NPP.latitude[0], 0)
    data = pd.DataFrame({'latitude': [10., lat, 10., 95.], 'longitude': [20., NPP.longitude[0], 20., 0.]})
    np.testing.assert_array_equal(npp_at(data, NPP), [258.568, 1388.5520000000001, 258.568, np.nan])


def test_npp_at_raises_for_unknown_locations():
    """A valid location missing from the table means the table is out of date."""
    with pytest.raises(ValueError, match='no NPP'):
        npp_at(pd.DataFrame({'latitude': [5.], 'longitude': [5.]}), NPP)


def fake_earth_engine(table):
    """Mock `ee` and `geemap` modules whose query returns `table`."""
    geemap = MagicMock()
    geemap.ee_to_df.return_value = table
    return patch.dict(sys.modules, {'ee': MagicMock(), 'geemap': geemap})


def test_fetch_matches_results_to_locations_by_row_id():
    """Results are matched on the row id each point carries, whatever order they come back in,
    and a location without a valid pixel (no property) gets NaN."""
    locations = pd.DataFrame({'latitude': [0., 1., 2.], 'longitude': [0., 1., 2.]})
    returned = pd.DataFrame({'row': [2, 0, 1], 'mean': [30., 10., np.nan]})
    with fake_earth_engine(returned):
        result = fetch_modis_npp(locations)
    np.testing.assert_array_equal(result.NPP, [10., np.nan, 30.])


def test_fetch_with_sd_returns_both_statistics():
    locations = pd.DataFrame({'latitude': [0., 1.], 'longitude': [0., 1.]})
    returned = pd.DataFrame({'row': [1, 0], 'Npp_mean': [20., 10.], 'Npp_stdDev': [2., 1.]})
    with fake_earth_engine(returned):
        result = fetch_modis_npp(locations, sd=True)
    np.testing.assert_array_equal(result.NPP, [10., 20.])
    np.testing.assert_array_equal(result.NPP_sd, [1., 2.])


def test_fetch_rejects_unexpected_columns():
    locations = pd.DataFrame({'latitude': [0.], 'longitude': [0.]})
    with fake_earth_engine(pd.DataFrame({'row': [0], 'Npp': [10.]})), pytest.raises(ValueError, match='unexpected'):
        fetch_modis_npp(locations)
