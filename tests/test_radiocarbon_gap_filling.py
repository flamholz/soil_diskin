"""Warn about sites whose radiocarbon is gap-filled from another Shi grid cell.

Step 02 takes each site's Delta14C profile from the nearest cell of the Shi et
al. (2020) map, or, where that cell is empty (e.g. on a small island), from the
nearest non-empty cell (see soil_diskin.site_inputs.sample_radiocarbon). The
tables flag such sites (radiocarbon_spatially_filled) and record the cell used
and its distance from the site. This test never fails: it issues one warning
per gap-filled location, so they are listed in pytest's warnings summary. Sites
are named by the Country column of the Balesdent workbook (country, and island or
region where given), if the workbook has been downloaded. The test is skipped if
the step 02 tables have not been made.

Every site is some distance from the centre of its own 0.5-degree cell (21 km
on median, at most 35 km), so distance alone does not single out gap filling.
"""
import warnings
from pathlib import Path

import pandas as pd
import pytest

from soil_diskin.site_inputs import COORDS, location_keys

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT/'data/balesdent_2018/balesdent_2018_raw.xlsx'
TABLES = {  # step 02 output -> its coordinate columns
    'results/all_sites_14C_turnover.csv': ['Latitude', 'Longitude'],
    'results/all_sites_14C_turnover_depth_resolved.csv': ['latitude', 'longitude'],
}


class GapFilledRadiocarbonWarning(UserWarning):
    """A site's radiocarbon comes from a grid cell other than its own."""


def site_names():
    """Country column of the Balesdent workbook by location (rounded coordinates); empty without the workbook."""
    if not WORKBOOK.exists():
        return {}
    profiles = pd.read_excel(WORKBOOK, sheet_name='Profiles', skiprows=7)[['Latitude', 'Longitude', 'Country']]
    profiles = profiles.dropna().set_axis([*COORDS, 'Country'], axis=1)
    keys = location_keys(profiles)
    names = profiles.Country.str.strip().groupby([keys.latitude, keys.longitude]).agg(lambda c: ' / '.join(sorted(set(c))))
    return names.to_dict()


@pytest.fixture(scope='module')
def names():
    return site_names()


@pytest.mark.parametrize('table', TABLES)
def test_warn_about_gap_filled_radiocarbon(table, names):
    path = ROOT/table
    if not path.exists():
        pytest.skip(f'{table} has not been made')
    coords = TABLES[table]
    data = pd.read_csv(path)
    filled = data[data.radiocarbon_spatially_filled.astype(bool)].drop_duplicates(coords)
    keys = location_keys(filled.set_axis(filled.columns.str.lower(), axis=1))
    for (_, site), key in zip(filled.iterrows(), keys.itertuples(index=False)):
        name = names.get(tuple(key), 'unnamed site')
        warnings.warn(f'{table}: {name} ({site[coords[0]]:.3f}, {site[coords[1]]:.3f}) has no Shi data in its '
                      f'cell; its radiocarbon is gap-filled from the cell at '
                      f'({site.radiocarbon_source_latitude:.2f}, {site.radiocarbon_source_longitude:.2f}), '
                      f'{site.radiocarbon_source_km:.0f} km away', GapFilledRadiocarbonWarning)
