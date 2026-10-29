"""Warn about ISRaD profiles close to a Balesdent site.

An ISRaD profile near a Balesdent site may sample the same soils, so the two
datasets are not fully independent there. This test never fails: it issues one
warning per ISRaD location within MAX_KM of a Balesdent site
(km_to_nearest_balesdent_site, from 01_preprocess_israd.py), so they are listed
in pytest's warnings summary. It is skipped if the ISRaD profile table has not
been made (`snakemake israd`).
"""
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT/'results/israd/israd_profiles.csv'
MAX_KM = 5.


class NearBalesdentWarning(UserWarning):
    """An ISRaD profile lies within MAX_KM of a Balesdent site."""


def test_warn_about_israd_profiles_near_balesdent_sites():
    if not PROFILES.exists():
        pytest.skip(f'{PROFILES.relative_to(ROOT)} has not been made')
    profiles = pd.read_csv(PROFILES)
    near = profiles[profiles.km_to_nearest_balesdent_site <= MAX_KM]
    for (entry, site, lat, lon), group in near.groupby(['entry_name', 'site_name', 'latitude', 'longitude']):
        warnings.warn(f'ISRaD {entry} / {site} ({lat:.3f}, {lon:.3f}), {len(group)} profile-year(s), is '
                      f'{group.km_to_nearest_balesdent_site.iloc[0]:.1f} km from a Balesdent site', NearBalesdentWarning)
