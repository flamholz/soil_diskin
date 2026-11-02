"""Tests for the ISRaD tables: soil_diskin.israd (layers -> pseudo-bulk profiles) and the
ISRaD step 01/02 scripts, on a small synthetic ISRaD flat layer table.
"""
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from soil_diskin.constants import LAMBDA_14C
from soil_diskin.israd import load_layers, permafrost_or_peat, pseudo_bulk

step01 = import_module('notebooks.01_preprocess_israd')
step02 = import_module('notebooks.02_get_israd_turnover')

COLUMNS = ['entry_name', 'site_name', 'pro_name', 'lyr_name', 'site_lat', 'site_long', 'lyr_top', 'lyr_bot',
           'lyr_obs_date_y', 'lyr_c_org', 'lyr_c_tot', 'lyr_soc', 'lyr_bd_samp', 'lyr_bd_tot', 'lyr_14c',
           'lyr_fraction_modern', 'pro_permafrost', 'pro_peatland', 'pro_thaw_depth', 'pro_usda_soil_order']


def layer(profile, top, bottom, soc=None, d14c=0., fm=None, year=2000, c_org=None, c_tot=None, bd=None, bd_tot=None,
          lat=10., lon=20.):
    return dict(entry_name='E', site_name=profile, pro_name=profile, lyr_name=f'{profile}{top}', site_lat=lat,
                site_long=lon, lyr_top=top, lyr_bot=bottom, lyr_obs_date_y=year, lyr_c_org=c_org, lyr_c_tot=c_tot,
                lyr_soc=soc, lyr_bd_samp=bd, lyr_bd_tot=bd_tot, lyr_14c=None if fm is not None else d14c,
                lyr_fraction_modern=fm)


def profiles_from(tmp_path, layers):
    path = tmp_path/'layers.csv'
    pd.DataFrame(layers, columns=COLUMNS).to_csv(path, index=False)
    return pseudo_bulk(load_layers(path)).set_index('site_name')


def test_complete_profile_sums_stock_and_weights_radiocarbon(tmp_path):
    """Two abutting layers covering 0-100 cm: stock in kg C/m² (lyr_soc in g C/cm² x 10) and
    stock-weighted R = Delta14C/1000 + 1."""
    p = profiles_from(tmp_path, [layer('A', 0, 30, soc=.3, d14c=100.), layer('A', 30, 100, soc=.1, d14c=-200.)])
    assert p.loc['A', 'stock_kg_m2'] == pytest.approx(4.)
    assert p.loc['A', 'R'] == pytest.approx((3*1.1 + 1*.8)/4)
    assert p.loc['A', 'n_layers'] == 2 and p.loc['A', 'n_fallback_layers'] == 0


def test_layer_crossing_100_cm_is_prorated_and_organic_horizon_excluded(tmp_path):
    """A 50-150 cm layer counts half its stock; an O horizon (-5 to 0 cm) is not part of the mineral column."""
    p = profiles_from(tmp_path, [layer('B', -5, 0, soc=1., d14c=500.), layer('B', 0, 50, soc=.2),
                                 layer('B', 50, 150, soc=.2)])
    assert p.loc['B', 'stock_kg_m2'] == pytest.approx(2. + 1.)
    assert p.loc['B', 'R'] == pytest.approx(1.)  # the O horizon's 500 permil is excluded


@pytest.mark.parametrize('layers', [
    [layer('C', 0, 30, soc=.1), layer('C', 40, 100, soc=.1)],  # gap
    [layer('C', 0, 50, soc=.1), layer('C', 40, 100, soc=.1)],  # overlap
    [layer('C', 0, 30, soc=.1), layer('C', 30, 80, soc=.1)],   # stops at 80 cm
    [layer('C', 10, 100, soc=.1)],                             # starts at 10 cm
])
def test_incomplete_or_overlapping_profiles_are_dropped(tmp_path, layers):
    assert 'C' not in profiles_from(tmp_path, layers).index


def test_stock_fallbacks_and_fm_conversion(tmp_path):
    """Without lyr_soc the stock is carbon % x bulk density x thickness; using total carbon or
    bulk density with coarse fragments counts as a fallback. With only fraction modern,
    R = Fm exp(-lambda (year - 1950))."""
    p = profiles_from(tmp_path, [
        layer('D', 0, 50, c_org=1., bd=1.2, fm=1.05, year=2010),
        layer('D', 50, 100, c_tot=.5, bd_tot=1.4, fm=1.05, year=2010)])
    assert p.loc['D', 'stock_kg_m2'] == pytest.approx((1/100*1.2*50 + .5/100*1.4*50)*10)
    assert p.loc['D', 'n_fallback_layers'] == 1
    assert p.loc['D', 'R'] == pytest.approx(1.05*np.exp(-60*LAMBDA_14C))


def test_preprocess_drops_implausible_stocks_and_measures_distance_to_balesdent(tmp_path):
    """Profiles above the stock limit are dropped and reported; the table is in kg C/m² with 14C in
    per mil, and gives each profile's distance to the nearest Balesdent site."""
    layers = [layer('ok', 0, 100, soc=1., d14c=50., lat=0., lon=0.), layer('huge', 0, 100, soc=100., lat=0., lon=1.)]
    pd.DataFrame(layers, columns=COLUMNS).to_csv(tmp_path/'layers.csv', index=False)
    pd.DataFrame({'Latitude': [0.], 'Longitude': [.1]}).to_csv(tmp_path/'balesdent.csv', index=False)
    profiles, summary = step01.preprocess(tmp_path/'layers.csv', tmp_path/'balesdent.csv', max_stock=150.)
    assert profiles.site_name.tolist() == ['ok']
    assert summary['dropped_stock_above_max'] == {'E': {'profile_years': 1, 'median_stock_kg_m2': 1000.}}
    row = profiles.iloc[0]
    assert (row.stock_kg_m2, row['14C'], row.fm_obs) == pytest.approx((10., 50., 1.05))
    assert row.km_to_nearest_balesdent_site == pytest.approx(11.1, abs=.1)  # 0.1 degree of longitude at the equator


def test_turnover_in_years_from_kg_and_drops_profiles_without_npp():
    """turnover = stock_kg_m2 / (NPP g C/m²/yr / 1000); profiles without NPP > 0 are dropped."""
    profiles = pd.DataFrame({'entry_name': 'E', 'site_name': ['a', 'b'], 'pro_name': ['a', 'b'], 'sampling_year': 2000,
                             'latitude': [0., 1.], 'longitude': [0., 1.], 'stock_kg_m2': [10., 10.]})
    npp = pd.DataFrame({'latitude': [0., 1.], 'longitude': [0., 1.], 'NPP': [500., 0.], 'NPP_sd': [50., 0.]})
    result, dropped = step02.israd_turnover(profiles, npp)
    assert result.site_name.tolist() == ['a'] and dropped.site_name.tolist() == ['b']
    assert (result.npp_kg_m2_yr[0], result.npp_sd_kg_m2_yr[0], result.turnover[0]) == pytest.approx((.5, .05, 20.))


def test_preprocess_drops_excluded_entries_and_records_why(tmp_path):
    """Entries in EXCLUDED_ENTRIES (e.g. Min_2025) are dropped whole, and the summary gives the reason."""
    layers = [layer('kept', 0, 100, soc=1.), {**layer('bad', 0, 100, soc=1.), 'entry_name': 'Min_2025'}]
    pd.DataFrame(layers, columns=COLUMNS).to_csv(tmp_path/'layers.csv', index=False)
    pd.DataFrame({'Latitude': [0.], 'Longitude': [0.]}).to_csv(tmp_path/'balesdent.csv', index=False)
    profiles, summary = step01.preprocess(tmp_path/'layers.csv', tmp_path/'balesdent.csv')
    assert profiles.entry_name.tolist() == ['E']
    assert summary['excluded_entries']['Min_2025']['profile_years'] == 1
    assert summary['excluded_entries']['Min_2025']['reason'] == step01.EXCLUDED_ENTRIES['Min_2025']


def test_permafrost_and_peat_profiles_are_excluded_with_their_evidence(tmp_path):
    """Profiles reported as permafrost or peat (flag, thaw depth, or Gelisol/Histosol soil order)
    are dropped, and the summary records the evidence; other soil orders are kept."""
    layers = [{**layer('frozen', 0, 100, soc=1.), 'pro_usda_soil_order': 'Gelisols', 'pro_thaw_depth': 60.},
              {**layer('peat', 0, 100, soc=1.), 'pro_peatland': 'Yes'},
              {**layer('mineral', 0, 100, soc=1.), 'pro_usda_soil_order': 'Mollisols'}]
    pd.DataFrame(layers, columns=COLUMNS).to_csv(tmp_path/'layers.csv', index=False)
    pd.DataFrame({'Latitude': [0.], 'Longitude': [0.]}).to_csv(tmp_path/'balesdent.csv', index=False)
    profiles, summary = step01.preprocess(tmp_path/'layers.csv', tmp_path/'balesdent.csv')
    assert profiles.site_name.tolist() == ['mineral']
    evidence = {r['site_name']: r['evidence'] for r in summary['excluded_permafrost_or_peat']}
    assert evidence == {'frozen': 'thaw depth 60 cm; soil order Gelisols (permafrost)', 'peat': 'pro_peatland = yes'}


def test_permafrost_or_peat_flags_any_layer_and_reports_all_evidence():
    """Each kind of evidence flags a profile-year, from any of its layers; matching ignores case and
    stray spaces; 'no' flags, other soil orders and non-numeric thaw depths do not flag."""
    rows = [
        {**layer('permafrost', 0, 30), 'pro_permafrost': ' YES '},
        # Evidence on the second layer only still flags the profile-year.
        layer('peat', 0, 30), {**layer('peat', 30, 100), 'pro_usda_soil_order': 'Histosols '},
        # Every kind at once, reported in a fixed order.
        {**layer('all', 0, 100), 'pro_permafrost': 'yes', 'pro_peatland': 'yes', 'pro_thaw_depth': 45.,
         'pro_usda_soil_order': 'gelisols'},
        {**layer('all', 100, 150), 'pro_usda_soil_order': 'Histosols'},
        # Not flagged.
        {**layer('negative', 0, 100), 'pro_permafrost': 'no', 'pro_peatland': 'No',
         'pro_thaw_depth': 'unknown', 'pro_usda_soil_order': 'Mollisols'},
        layer('unreported', 0, 100),
        # Profile-years are separate: only the 2010 sampling reports a thaw depth.
        layer('resampled', 0, 100, year=1990), {**layer('resampled', 0, 100, year=2010), 'pro_thaw_depth': 80.},
    ]
    reasons = permafrost_or_peat(pd.DataFrame(rows, columns=COLUMNS))
    assert reasons == {
        ('E', 'permafrost', 'permafrost', 2000): 'pro_permafrost = yes',
        ('E', 'peat', 'peat', 2000): 'soil order Histosols (peat)',
        ('E', 'all', 'all', 2000): 'pro_permafrost = yes; pro_peatland = yes; thaw depth 45 cm; '
                                   'soil order Gelisols (permafrost); soil order Histosols (peat)',
        ('E', 'resampled', 'resampled', 2010): 'thaw depth 80 cm',
    }


def test_permafrost_or_peat_keeps_profiles_with_a_missing_identifier():
    """A profile-year missing part of its key (e.g. no pro_name) is still checked, not silently skipped."""
    rows = [{**layer('frozen', 0, 100), 'pro_name': None, 'pro_permafrost': 'yes'}]
    reasons = permafrost_or_peat(pd.DataFrame(rows, columns=COLUMNS))
    assert list(reasons.values()) == ['pro_permafrost = yes']
