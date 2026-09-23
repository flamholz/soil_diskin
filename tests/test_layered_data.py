"""Public data-adapter seam: profile identity, units, and complete-case selection."""

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from soil_diskin.layered_data import prepare_profiles


def test_profiles_remain_distinct_and_missing_layer_values_are_excluded():
    raw = pd.DataFrame({'Internal_profile_ID': ['pasture', 'forest', 'incomplete'],
                        'Latitude': [1., 1., 1.], 'Longitude': [2., 2., 2.],
                        'Duration_labeling': [10., 30., 20.]})
    for i in range(11):
        raw[f'Ctotal_0-{10*i}'] = [float(i), float(2*i), float(i)]
        raw['Cnew_0_0' if i == 0 else f'Cnew_0-{10*i}'] = [0.2*i, 0.8*i, 0.2*i]
    raw.loc[2, 'Ctotal_0-50'] = np.nan
    npp = pd.DataFrame({'Latitude': [1., 1.], 'Longitude': [2., 2.], 'NPP': [500., 500.]})
    # Deliberately shuffled dimensions: depth aggregation must use named axes.
    shi = xr.Dataset({'temp': (('lon', 'level', 'lat'), np.arange(100.).reshape(1,100,1))},
                     coords={'lat': [1.], 'lon': [2.], 'level': np.arange(100)})
    prepared = prepare_profiles(raw, shi, npp)
    assert prepared.profiles.profile_id.unique().tolist() == ['pasture', 'forest']
    assert len(prepared.profiles) == 20
    pasture = prepared.profiles.query("profile_id == 'pasture'")
    forest = prepared.profiles.query("profile_id == 'forest'")
    np.testing.assert_allclose(pasture.stock_kg_m2, 1)
    np.testing.assert_allclose(forest.stock_kg_m2, 2)
    np.testing.assert_allclose(pasture.npp_kg_m2_yr, 0.5)
    np.testing.assert_allclose(pasture.fm_obs.iloc[:2], [1.0045, 1.0145])
    np.testing.assert_allclose(pasture.fnew_obs, 0.2)
    np.testing.assert_allclose(forest.fnew_obs, 0.4)
    assert forest.duration_years.unique().tolist() == [30.]
    assert prepared.excluded.profile_id.tolist() == ['incomplete']
    assert 'stock' in prepared.excluded.reason.iloc[0]


def test_missing_native_radiocarbon_is_excluded_but_missing_evaluation_data_is_allowed():
    raw = pd.DataFrame({'Internal_profile_ID': ['missing-radio', 'complete'],
                        'Latitude': [0., 2.], 'Longitude': [0., 0.]})
    for i in range(11):
        raw[f'Ctotal_0-{10*i}'] = float(i)
    values = np.zeros((100, 2, 1))
    values[5, 0, 0] = np.nan
    shi = xr.Dataset({'temp': (('level', 'lat', 'lon'), values)},
                     coords={'level': np.arange(100), 'lat': [0., 2.], 'lon': [0.]})
    npp = pd.DataFrame({'Latitude': [0., 2.], 'Longitude': [0., 0.], 'NPP': [500., 500.]})
    prepared = prepare_profiles(raw, shi, npp)
    assert prepared.profiles.profile_id.unique().tolist() == ['complete']
    assert prepared.profiles.fnew_obs.isna().all()
    assert prepared.profiles.duration_years.isna().all()
    assert prepared.excluded.reason.tolist() == ['incomplete native-cell radiocarbon']
    conflicting = pd.concat([npp, npp.iloc[[0]].assign(NPP=600.)], ignore_index=True)
    with pytest.raises(ValueError, match='conflicting cached NPP'):
        prepare_profiles(raw, shi, conflicting)
