"""Public data-adapter seam: profile identity, units, and complete-case selection."""

import numpy as np
import pandas as pd
import rioxarray as rio
import xarray as xr
import pytest

from soil_diskin.layered_data import prepare_profiles


def test_radiocarbon_matches_original_raster_extraction_and_precision(tmp_path):
    # Float32 values, a spatial tie, reversed rows, and averaging expose differences
    # between xarray ingestion and the original rasterio/interpolate/reshape path.
    values = (np.arange(900, dtype=np.float32).reshape(100, 3, 3)/7)-100
    values[5, 1, 1] = np.nan
    shi = xr.Dataset({'temp': (('level', 'lat', 'lon'), values)},
                     coords={'level': np.arange(100), 'lat': [-.5, 0., .5], 'lon': [-.5, 0., .5]})
    shi.lat.attrs['units'] = 'degrees_north'
    shi.lon.attrs['units'] = 'degrees_east'
    path = tmp_path/'shi.nc'
    shi.to_netcdf(path)
    with rio.open_rasterio(path, masked=True) as original:
        original = original.rio.write_crs('EPSG:4326').rio.write_nodata(np.nan)
        original = original.rio.interpolate_na(method='nearest')
        reference = xr.concat([original.sel(y=0., x=0., method='nearest')], dim='site')
        expected = 1+reference.values.reshape(-1, 10, 10).mean(axis=2)[0]/1000
    raw = pd.DataFrame({'Internal_profile_ID': ['test'], 'Latitude': [0.], 'Longitude': [0.]})
    for i in range(11):
        raw[f'Ctotal_0-{10*i}'] = float(i)
    npp = pd.DataFrame({'Latitude': [0.], 'Longitude': [0.], 'NPP': [500.]})
    prepared = prepare_profiles(raw, shi, npp)
    np.testing.assert_array_equal(prepared.profiles.fm_obs, expected)
    assert prepared.profiles.radiocarbon_spatially_filled.tolist() == [True]+[False]*9


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
    pd.testing.assert_frame_equal(prepared.raw_profiles, raw)
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

    partial = prepare_profiles(raw, shi, npp, allow_partial=True)
    retained = partial.profiles.query("profile_id == 'incomplete'")
    # A missing cumulative value at 50 cm removes both adjacent differences.
    assert retained.layer.tolist() == [0, 1, 2, 3, 6, 7, 8, 9]
    assert retained.z_top_cm.tolist() == [0., 10., 20., 30., 60., 70., 80., 90.]
    assert partial.excluded.layer.tolist() == [4, 5]
    np.testing.assert_allclose(retained.stock_kg_m2, 1.)
    np.testing.assert_allclose(retained.npp_kg_m2_yr, .5)


def test_radiocarbon_uses_spatial_filling_but_never_borrows_other_depths():
    raw = pd.DataFrame({'Internal_profile_ID': ['missing-radio', 'complete'],
                        'Latitude': [0., 2.], 'Longitude': [0., 0.]})
    for i in range(11):
        raw[f'Ctotal_0-{10*i}'] = float(i)
    values = np.zeros((100, 2, 1))
    values[5, 0, 0] = np.nan
    values[5, 1, 0] = 200.
    shi = xr.Dataset({'temp': (('level', 'lat', 'lon'), values)},
                     coords={'level': np.arange(100), 'lat': [0., 2.], 'lon': [0.]})
    npp = pd.DataFrame({'Latitude': [0., 2.], 'Longitude': [0., 0.], 'NPP': [500., 500.]})
    prepared = prepare_profiles(raw, shi, npp)
    assert prepared.profiles.profile_id.unique().tolist() == ['missing-radio', 'complete']
    assert prepared.profiles.fnew_obs.isna().all()
    assert prepared.profiles.duration_years.isna().all()
    assert prepared.excluded.empty
    top = prepared.profiles.query('layer == 0')
    np.testing.assert_allclose(top.fm_obs, 1.02)
    assert top.radiocarbon_spatially_filled.tolist() == [True, False]

    partial = prepare_profiles(raw, shi, npp, allow_partial=True)
    assert len(partial.profiles) == 20
    assert partial.profiles.fnew_obs.isna().all()  # Evaluation data never gates fitting.
    assert partial.excluded.empty
    # No spatial source at this depth: both profiles must still lose layer 2.
    unavailable = shi.copy(deep=True)
    unavailable['temp'][25, :, :] = np.nan
    incomplete = prepare_profiles(raw, unavailable, npp, allow_partial=True)
    assert len(incomplete.profiles) == 18
    assert incomplete.excluded.layer.tolist() == [2, 2]
    assert incomplete.excluded.reason.eq('incomplete radiocarbon after spatial filling').all()
    no_npp = prepare_profiles(raw, shi, npp.iloc[[1]], allow_partial=True)
    assert no_npp.profiles.profile_id.unique().tolist() == ['complete']
    assert 'missing or nonpositive NPP' in no_npp.excluded.reason.iloc[0]
    assert pd.isna(no_npp.excluded.layer.iloc[0])  # Whole-profile exclusion.
    empty = prepare_profiles(raw, shi, npp.iloc[:0], allow_partial=True)
    assert empty.profiles.empty and len(empty.excluded) == 2
    assert empty.excluded.reason.eq('missing or nonpositive NPP').all()
    conflicting = pd.concat([npp, npp.iloc[[0]].assign(NPP=600.)], ignore_index=True)
    with pytest.raises(ValueError, match='conflicting cached NPP'):
        prepare_profiles(raw, shi, conflicting)


def test_npp_matching_recovers_coordinate_roundoff_without_borrowing_nearby_values():
    latitude, longitude = -19.433333333333334, -44.166666666666664
    raw = pd.DataFrame({'Internal_profile_ID': ['roundoff', 'nearby', 'missing'],
                        'Latitude': [latitude, latitude + 1e-6, 10.], 'Longitude': longitude})
    for i in range(11):
        raw[f'Ctotal_0-{10*i}'] = float(i)
    shi = xr.Dataset({'temp': (('level', 'lat', 'lon'), np.zeros((100, 1, 1)))},
                     coords={'level': np.arange(100), 'lat': [latitude], 'lon': [longitude]})
    npp = pd.DataFrame({'Latitude': [np.nextafter(latitude, -np.inf), 10.],
                        'Longitude': longitude, 'NPP': [1093.076, np.nan]})
    result = prepare_profiles(raw, shi, npp, allow_partial=True)
    assert result.profiles.profile_id.unique().tolist() == ['roundoff']
    np.testing.assert_allclose(result.profiles.npp_kg_m2_yr, 1.093076)
    assert result.profiles.latitude.eq(latitude).all()  # Original coordinates are preserved.
    assert result.excluded.profile_id.tolist() == ['nearby', 'missing']
    assert result.excluded.reason.eq('missing or nonpositive NPP').all()

    conflicting = pd.concat([npp, npp.iloc[[0]].assign(Latitude=latitude, NPP=900.)])
    with pytest.raises(ValueError, match='conflicting cached NPP'):
        prepare_profiles(raw, shi, conflicting)
