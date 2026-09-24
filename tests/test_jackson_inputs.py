"""Published cumulative root fractions and outcome-independent assignments."""
import numpy as np
import pandas as pd
import pytest

from notebooks.compare_jackson_inputs import JACKSON_BETA, jackson_assignments
from soil_diskin.layered_lognormal import input_weights


def test_published_root_cdf_matches_existing_exponential_layer_inputs():
    assert JACKSON_BETA == {'global': .966, 'crop': .961, 'grass': .952, 'tree': .970, 'shrub': .978}
    for beta in JACKSON_BETA.values():
        depths = np.arange(0., 101., 10.)
        # Independent paper equation: integrate its cumulative distribution.
        expected = np.diff(1-beta**depths)/(1-beta**100)
        weights = input_weights(float(-1/np.log(beta)))
        np.testing.assert_allclose(weights, expected, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(weights.sum(), 1., rtol=1e-14)


def test_vegetation_assignments_use_metadata_and_flag_ambiguous_groups():
    raw = pd.DataFrame({'Internal_profile_ID': list('abcdefgh'),
        'Land_Use': ['CROP', 'GRASSLAND', 'FOREST', 'FOREST', 'FOREST', 'GRASSLAND', 'GRASSLAND', 'unknown'],
        'Vegetation': ['C3C4 mix', 'Brachiaria pasture', 'Pinus', 'Shrub Prosopsis glandulosa',
                       'C3C4 sylvopastoral', 'C4 savanna', 'FACE Trifolium', 'unknown'],
        'fnew_obs': np.arange(8)/10})
    assignment = jackson_assignments(raw)
    assert assignment.jackson_group.tolist() == ['crop', 'grass', 'tree', 'shrub', 'global', 'global', 'global', 'global']
    assert assignment.profile_id.tolist() == list('abcdefgh')
    assert assignment.global_fallback.tolist() == [False]*4 + [True]*4
    pd.testing.assert_frame_equal(assignment, jackson_assignments(raw.assign(fnew_obs=.9)))


def test_half_surface_input_conserves_npp_and_keeps_jackson_roots_in_top_layer():
    # Independent cumulative-root calculation with an extra half-NPP at the surface.
    beta = .966
    roots = np.diff(1-beta**np.arange(0., 101., 10.))/(1-beta**100)
    weights = input_weights(float(-1/np.log(beta)), surface_fraction=.5)
    np.testing.assert_allclose(weights[0], .5+.5*roots[0], rtol=1e-13)
    np.testing.assert_allclose(weights[1:], .5*roots[1:], rtol=1e-13)
    np.testing.assert_allclose(weights.sum(), 1., rtol=1e-14)
    assert weights[0] > .5
    # A wholly surface input would leave zero input for the fitted lower layers.
    for invalid in [-.01, 1., np.nan, np.inf]:
        with pytest.raises(ValueError, match='surface_fraction'):
            input_weights(30., surface_fraction=invalid)


def test_comparison_uses_retained_metadata_and_shared_allocation(tmp_path, monkeypatch):
    import json
    from notebooks import compare_jackson_inputs as experiment
    from soil_diskin.layered_data import PreparedProfiles
    from soil_diskin.radiocarbon_utils import AtmC14

    profiles = pd.DataFrame({'profile_id': ['a', 'b'], 'layer': [0, 4],
        'latitude': [0., 1.], 'longitude': [0., 1.], 'z_top_cm': [0., 40.], 'z_bottom_cm': [10., 50.],
        'stock_kg_m2': [1., 1.], 'fm_obs': [.9, .8], 'npp_kg_m2_yr': [.5, .5],
        'duration_years': [20., 20.], 'fnew_obs': [.2, .1]})
    raw = pd.DataFrame({'Internal_profile_ID': ['a', 'b'], 'Land_Use': ['CROP', 'FOREST'],
                        'Vegetation': ['maize', 'pine']})
    # No workbook path is present: the driver must use the retained metadata.
    prepared = PreparedProfiles(profiles, pd.DataFrame(), raw_profiles=raw)
    monkeypatch.setattr(experiment, 'load_profiles', lambda **kwargs: prepared)
    monkeypatch.setattr(experiment, 'load_atm14c', lambda: AtmC14(np.array([0.]), np.array([1.]), 1.))
    experiment.run_comparison(tmp_path, surface_fraction=.5, soil_npp_fraction=.5)
    summary = pd.read_csv(tmp_path/'metrics.csv')
    assert len(summary) == 5 and summary.n_layer_pairs.eq(2).all()
    assert summary.eligible.all()
    assignments = pd.read_csv(tmp_path/'vegetation_assignments.csv')
    assert assignments.jackson_group.tolist() == ['crop', 'tree']
    assert json.loads((tmp_path/'protocol.json').read_text())['status'] == 'complete'
    fitted = pd.read_csv(tmp_path/'jackson_vegetation_surface/layers.csv')
    assert fitted.surface_fraction.eq(.5).all() and fitted.soil_npp_fraction.eq(.5).all()
    assert (tmp_path/'comparison.png').is_file()
