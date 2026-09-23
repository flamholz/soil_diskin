"""Published cumulative root fractions and outcome-independent assignments."""
import numpy as np
import pandas as pd

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
