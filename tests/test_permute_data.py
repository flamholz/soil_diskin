"""Tests for notebooks/02b_permute_data.py, the permutation null for the calibrations."""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1]/'notebooks/02b_permute_data.py'
permute = runpy.run_path(str(SCRIPT))['permute']


def sites():
    rng = np.random.default_rng(0)
    return pd.DataFrame({'Latitude': rng.uniform(-60, 60, 30), 'turnover': rng.uniform(5, 500, 30),
                         'fm_obs': rng.uniform(.5, 1.1, 30), 'turnover_q05': np.nan, 'turnover_q95': np.nan})


def test_each_replicate_shuffles_only_turnover_by_default():
    """Each row keeps its own fm_obs (and, for ISRaD, the sampling year whose atmosphere its
    radiocarbon refers to) and gets another row's turnover."""
    original = sites().assign(sampling_year=np.arange(1980, 2010))
    permuted = permute(original, replicates=3, seed=7)
    assert list(permuted.columns) == ['replicate', 'site_index', 'Latitude', 'turnover', 'fm_obs', 'sampling_year',
                                      'turnover_original']
    for _, frame in permuted.groupby('replicate'):
        frame = frame.set_index('site_index')
        # Every other column stays with its site; turnover keeps the same values in another order.
        pd.testing.assert_frame_equal(frame[['Latitude', 'fm_obs', 'sampling_year']],
                                      original[['Latitude', 'fm_obs', 'sampling_year']], check_names=False)
        np.testing.assert_array_equal(frame.turnover_original, original.turnover)
        np.testing.assert_array_equal(np.sort(frame.turnover), np.sort(original.turnover))
        assert (frame.turnover != original.turnover).sum() > 20
    # Replicates differ from each other.
    assert not np.array_equal(permuted.turnover[permuted.replicate == 0], permuted.turnover[permuted.replicate == 1])


def test_another_column_can_be_shuffled():
    """--column fm_obs shuffles radiocarbon instead, keeping each site's turnover."""
    original = sites()
    permuted = permute(original, replicates=2, seed=7, column='fm_obs')
    assert list(permuted.columns) == ['replicate', 'site_index', 'Latitude', 'turnover', 'fm_obs', 'fm_obs_original']
    for _, frame in permuted.groupby('replicate'):
        frame = frame.set_index('site_index')
        np.testing.assert_array_equal(frame.turnover, original.turnover)
        np.testing.assert_array_equal(frame.fm_obs_original, original.fm_obs)
        np.testing.assert_array_equal(np.sort(frame.fm_obs), np.sort(original.fm_obs))


def test_replicates_are_reproducible_and_prefix_stable():
    """Adding replicates never changes existing ones."""
    one = permute(sites(), replicates=1, seed=7)
    three = permute(sites(), replicates=3, seed=7)
    pd.testing.assert_frame_equal(one, three[three.replicate == 0])
    pd.testing.assert_frame_equal(three, permute(sites(), replicates=3, seed=7))
    assert not one.equals(permute(sites(), replicates=1, seed=8))


def test_only_included_rows_take_part():
    """Rows outside the atmospheric record are left out; site_index keeps the input row labels."""
    original = sites().assign(sampling_year=np.arange(1995, 2025))
    within = runpy.run_path(str(SCRIPT))['within_record'](original.sampling_year, 'data/14C_atm_annot.csv')
    assert within.sum() == 2013 - 1995 + 1
    permuted = permute(original, replicates=1, seed=7, include=within)
    np.testing.assert_array_equal(permuted.site_index, original.index[within])
    np.testing.assert_array_equal(np.sort(permuted.turnover), np.sort(original.turnover[within]))
