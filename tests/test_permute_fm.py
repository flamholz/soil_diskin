"""Tests for notebooks/02b_permute_fm.py, the permuted-fm null for the calibrations."""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1]/'notebooks/02b_permute_fm.py'
permute_fm = runpy.run_path(str(SCRIPT))['permute_fm']


def sites():
    rng = np.random.default_rng(0)
    return pd.DataFrame({'Latitude': rng.uniform(-60, 60, 30), 'turnover': rng.uniform(5, 500, 30),
                         'fm_obs': rng.uniform(.5, 1.1, 30), 'turnover_q05': np.nan, 'turnover_q95': np.nan})


def test_each_replicate_shuffles_only_fm():
    original = sites()
    permuted = permute_fm(original, replicates=3, seed=7)
    assert list(permuted.columns) == ['replicate', 'site_index', 'Latitude', 'turnover', 'fm_obs', 'fm_obs_original']
    for replicate, frame in permuted.groupby('replicate'):
        frame = frame.set_index('site_index')
        # Every other column stays with its site; fm_obs keeps the same values in another order.
        pd.testing.assert_frame_equal(frame[['Latitude', 'turnover']], original[['Latitude', 'turnover']],
                                      check_names=False)
        np.testing.assert_array_equal(frame.fm_obs_original, original.fm_obs)
        np.testing.assert_array_equal(np.sort(frame.fm_obs), np.sort(original.fm_obs))
        assert (frame.fm_obs != original.fm_obs).sum() > 20
    # Replicates differ from each other.
    assert not np.array_equal(permuted.fm_obs[permuted.replicate == 0], permuted.fm_obs[permuted.replicate == 1])


def test_replicates_are_reproducible_and_prefix_stable():
    """Adding replicates never changes existing ones."""
    one = permute_fm(sites(), replicates=1, seed=7)
    three = permute_fm(sites(), replicates=3, seed=7)
    pd.testing.assert_frame_equal(one, three[three.replicate == 0])
    pd.testing.assert_frame_equal(three, permute_fm(sites(), replicates=3, seed=7))
    assert not one.equals(permute_fm(sites(), replicates=1, seed=8))


def test_shuffling_turnover_keeps_radiocarbon_and_year_with_each_profile():
    """For ISRaD, turnover is shuffled: each row keeps its own fm_obs and sampling year (whose
    atmosphere its radiocarbon refers to) and gets another row's turnover."""
    original = sites().assign(sampling_year=np.arange(1980, 2010))
    permuted = permute_fm(original, replicates=2, seed=7, column='turnover')
    assert list(permuted.columns)[-1] == 'turnover_original'
    for _, frame in permuted.groupby('replicate'):
        frame = frame.set_index('site_index')
        np.testing.assert_array_equal(frame.fm_obs, original.fm_obs)
        np.testing.assert_array_equal(frame.sampling_year, original.sampling_year)
        np.testing.assert_array_equal(frame.turnover_original, original.turnover)
        np.testing.assert_array_equal(np.sort(frame.turnover), np.sort(original.turnover))
        assert (frame.turnover != original.turnover).sum() > 20


def test_only_included_rows_take_part():
    """Rows outside the atmospheric record are left out; site_index keeps the input row labels."""
    original = sites().assign(sampling_year=np.arange(1995, 2025))
    within = runpy.run_path(str(SCRIPT))['within_record'](original.sampling_year, 'data/14C_atm_annot.csv')
    assert within.sum() == 2013 - 1995 + 1
    permuted = permute_fm(original, replicates=1, seed=7, column='turnover', include=within)
    np.testing.assert_array_equal(permuted.site_index, original.index[within])
    np.testing.assert_array_equal(np.sort(permuted.turnover), np.sort(original.turnover[within]))
