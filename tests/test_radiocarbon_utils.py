import numpy as np
import pandas as pd

import soil_diskin.radiocarbon_utils as rc


def _make_df(raw_age: np.ndarray, raw_fm: np.ndarray) -> pd.DataFrame:
    """Create a minimal mock atmospheric 14C dataframe with expected columns."""
    df = pd.DataFrame(
        {
            "c0": np.zeros_like(raw_age),
            "c1": np.zeros_like(raw_age),
            "c2": np.zeros_like(raw_age),
            "years_before_2000": raw_age,
            "R_14C": raw_fm,
        }
    )
    return df


def test_atmc14_stores_float64_contiguous_arrays():
    """AtmC14 should coerce arrays to contiguous float64 and scalar mean_R."""
    ages = np.array([0, 1, 2], dtype=np.int32)
    fm = np.array([0.9, 1.0, 1.1], dtype=np.float32)
    atm = rc.AtmC14(ages, fm, mean_R=np.float32(1.03))

    assert atm.ages.dtype == np.float64
    assert atm.fm.dtype == np.float64
    assert atm.ages.flags["C_CONTIGUOUS"]
    assert atm.fm.flags["C_CONTIGUOUS"]
    assert isinstance(atm.mean_R, float)


def test_load_atm14c_sorts_filters_and_uses_last_50k_mean(monkeypatch):
    """Loader should sort/filter data and compute mean_R from last 50k raw rows."""
    n = 50_020
    raw_age = np.linspace(-20.0, 1000.0, n)
    raw_fm = np.linspace(0.7, 1.3, n)

    rng = np.random.default_rng(42)
    order = rng.permutation(n)
    shuffled_age = raw_age[order]
    shuffled_fm = raw_fm[order]
    df = _make_df(shuffled_age, shuffled_fm)

    monkeypatch.setattr(rc.pd, "read_csv", lambda _: df)
    atm = rc.load_atm14c("dummy.csv")

    # Testing existing behavior: mean_R is computed from the last 50k rows of the original
    # unclear if this is the right behavior, however. 
    expected_mean = float(shuffled_fm[-50_000:].mean())
    np.testing.assert_allclose(atm.mean_R, expected_mean, rtol=0.0, atol=0.0)

    perm = np.argsort(shuffled_age)
    a_sorted = shuffled_age[perm]
    f_sorted = shuffled_fm[perm]
    mask = a_sorted >= 0.0

    np.testing.assert_allclose(atm.ages, a_sorted[mask], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(atm.fm, f_sorted[mask], rtol=0.0, atol=0.0)
    assert np.all(atm.ages >= 0.0)
    assert np.all(np.diff(atm.ages) >= 0.0)


def test_load_atm14c_uses_all_rows_when_less_than_50k(monkeypatch):
    """For short inputs, the tail-mean logic should reduce to full-array mean."""
    raw_age = np.array([5.0, -1.0, 2.0, 0.0, 3.0], dtype=np.float64)
    raw_fm = np.array([1.01, 0.99, 1.02, 1.00, 1.03], dtype=np.float64)
    df = _make_df(raw_age, raw_fm)

    monkeypatch.setattr(rc.pd, "read_csv", lambda _: df)
    atm = rc.load_atm14c("dummy.csv")

    np.testing.assert_allclose(atm.mean_R, raw_fm.mean(), rtol=0.0, atol=0.0)
