import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

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
    """AtmC14 should coerce arrays to contiguous float64 and a scalar tail_fm."""
    ages = np.array([0, 1, 2], dtype=np.int32)
    fm = np.array([0.9, 1.0, 1.1], dtype=np.float32)
    atm = rc.AtmC14(ages, fm, tail_fm=np.float32(1.03))

    assert atm.ages.dtype == np.float64
    assert atm.fm.dtype == np.float64
    assert atm.ages.flags["C_CONTIGUOUS"]
    assert atm.fm.flags["C_CONTIGUOUS"]
    assert isinstance(atm.tail_fm, float)
    # The tail defaults to the last value.
    assert rc.AtmC14(ages, fm).tail_fm == np.float32(1.1)


def test_load_atm14c_sorts_filters_and_holds_last_value(monkeypatch):
    """Loader should sort by age, drop negative ages and hold the oldest value."""
    n = 1_000
    raw_age = np.linspace(-20.0, 1000.0, n)
    raw_fm = np.linspace(0.7, 1.3, n)

    rng = np.random.default_rng(42)
    order = rng.permutation(n)
    df = _make_df(raw_age[order], raw_fm[order])

    monkeypatch.setattr(rc.pd, "read_csv", lambda _: df)
    atm = rc.load_atm14c("dummy.csv")

    mask = raw_age >= 0.0
    np.testing.assert_array_equal(atm.ages, raw_age[mask])
    np.testing.assert_array_equal(atm.fm, raw_fm[mask])
    assert atm.tail_fm == raw_fm[-1]


def test_R_is_piecewise_constant_with_constant_tail():
    atm = rc.AtmC14(np.array([0.0, 10.0, 20.0]), np.array([1.0, 2.0, 3.0]), tail_fm=4.0)
    ages = np.array([0.0, 5.0, 10.0, 19.9, 20.0, 1e6])
    np.testing.assert_array_equal(atm.R(ages), [1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    assert atm.R(5.0) == 1.0


def test_laplace_matches_numerical_integral():
    atm = rc.AtmC14(np.array([0.0, 10.0, 20.0]), np.array([1.0, 2.0, 3.0]), tail_fm=4.0)
    ages = np.linspace(0.0, 2000.0, 2_000_001)
    for alpha in [0.01, 0.1, 1.0]:
        expected = np.trapezoid(atm.R(ages) * np.exp(-alpha * ages), ages)
        np.testing.assert_allclose(atm.laplace(alpha), expected, rtol=1e-4)


def test_R_matches_zero_order_interpolation_of_the_data():
    """The models previously used interp1d(kind='zero', fill_value='extrapolate')."""
    data = pd.read_csv("data/14C_atm_annot.csv")
    interpolator = interp1d(data.years_before_2000, data.R_14C, kind="zero", fill_value="extrapolate")
    ages = np.concatenate([np.linspace(0.0, 60_000.0, 600_001), [1e5, 1e7]])
    np.testing.assert_array_equal(rc.load_atm14c().R(ages), interpolator(ages))


def _constant_atm(value: float = 1.0) -> rc.AtmC14:
    """Build a small synthetic atmospheric lookup with constant fm over age."""
    ages = np.array([0.0, 10.0, 25.0, 80.0], dtype=np.float64)
    fm = np.full_like(ages, value)
    return rc.AtmC14(ages, fm, tail_fm=value)


def _impulse_atm(impulse_age: float, width: float) -> rc.AtmC14:
    """Approximate a unit-area delta impulse as a narrow rectangular pulse."""
    if impulse_age <= 0.0:
        raise ValueError("impulse_age must be positive")
    if width <= 0.0 or width >= 2.0 * impulse_age:
        raise ValueError("width must be positive and less than 2 * impulse_age")

    lo = impulse_age - 0.5 * width
    hi = impulse_age + 0.5 * width
    # Piecewise-constant segments: 0 on [0, lo), 1/width on [lo, hi), and 0 afterward.
    ages = np.array([0.0, lo, hi, hi + max(1.0, width)], dtype=np.float64)
    fm = np.array([0.0, 1.0 / width, 0.0, 0.0], dtype=np.float64)
    return rc.AtmC14(ages, fm, tail_fm=0.0)


def test_laplace_constant_atm_matches_closed_form():
    """Constant atmospheric ratio should integrate to value / alpha."""
    atm = _constant_atm(value=1.7)
    alpha = 0.023
    expected = 1.7 / alpha
    actual = atm.laplace(alpha)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def test_laplace_impulse_atm_matches_exponential_kernel():
    """A narrow impulse should yield the expected exponential-kernel response."""
    impulse_age = 60.0
    width = 0.2
    alpha = 0.037
    atm = _impulse_atm(impulse_age=impulse_age, width=width)

    # Exact response for a unit-area box pulse on [lo, hi].
    lo = impulse_age - 0.5 * width
    hi = impulse_age + 0.5 * width
    expected = (np.exp(-alpha * lo) - np.exp(-alpha * hi)) / (alpha * width)

    actual = atm.laplace(alpha)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)

    # Narrow-pulse limit recovers the simple impulse form exp(-alpha * impulse_age).
    np.testing.assert_allclose(actual, np.exp(-alpha * impulse_age), rtol=1e-5, atol=0.0)
