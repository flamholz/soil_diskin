import numpy as np

from soil_diskin.constants import LAMBDA_14C
from soil_diskin.continuum_models import LognormalDisKin
from soil_diskin.lognormal import lognormal_radiocarbon, scan_ages
from soil_diskin.radiocarbon_utils import AtmC14


def _constant_atm(value: float = 1.0) -> AtmC14:
    """Build a small synthetic atmospheric lookup with constant fm over age."""
    ages = np.array([0.0, 10.0, 25.0, 80.0], dtype=np.float64)
    fm = np.full_like(ages, value)
    return AtmC14(ages, fm, tail_fm=value)


def _impulse_atm(impulse_age: float, width: float) -> AtmC14:
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
    return AtmC14(ages, fm, tail_fm=0.0)


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


def test_scan_ages_matches_scalar_lognormal_calls():
    """Vectorized age scan should match repeated scalar model evaluations."""
    atm = _constant_atm(value=1.0)
    tau = 10.0
    ages = np.array([10.5, 15.0, 25.0, 40.0], dtype=np.float64)

    scanned = scan_ages(atm, tau, ages, rtol=1e-6)
    manual = np.array([lognormal_radiocarbon(atm, tau, a, rtol=1e-6) for a in ages])

    np.testing.assert_allclose(scanned, manual, rtol=1e-11, atol=0.0)


def test_lognormal_radiocarbon_repeated_calls_are_consistent():
    """With fixed internal u-bounds, repeated evaluations should agree."""
    atm = _constant_atm(value=1.0)
    tau = 10.0
    age = 25.0

    v1 = lognormal_radiocarbon(atm, tau, age, rtol=1e-8)
    v2 = lognormal_radiocarbon(atm, tau, age, rtol=1e-8)

    np.testing.assert_allclose(v2, v1, rtol=1e-12, atol=0.0)


def test_lognormal_radiocarbon_narrow_sigma_matches_single_rate_limit():
    """As sigma -> 0, the full model should match the single-rate closed form."""
    atm = _constant_atm(value=1.0)

    # Pick age/tau very close so sigma is tiny; large tau keeps k=exp(mu) very small.
    tau = 1.0e6
    age = tau * (1.0 + 1.0e-8)

    sigma = np.sqrt(np.log(age / tau))
    mu = -np.log(np.sqrt(tau ** 3 / age))
    assert sigma < 2.0e-4

    k0 = np.exp(mu)
    expected = k0 / (LAMBDA_14C + k0)

    actual = lognormal_radiocarbon(atm, tau=tau, age=age, rtol=1e-8)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=0.0)


def test_stock_younger_than_is_zero_at_zero_and_increasing():
    """The stock of a pool started empty starts at zero and grows monotonically."""
    ts = np.array([0.0, 0.1, 1.0, 10.0, 100.0], dtype=np.float64)
    C = LognormalDisKin(0.25, 0.5).stock_younger_than(ts)

    np.testing.assert_allclose(C[0], 0.0, atol=1e-14)
    assert np.all(C >= 0.0)
    assert np.all(np.diff(C) >= 0.0)


def test_stock_younger_than_approaches_steady_state():
    """At long times the stock (per unit input) approaches the turnover time T."""
    model = LognormalDisKin(0.1, 0.45)
    ts = np.array([0.5, 5.0, 50.0, 1000.0], dtype=np.float64)

    C = model.stock_younger_than(ts)

    assert C[-1] > C[-2]
    np.testing.assert_allclose(C[-1], model.T, rtol=2e-3, atol=0.0)


def test_stock_younger_than_accepts_scalars_and_arrays_and_gives_cdfA():
    """Scalar and array calls agree, and the age CDF is the stock over T."""
    model = LognormalDisKin.from_age_and_transit_time(30.0, 8.0)
    ts = np.logspace(-1, np.log10(200.0), 32)

    C = model.stock_younger_than(ts)

    assert C.shape == ts.shape
    np.testing.assert_array_equal(C, [model.stock_younger_than(t) for t in ts])
    np.testing.assert_allclose(model.cdfA(ts), C / model.T, rtol=1e-15)
    assert isinstance(model.cdfA(5.0), float)
