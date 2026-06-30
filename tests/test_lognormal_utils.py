import numpy as np

from soil_diskin.lognormal import (
    C14_MEAN_LIFE,
    diskin_C_of_t,
    inner_integral,
    lognormal_radiocarbon,
    run_diskin_fast,
    scan_ages,
)
from soil_diskin.radiocarbon_utils import AtmC14


def _constant_atm(value: float = 1.0) -> AtmC14:
    """Build a small synthetic atmospheric lookup with constant fm over age."""
    ages = np.array([0.0, 10.0, 25.0, 80.0], dtype=np.float64)
    fm = np.full_like(ages, value)
    return AtmC14(ages, fm, mean_R=value)


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
    return AtmC14(ages, fm, mean_R=0.0)


def test_inner_integral_constant_atm_matches_closed_form():
    """Constant atmospheric ratio should integrate to value / alpha."""
    atm = _constant_atm(value=1.7)
    alpha = 0.023
    expected = 1.7 / alpha
    actual = inner_integral(atm, alpha)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def test_inner_integral_impulse_atm_matches_exponential_kernel():
    """A narrow impulse should yield the expected exponential-kernel response."""
    impulse_age = 60.0
    width = 0.2
    alpha = 0.037
    atm = _impulse_atm(impulse_age=impulse_age, width=width)

    # Exact response for a unit-area box pulse on [lo, hi].
    lo = impulse_age - 0.5 * width
    hi = impulse_age + 0.5 * width
    expected = (np.exp(-alpha * lo) - np.exp(-alpha * hi)) / (alpha * width)

    actual = inner_integral(atm, alpha)
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
    lam = 1.0 / C14_MEAN_LIFE
    expected = k0 / (lam + k0)

    actual = lognormal_radiocarbon(atm, tau=tau, age=age, rtol=1e-8)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=0.0)


def test_diskin_c_of_t_is_zero_at_zero_and_increasing():
    """Closed-form concentration starts at zero and grows monotonically."""
    ts = np.array([0.0, 0.1, 1.0, 10.0, 100.0], dtype=np.float64)
    C = diskin_C_of_t(ts, mu=0.25, sigma=0.5, input_=1.3)

    np.testing.assert_allclose(C[0], 0.0, atol=1e-14)
    assert np.all(C >= 0.0)
    assert np.all(np.diff(C) >= 0.0)


def test_diskin_c_of_t_approaches_steady_state():
    """Long-time concentration should approach the analytic steady state."""
    mu = 0.1
    sigma = 0.45
    input_ = 2.0
    ts = np.array([0.5, 5.0, 50.0, 1000.0], dtype=np.float64)

    C = diskin_C_of_t(ts, mu=mu, sigma=sigma, input_=input_)
    expected_steady = input_ * np.exp(-mu + 0.5 * sigma * sigma)

    assert C[-1] > C[-2]
    np.testing.assert_allclose(C[-1], expected_steady, rtol=2e-3, atol=0.0)


def test_run_diskin_fast_grid_and_values_match_direct_eval():
    """Fast wrapper should return the expected grid and direct model values."""
    tau = 8.0
    age = 30.0
    tmax = 200.0
    ts_size = 32
    input_ = 0.7

    ts, C = run_diskin_fast(tau, age, input_=input_, tmax=tmax, ts_size=ts_size, rtol=1e-8)

    assert ts.shape == (ts_size,)
    assert C.shape == (ts_size,)
    np.testing.assert_allclose(ts[0], 0.1, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(ts[-1], tmax, rtol=1e-12, atol=0.0)

    sigma = np.sqrt(np.log(age / tau))
    mu = -np.log(np.sqrt(tau ** 3 / age))
    C_direct = diskin_C_of_t(ts, mu, sigma, input_=input_, rtol=1e-8)
    np.testing.assert_allclose(C, C_direct, rtol=1e-10, atol=0.0)
