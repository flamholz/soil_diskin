"""Contract tests: every continuum model must satisfy AbstractDiskinModel's interface.

Each model is checked against numerical integrals of its own survival function
s(t), independent of how the class computes T, A, cdfA or radiocarbon:
  * params_valid() is implemented and True for the test parameters
  * s(0) = 1, s is non-increasing, and s accepts scalars and arrays alike
  * T = integral of s(t) dt and A = integral of a * pA(a) da, with pA = s/T
  * cdfA matches the integral of pA and tends to 1
  * calc_radiocarbon_ratio_ss() matches integral of pA(a) R(a) exp(-lambda a) da

Known violations are marked xfail(strict=True), so fixing one makes its test
fail until the mark is removed. They document the lognormal gaps that the
calibration refactor is meant to close by merging the two lognormal classes.
"""
import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, trapezoid

from soil_diskin import constants
from soil_diskin.constants import GAMMA
from soil_diskin.continuum_models import (
    GammaDisKin, GeneralPowerLawDisKin, LogUniformDisKin, LognormalDisKin, LognormalDisKinFast,
    PowerLawDisKin, SoftCutoffPowerLawDisKin, WeibullDisKin)
from soil_diskin.radiocarbon_utils import load_atm14c

# Representative soil-like parameters: turnover times of years to millennia.
# The broad lognormal (sigma ~ 2.6) is the site-0 fit, the hardest case numerically.
CASES = {
    'gamma': lambda: GammaDisKin(3., .1),
    'powerlaw': lambda: PowerLawDisKin(1., 1e4),
    'general_powerlaw-e^-gamma': lambda: GeneralPowerLawDisKin(1., 1e4, np.exp(-GAMMA)),
    'general_powerlaw-0.4': lambda: GeneralPowerLawDisKin(.01, 2000., .4),
    'soft_cutoff-0.4': lambda: SoftCutoffPowerLawDisKin(.01, 2000., .4),
    'soft_cutoff-1.5': lambda: SoftCutoffPowerLawDisKin(.5, 8000., 1.5),
    'loguniform': lambda: LogUniformDisKin(1e-4, 10.),
    'weibull-0.5': lambda: WeibullDisKin(.05, .5),
    'weibull-1.0': lambda: WeibullDisKin(.01, 1.),
    'lognormal-narrow': lambda: LognormalDisKin(-3., 1.),
    'lognormal-broad': lambda: LognormalDisKin(.68, 2.56),
    'lognormal_fast-narrow': lambda: LognormalDisKinFast(-3., 1., load_atm14c()),
    'lognormal_fast-broad': lambda: LognormalDisKinFast(.68, 2.56, load_atm14c()),
}

FAST_INIT = 'LognormalDisKinFast skips AbstractDiskinModel.__init__ and has no params_valid'
FAST_SCALAR = 'LognormalDisKinFast.s only accepts 1-D arrays'
FAST_TRUNCATED = ('LognormalDisKinFast.s truncates p(k) at its 1e-3 quantiles, dropping slow '
                  'rates that dominate the 1/k-weighted stock')
BROKEN_S = ('LognormalDisKin.s passes (t, k) to quad in the wrong order, so it integrates '
            'over t with k fixed at the age; s(0) = 0 for any sigma')
SCALAR_S = 'LognormalDisKin.s uses scalar quad and does not accept arrays'
# LognormalDisKin's radiocarbon check passes only because the default method
# integrates the same broken s; the moments check is the one that catches it.
KNOWN_FAILURES = {
    'params_valid': {'lognormal_fast-narrow': FAST_INIT, 'lognormal_fast-broad': FAST_INIT},
    'survival': {'lognormal-narrow': BROKEN_S, 'lognormal-broad': BROKEN_S},
    'vectorized': {'lognormal-narrow': SCALAR_S, 'lognormal-broad': SCALAR_S,
                   'lognormal_fast-narrow': FAST_SCALAR, 'lognormal_fast-broad': FAST_SCALAR},
    'moments': {'lognormal-narrow': BROKEN_S, 'lognormal-broad': BROKEN_S,
                'lognormal_fast-narrow': FAST_TRUNCATED, 'lognormal_fast-broad': FAST_TRUNCATED},
    'cdf': {'lognormal-narrow': BROKEN_S, 'lognormal-broad': BROKEN_S,
            'lognormal_fast-narrow': FAST_SCALAR, 'lognormal_fast-broad': FAST_SCALAR},
    'radiocarbon': {'lognormal_fast-narrow': FAST_TRUNCATED, 'lognormal_fast-broad': FAST_TRUNCATED},
}

# Dense linear grid where the atmosphere varies year to year (bomb peak), then
# log-spaced out to ages where even the broad lognormal has no stock left.
AGES = np.union1d(np.linspace(0., 300., 3001), np.geomspace(1e-10, 1e14, 3000))
SPOT_AGES = np.array([0., 1., 10., 100., 1e3, 1e4])


def cases(check):
    """Parametrize over CASES, marking this check's known failures as strict xfails."""
    known = KNOWN_FAILURES.get(check, {})
    return [pytest.param(name, id=name, marks=[pytest.mark.xfail(strict=True, reason=known[name])]
                         if name in known else []) for name in CASES]


def survival(model, ages):
    """s on an age grid, element by element so array support is not assumed here."""
    return np.array([float(np.squeeze(model.s(np.atleast_1d(age))))
                     if isinstance(model, LognormalDisKinFast) else float(model.s(age))
                     for age in ages])


_survival_cache = {}


def model_and_survival(name):
    if name not in _survival_cache:
        model = CASES[name]()
        _survival_cache[name] = model, survival(model, AGES)
    return _survival_cache[name]


@pytest.mark.parametrize('name', cases('params_valid'))
def test_params_valid(name):
    assert CASES[name]().params_valid()


@pytest.mark.parametrize('name', cases('survival'))
def test_survival_starts_at_one_and_decreases(name):
    _, s = model_and_survival(name)
    assert s[0] == pytest.approx(1., abs=1e-6)
    assert np.all(np.diff(s) <= 1e-9)


@pytest.mark.parametrize('name', cases('vectorized'))
def test_survival_accepts_scalars_and_arrays(name):
    model = CASES[name]()
    array = np.asarray(model.s(SPOT_AGES), dtype=float)
    assert array.shape == SPOT_AGES.shape
    np.testing.assert_allclose(array, [model.s(age) for age in SPOT_AGES], rtol=1e-10)


@pytest.mark.parametrize('name', cases('moments'))
def test_transit_time_and_mean_age_match_survival(name):
    """T = integral of s; A = integral of a * pA(a), with pA = s/T."""
    model, s = model_and_survival(name)
    assert trapezoid(s, AGES) == pytest.approx(model.T, rel=1e-3)
    assert trapezoid(AGES * s, AGES) / model.T == pytest.approx(model.A, rel=1e-3)


@pytest.mark.parametrize('name', cases('cdf'))
def test_cdf_matches_integrated_density(name):
    model, s = model_and_survival(name)
    cumulative = cumulative_trapezoid(s / model.T, AGES, initial=0.)
    expected = np.interp(SPOT_AGES, AGES, cumulative)
    np.testing.assert_allclose([model.cdfA(age) for age in SPOT_AGES], expected, atol=1e-3)
    assert model.cdfA(1e14) == pytest.approx(1., abs=1e-4)


@pytest.mark.parametrize('name', cases('radiocarbon'))
def test_radiocarbon_matches_integrated_density(name):
    """The default method integrates pA with epsabs = 1e-3, hence the tolerance."""
    model, s = model_and_survival(name)
    weights = constants.INTERP_R_14C(AGES) * np.exp(-constants.LAMBDA_14C * AGES)
    expected = trapezoid(s / model.T * weights, AGES)
    fm, _ = model.calc_radiocarbon_ratio_ss()
    assert 0 < fm < 2
    assert fm == pytest.approx(expected, abs=2e-3)
