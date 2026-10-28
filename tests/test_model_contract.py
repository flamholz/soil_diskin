"""Contract tests: every continuum model must satisfy AbstractDiskinModel's interface.

Each model is checked against numerical integrals of its own survival function
s(t), independent of how the class computes T, A, cdfA or radiocarbon:
  * params_valid() is implemented and True for the test parameters
  * s(0) = 1, s is non-increasing, and s accepts scalars and arrays alike
  * T = integral of s(t) dt and A = integral of a * pA(a) da, with pA = s/T
  * cdfA matches the integral of pA and tends to 1
  * calc_radiocarbon_ratio_ss() matches integral of pA(a) R(a) exp(-lambda a) da;
    for models with a rate distribution this checks the rate-space calculation
    against the age-space one
"""
import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, trapezoid

from soil_diskin import constants
from soil_diskin.constants import GAMMA
from soil_diskin.continuum_models import (
    GammaDisKin, GeneralPowerLawDisKin, LogUniformDisKin, LognormalDisKin,
    PowerLawDisKin, SoftCutoffPowerLawDisKin, WeibullDisKin)

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
}

# Dense linear grid where the atmosphere varies year to year (bomb peak), then
# log-spaced out to ages where even the broad lognormal has no stock left.
AGES = np.union1d(np.linspace(0., 300., 3001), np.geomspace(1e-10, 1e14, 3000))
SPOT_AGES = np.array([0., 1., 10., 100., 1e3, 1e4])


def survival(model, ages):
    """s on an age grid, element by element so array support is not assumed here."""
    return np.array([float(model.s(age)) for age in ages])


_survival_cache = {}


def model_and_survival(name):
    if name not in _survival_cache:
        model = CASES[name]()
        _survival_cache[name] = model, survival(model, AGES)
    return _survival_cache[name]


@pytest.mark.parametrize('name', CASES)
def test_params_valid(name):
    assert CASES[name]().params_valid()


@pytest.mark.parametrize('name', CASES)
def test_survival_starts_at_one_and_decreases(name):
    _, s = model_and_survival(name)
    assert s[0] == pytest.approx(1., abs=1e-6)
    assert np.all(np.diff(s) <= 1e-9)


@pytest.mark.parametrize('name', CASES)
def test_survival_accepts_scalars_and_arrays(name):
    model = CASES[name]()
    array = np.asarray(model.s(SPOT_AGES), dtype=float)
    assert array.shape == SPOT_AGES.shape
    np.testing.assert_allclose(array, [model.s(age) for age in SPOT_AGES], rtol=1e-10)


@pytest.mark.parametrize('name', CASES)
def test_transit_time_and_mean_age_match_survival(name):
    """T = integral of s; A = integral of a * pA(a), with pA = s/T."""
    model, s = model_and_survival(name)
    assert trapezoid(s, AGES) == pytest.approx(model.T, rel=1e-3)
    assert trapezoid(AGES * s, AGES) / model.T == pytest.approx(model.A, rel=1e-3)


@pytest.mark.parametrize('name', CASES)
def test_cdf_matches_integrated_density(name):
    model, s = model_and_survival(name)
    cumulative = cumulative_trapezoid(s / model.T, AGES, initial=0.)
    expected = np.interp(SPOT_AGES, AGES, cumulative)
    np.testing.assert_allclose([model.cdfA(age) for age in SPOT_AGES], expected, atol=1e-3)
    assert model.cdfA(1e14) == pytest.approx(1., abs=1e-4)


@pytest.mark.parametrize('name', CASES)
def test_radiocarbon_matches_integrated_density(name):
    """The default method integrates pA with epsabs = 1e-3, hence the tolerance."""
    model, s = model_and_survival(name)
    weights = constants.INTERP_R_14C(AGES) * np.exp(-constants.LAMBDA_14C * AGES)
    expected = trapezoid(s / model.T * weights, AGES)
    fm, _ = model.calc_radiocarbon_ratio_ss()
    assert 0 < fm < 2
    assert fm == pytest.approx(expected, abs=2e-3)
