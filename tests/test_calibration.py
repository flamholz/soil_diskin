"""Tests for soil_diskin.calibration, the calibration shared by all models."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from soil_diskin.calibration import (FIT_COLUMNS, ModelSpec, calibrate, calibrate_sweep,
                                     fit_observation, lognormal_spec)
from soil_diskin.continuum_models import LognormalDisKin
from soil_diskin.radiocarbon_utils import default_atmosphere

# Slow sites (fm < 1), where turnover and fm determine (mu, sigma) uniquely.
TRUE_PARAMS = [(-5., 1.5), (-3., .8), (-7., 2.2)]


def observation(mu, sigma):
    model = LognormalDisKin(mu, sigma)
    return model.T, model.calc_radiocarbon_ratio_ss()[0]


def fixed_sigma_spec(sigma):
    """A one-parameter spec (mu free, sigma fixed), for sweeps over sigma."""
    return ModelSpec(model_class=LognormalDisKin, param_names=('mu',), lower=(-15.,), upper=(10.,),
                     starts=lambda turnover, fm_obs: [[sigma**2 / 2 - np.log(turnover)]],
                     fixed={'sigma': sigma})


@pytest.fixture(scope='module')
def sites():
    rows = [dict(zip(['turnover', 'fm_obs'], observation(*params))) for params in TRUE_PARAMS]
    sites = pd.DataFrame(rows, index=[3, 7, 11])  # Non-default index: results align by label.
    sites['turnover_q05'] = [np.nan, .8 * sites.turnover[7], np.nan]
    sites['turnover_q95'] = [np.nan, 1.2 * sites.turnover[7], np.nan]
    return sites


@pytest.mark.parametrize('mu, sigma', TRUE_PARAMS)
def test_lognormal_recovers_known_parameters(mu, sigma):
    turnover, fm_obs = observation(mu, sigma)
    fit = fit_observation(lognormal_spec(), turnover, fm_obs)
    assert fit['calibration_status'] == 'calibrated'
    assert fit['mu'] == pytest.approx(mu, abs=1e-6)
    assert fit['sigma'] == pytest.approx(sigma, abs=1e-6)
    assert fit['model_T'] == pytest.approx(turnover, rel=1e-10)
    assert fit['model_fm'] == pytest.approx(fm_obs, abs=1e-10)
    assert list(fit) == ['mu', 'sigma'] + FIT_COLUMNS


def test_calibrate_keeps_rows_it_cannot_fit():
    sites = pd.DataFrame({'turnover': [observation(-5., 1.5)[0], np.nan, 100.],
                          'fm_obs': [observation(-5., 1.5)[1], .9, 0.]})
    fits = calibrate(lognormal_spec(), sites)
    assert fits.index.equals(sites.index)
    assert fits.calibration_status.tolist() == ['calibrated', 'invalid_input', 'invalid_input']
    assert fits.loc[[1, 2], ['mu', 'sigma', 'model_T']].isna().all().all()
    # Scenario columns missing from the input are reported as invalid, not dropped.
    assert (fits.calibration_status_05 == 'invalid_input').all()
    assert fits.mu_95.isna().all()


def test_scenarios_are_fit_where_given(sites):
    fits = calibrate(lognormal_spec(), sites)
    assert fits.index.equals(sites.index)
    assert (fits.calibration_status == 'calibrated').all()
    assert fits.calibration_status_05.tolist() == ['invalid_input', 'calibrated', 'invalid_input']
    assert fits.model_T_05[7] == pytest.approx(sites.turnover_q05[7], rel=1e-10)
    assert fits.model_T_95[7] == pytest.approx(sites.turnover_q95[7], rel=1e-10)


def test_parallel_results_match_serial(sites):
    serial = calibrate(lognormal_spec(), sites, n_jobs=1, warm_start_scenarios=True)
    parallel = calibrate(lognormal_spec(), sites, n_jobs=2, warm_start_scenarios=True)
    pd.testing.assert_frame_equal(serial, parallel)


def test_warm_start_is_tried_first():
    """With no default starts, only the warm start can produce a fit."""
    turnover, fm_obs = observation(-5., 1.5)
    spec = replace(lognormal_spec(), starts=lambda turnover, fm_obs: [])
    assert fit_observation(spec, turnover, fm_obs)['calibration_status'] == 'fit_failed'
    fit = fit_observation(spec, turnover, fm_obs, warm_start={'mu': -4.9, 'sigma': 1.4})
    assert fit['calibration_status'] == 'calibrated'
    assert fit['sigma'] == pytest.approx(1.5, abs=1e-6)


def test_sweep_is_long_format_and_independent_of_n_jobs(sites):
    values = [.8, 1.5]
    sweep = calibrate_sweep(fixed_sigma_spec, values, sites, name='sigma_fixed', n_jobs=1)
    assert list(sweep.columns[:3]) == ['sigma_fixed', 'site_index', 'mu']
    assert sweep[['sigma_fixed', 'site_index']].values.tolist() == [
        [v, i] for v in values for i in sites.index]
    np.testing.assert_array_equal(sweep.sigma, sweep.sigma_fixed)
    # Site 7 was generated with sigma = 0.8, so that fit is exact.
    exact = sweep[(sweep.sigma_fixed == .8) & (sweep.site_index == 7)].iloc[0]
    assert exact.calibration_status == 'calibrated'
    assert exact.mu == pytest.approx(-3., abs=1e-6)
    pd.testing.assert_frame_equal(
        sweep, calibrate_sweep(fixed_sigma_spec, values, sites, name='sigma_fixed', n_jobs=2))


def test_sweep_over_a_fixed_parameter_keeps_one_column(sites):
    """Sweeping a fixed model parameter moves its column to the front, not twice."""
    sweep = calibrate_sweep(fixed_sigma_spec, [.8], sites, name='sigma', n_jobs=1)
    assert list(sweep.columns[:3]) == ['sigma', 'site_index', 'mu']
    assert list(sweep.columns).count('sigma') == 1


# Sampling years other than 2000 (e.g. ISRaD profiles): calibrate(..., year_column=...).

def test_atmosphere_reference_year_shifts_ages():
    """Ages count back from the reference year: carbon of age a sampled in 1990 was fixed
    in 1990 - a, i.e. at age a + 10 counted from 2000. The record ends in 2013."""
    from soil_diskin import constants
    from soil_diskin.radiocarbon_utils import load_atm14c
    base = load_atm14c(constants.C14_DATA_PATH)
    same = load_atm14c(constants.C14_DATA_PATH, reference_year=2000)
    np.testing.assert_array_equal(same.ages, base.ages)
    np.testing.assert_array_equal(same.R(base.ages), base.R(base.ages))
    ages = np.array([0., 5., 30., 500.])
    np.testing.assert_allclose(load_atm14c(constants.C14_DATA_PATH, reference_year=1990).R(ages), base.R(ages + 10))
    with pytest.raises(ValueError, match='ends'):
        load_atm14c(constants.C14_DATA_PATH, reference_year=2020)


def test_year_column_of_2000_matches_the_default(sites):
    default = calibrate(lognormal_spec(), sites, n_jobs=1)
    by_year = calibrate(lognormal_spec(), sites.assign(sampling_year=2000), n_jobs=1, year_column='sampling_year')
    pd.testing.assert_frame_equal(default, by_year)


def test_fit_uses_each_rows_sampling_year():
    """An observation made in 1980 is recovered when fit with its own year's atmosphere, and not
    with the default (2000) one."""
    from soil_diskin import constants
    from soil_diskin.radiocarbon_utils import load_atm14c
    model = LognormalDisKin(-3., .8, atm=load_atm14c(constants.C14_DATA_PATH, reference_year=1980))
    row = pd.DataFrame({'turnover': [model.T], 'fm_obs': [model.calc_radiocarbon_ratio_ss()[0]], 'sampling_year': [1980]})
    by_year = calibrate(lognormal_spec(), row, n_jobs=1, year_column='sampling_year')
    assert (by_year.mu[0], by_year.sigma[0]) == pytest.approx((-3., .8), abs=1e-6)
    default = calibrate(lognormal_spec(), row, n_jobs=1)
    assert abs(default.mu[0] - by_year.mu[0]) > 1e-3


def test_sampling_after_the_record_has_no_atmosphere(sites):
    """Rows sampled after the atmospheric record ends keep their row, with NaN parameters."""
    fits = calibrate(lognormal_spec(), sites.assign(sampling_year=[2000, 2020, 1990]), n_jobs=1,
                     year_column='sampling_year')
    assert fits.calibration_status.tolist()[1] == 'no_atmosphere'
    assert fits.calibration_status_05.tolist()[1] == 'no_atmosphere'
    assert np.isnan(fits.mu.iloc[1]) and np.isfinite(fits.mu.iloc[[0, 2]]).all()
