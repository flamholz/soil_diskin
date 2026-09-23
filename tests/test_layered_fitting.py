"""Public fitting seam: recover observables without using evaluation f_new."""

import numpy as np

from soil_diskin.layered_lognormal import LayeredLognormal
from soil_diskin.layered_fitting import FitSettings, fit_profile
from soil_diskin.radiocarbon_utils import AtmC14


def test_synthetic_profile_fit_recovers_stocks_and_radiocarbon_reproducibly():
    atmosphere = AtmC14(np.array([0.]), np.array([1.]), 1.)
    model = LayeredLognormal(0.2, 0.02, 30, atmosphere)
    mu = np.linspace(-4, -2, 10)
    sigma = np.linspace(0.5, 1.3, 10)
    observations = model.predict(mu, sigma, 0.6)
    initial = np.r_[mu + 0.1, sigma + 0.1]
    settings = FitSettings(n_starts=2, seed=17, max_nfev=200)
    fits = [fit_profile(model, observations.stocks, observations.fm, 0.6,
                        settings=settings, initial_parameters=initial) for _ in range(2)]
    for fit in fits:
        assert fit.best.success
        assert fit.best.objective < 1e-10
        np.testing.assert_allclose(fit.best.prediction.stocks, observations.stocks, rtol=1e-6)
        np.testing.assert_allclose(fit.best.prediction.fm, observations.fm, atol=1e-7)
        assert len(fit.attempts) == 2
        assert fit.best.jacobian_rank > 0
    np.testing.assert_allclose(fits[0].best.mu, fits[1].best.mu, rtol=0, atol=0)
    # Prediction remains a separate operation; the fitter has no f_new target.
    prediction = model.predict(fits[0].best.mu, fits[0].best.sigma, 0.6, (20.,))
    assert prediction.fnew.shape == (1, 10)


def test_equally_good_nonidentifiable_solutions_keep_different_new_carbon_predictions():
    # With no atmospheric tracer, stocks alone leave a continuum of solutions.
    atmosphere = AtmC14(np.array([0.]), np.array([0.]), 0.)
    model = LayeredLognormal(0, 0, 30, atmosphere)
    observations = model.predict(np.full(10, -1.), np.full(10, 2.5), 0.5)
    fit = fit_profile(model, observations.stocks, observations.fm, 0.5,
                      settings=FitSettings(n_starts=2, seed=11))
    near = [c for c in fit.candidates if c.near_best]
    assert len(near) == 2
    assert all(c.jacobian_rank == 10 for c in near)
    new = [model.predict(c.mu, c.sigma, 0.5, (20.,)).fnew for c in near]
    assert np.max(np.abs(new[0]-new[1])) > 1e-3


def test_evaluation_limit_preserves_unsuccessful_candidate_and_all_attempts():
    atm = AtmC14(np.array([0.]), np.array([1.]), 1.)
    model = LayeredLognormal(0.2, 0.01, 30, atm)
    observed = model.predict(np.full(10, -3.), np.ones(10), 0.5)
    fit = fit_profile(model, observed.stocks, observed.fm, 0.5,
                      settings=FitSettings(n_starts=2, max_nfev=1))
    assert len(fit.attempts) == 2
    assert fit.candidates
    assert not fit.best.success
    assert np.isfinite(fit.best.objective)
    assert not any(c.near_best for c in fit.candidates)
