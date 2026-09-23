"""Public forward-model seam: mass balance and independent limiting cases."""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.integrate import quad_vec
from scipy.special import roots_hermitenorm

from soil_diskin.layered_lognormal import input_weights, transport_matrix, LayeredLognormal
from soil_diskin.lognormal import diskin_C_of_t, lognormal_radiocarbon
from soil_diskin.radiocarbon_utils import AtmC14


@pytest.mark.parametrize('diffusion,velocity', [(0, 0), (1, 0), (0, 0.3), (1, 0.3)])
def test_closed_transport_preserves_mass_and_nonnegative_carbon(diffusion, velocity):
    operator = transport_matrix(diffusion, velocity)
    carbon = np.arange(1.0, 11.0)
    after = expm(operator * 100) @ carbon
    np.testing.assert_allclose(after.sum(), 55, rtol=1e-13)
    assert np.all(after >= 0)
    np.testing.assert_allclose(operator.sum(axis=0), 0, atol=1e-17)
    if diffusion == 0 and velocity > 0:
        # Only the first layer loses carbon; the closed bottom cannot export it.
        np.testing.assert_allclose(after[0], np.exp(-3), rtol=1e-13)
        np.testing.assert_allclose(operator @ np.eye(10)[:, -1], 0)


@pytest.mark.parametrize('depth', [0.01, 20, 1e12])
def test_input_allocation_integrates_exponential_and_preserves_npp(depth):
    weights = input_weights(depth)
    np.testing.assert_allclose(weights.sum(), 1, rtol=1e-14)
    assert np.all(weights >= 0)
    assert np.all(np.diff(weights) <= 0)
    if depth == 20:
        np.testing.assert_allclose(weights[1] / weights[0], np.exp(-0.5))
    if depth == 1e12:
        np.testing.assert_allclose(weights, 0.1, rtol=1e-9)


def test_uncoupled_profile_matches_existing_lognormal_model():
    atm = AtmC14(np.array([0., 20., 100., 1000.]),
                 np.array([1.1, 1.4, 0.95, 1.]), 1.02)
    model = LayeredLognormal(0, 0, 30, atm)
    mu = np.linspace(-2, 1, 10)
    sigma = np.linspace(0.4, 2.5, 10)
    times = np.array([0., 1., 100.])
    prediction = model.predict(mu, sigma, npp=0.6, times=times)
    inputs = 0.6 * input_weights(30)
    tau = np.exp(-mu + sigma**2 / 2)
    expected_fm = [lognormal_radiocarbon(atm, t, t*np.exp(s*s), rtol=1e-9)
                   for t, s in zip(tau, sigma)]
    expected_new = np.array([diskin_C_of_t(times, m, s, input_=i) / (i*t)
                             for m, s, i, t in zip(mu, sigma, inputs, tau)]).T
    np.testing.assert_allclose(prediction.stocks, inputs*tau, rtol=1e-8)
    np.testing.assert_allclose(prediction.fm, expected_fm, rtol=1e-7)
    np.testing.assert_allclose(prediction.fnew, expected_new, rtol=1e-7, atol=1e-13)


@pytest.mark.parametrize('diffusion,velocity', [(0, 0.2), (0.8, 0), (0.8, 0.2)])
def test_coupled_stocks_and_isotope_match_independent_adaptive_solve(diffusion, velocity):
    atm = AtmC14(np.array([0., 1.]), np.array([1.2, 1.2]), 1.2)
    model = LayeredLognormal(diffusion, velocity, 25, atm)
    mu, sigma = np.linspace(-3, -1, 10), np.linspace(0.3, 0.8, 10)
    pred = model.predict(mu, sigma, 0.5)
    inputs = 0.5*input_weights(25)

    def integrand(u):
        source = inputs*np.exp(-0.5*((u-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        carbon = np.linalg.solve(np.exp(u)*np.eye(10)-model.transport, source)
        radio = np.linalg.solve((np.exp(u)+1/8267)*np.eye(10)-model.transport, 1.2*source)
        return np.r_[carbon, radio]

    expected, _ = quad_vec(integrand, -15, 10, epsabs=1e-11, epsrel=1e-10)
    np.testing.assert_allclose(pred.stocks, expected[:10], rtol=1e-8)
    np.testing.assert_allclose(pred.fm, expected[10:]/expected[:10], rtol=1e-8)


def test_new_carbon_matches_age_integral_and_retains_label_during_transport():
    atm = AtmC14(np.array([0.]), np.array([1.]), 1.)
    model = LayeredLognormal(0.7, 0.2, 20, atm)
    mu, sigma = np.linspace(-3, -1, 10), np.full(10, 0.5)
    times = np.array([0., 1e-8, 1., 30., 1e6])
    pred = model.predict(mu, sigma, 0.5, times)
    nodes, weights = roots_hermitenorm(128)
    rates = np.exp(mu + sigma*nodes[:, None])
    inputs = 0.5*input_weights(20)

    def new_by_age(age):
        survivors = (weights[:, None]*np.exp(-rates*age)).sum(axis=0)/np.sqrt(2*np.pi)
        return expm(model.transport*age) @ (inputs*survivors)

    expected, _ = quad_vec(new_by_age, 0, 30, epsabs=1e-12, epsrel=1e-10)
    np.testing.assert_allclose(pred.fnew[3], expected/pred.stocks, rtol=1e-8)
    np.testing.assert_allclose(pred.fnew[1], times[1]*inputs/pred.stocks, rtol=1e-7)
    np.testing.assert_allclose(pred.fnew[0], 0)
    np.testing.assert_allclose(pred.fnew[-1], 1, rtol=1e-7)
    assert np.all(pred.fnew >= -1e-13)
    assert np.all(pred.fnew <= 1 + 1e-12)
    assert np.all(np.diff(pred.fnew, axis=0) >= -1e-12)


def test_broad_slow_rate_tail_preserves_column_stock_and_quadrature_converges():
    atm = AtmC14(np.array([0., 5., 30.]), np.array([1.1, 1.4, 1.]), 0.95)
    model = LayeredLognormal(1., 0.3, 30, atm)
    fine = LayeredLognormal(1., 0.3, 30, atm, log_rate_step=0.025)
    mu = np.linspace(-15, 10, 10)
    sigma = np.array([5., 4., 3., 2., 1., 0.5, 0.1, 0.05, 2., 5.])
    pred, refined = [m.predict(mu, sigma, 0.5, (10.,)) for m in (model, fine)]
    # Class-preserving transport cannot change the column total at steady state.
    total = (0.5*input_weights(30)*np.exp(-mu+sigma*sigma/2)).sum()
    np.testing.assert_allclose(pred.stocks.sum(), total, rtol=1e-8)
    for coarse, accurate in [(pred.stocks, refined.stocks), (pred.fm, refined.fm),
                              (pred.fnew, refined.fnew)]:
        np.testing.assert_allclose(coarse, accurate, rtol=1e-7, atol=1e-14)


def test_coupled_radiocarbon_matches_forward_propagation_from_atmospheric_tail():
    atm = AtmC14(np.array([0., 10., 100.]), np.array([1.1, 1.7, 1.]), 0.9)
    model = LayeredLognormal(0.4, 0.1, 30, atm)
    mu, sigma = np.linspace(-3, -1, 10), np.full(10, 0.3)
    inputs = 0.5*input_weights(30)
    predicted = model.predict(mu, sigma, 0.5)

    def forward_tracer(u):
        source = inputs*np.exp(-0.5*((u-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
        loss = (np.exp(u)+1/8267)*np.eye(10)-model.transport
        unit_equilibrium = np.linalg.solve(loss, source)
        tracer = 0.9*unit_equilibrium
        for dt, ratio in [(90, 1.7), (10, 1.1)]:
            equilibrium = ratio*unit_equilibrium
            tracer = expm(-loss*dt) @ (tracer-equilibrium)+equilibrium
        return tracer

    radio, _ = quad_vec(forward_tracer, -8, 4, epsabs=1e-11, epsrel=1e-10)
    np.testing.assert_allclose(predicted.fm, radio/predicted.stocks, rtol=1e-8)


def test_forward_jacobian_matches_finite_differences():
    atm = AtmC14(np.array([0., 10.]), np.array([1.2, 1.]), 1.)
    model = LayeredLognormal(1, 0.2, 25, atm)
    parameters = np.r_[np.linspace(-3, 0, 10), np.linspace(0.4, 2., 10)]
    _, jacobian = model.observables_and_jacobian(parameters, 0.6)
    numerical = np.empty_like(jacobian)
    for column in range(20):
        offset = np.eye(20)[column]*1e-5
        plus = model.predict((parameters+offset)[:10], (parameters+offset)[10:], 0.6)
        minus = model.predict((parameters-offset)[:10], (parameters-offset)[10:], 0.6)
        numerical[:, column] = (np.r_[plus.stocks, plus.fm]-np.r_[minus.stocks, minus.fm])/2e-5
    np.testing.assert_allclose(jacobian, numerical, rtol=1e-4, atol=1e-9)
