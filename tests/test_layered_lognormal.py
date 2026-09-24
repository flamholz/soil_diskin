"""Independent references for the no-transport forward model."""
import numpy as np
import pytest

from soil_diskin.layered_lognormal import LayerLognormal, input_weights
from soil_diskin.lognormal import diskin_C_of_t, lognormal_radiocarbon
from soil_diskin.radiocarbon_utils import AtmC14


@pytest.mark.parametrize('depth', [0.01, 30, 1e12])
def test_exponential_input_weights_sum_to_one(depth):
    weights = input_weights(depth)
    np.testing.assert_allclose(weights.sum(), 1, rtol=1e-14)
    assert np.all(weights >= 0)
    assert np.all(np.diff(weights) <= 0)


def test_layer_prediction_matches_existing_independent_model():
    atm = AtmC14(np.array([0., 20., 100., 1000.]), np.array([1.1, 1.4, .95, 1.]), 1.02)
    model = LayerLognormal(atm)
    times = (0., 1., 100.)
    for mu, sigma in [(-2., .4), (1., 2.5)]:
        tau = np.exp(-mu+sigma**2/2)
        pred = model.predict(mu, sigma, input_rate=.06, times=times)
        np.testing.assert_allclose(pred.stock, .06*tau, rtol=1e-12)
        np.testing.assert_allclose(pred.fm, lognormal_radiocarbon(atm, tau, tau*np.exp(sigma**2), rtol=1e-9), rtol=1e-7)
        expected = diskin_C_of_t(np.array(times), mu, sigma)/tau
        np.testing.assert_allclose(pred.fnew, expected, rtol=1e-7, atol=1e-13)


def test_broad_bound_adjacent_distributions_and_long_horizons():
    atm = AtmC14(np.array([0.]), np.array([1.]), 1.)
    coarse, fine = LayerLognormal(atm), LayerLognormal(atm, log_rate_step=.025)
    times = (0., 1e-8, 1., 1e12, 1e35)
    for mu, sigma in [(-15., 5.), (10., .05), (-2., 2.)]:
        pred, reference = [m.predict(mu, sigma, .5, times) for m in (coarse, fine)]
        np.testing.assert_allclose(pred.fm, reference.fm, rtol=1e-7)
        np.testing.assert_allclose(pred.fnew, reference.fnew, rtol=1e-7, atol=1e-12)
        assert pred.fnew[0] == 0
        assert np.all(np.diff(pred.fnew) >= 0)
        assert np.all((pred.fnew >= 0) & (pred.fnew <= 1+1e-8))
        np.testing.assert_allclose(pred.fnew[-1], 1, atol=1e-8)


def test_invalid_inputs_fail_clearly():
    atm = AtmC14(np.array([0.]), np.array([1.]), 1.)
    model = LayerLognormal(atm)
    with pytest.raises(ValueError):
        model.predict(0, 0, .5)
    with pytest.raises(ValueError):
        model.predict(0, 1, .5, (-1.,))
    with pytest.raises(ValueError):
        input_weights(0)
    with pytest.raises(ValueError):
        LayerLognormal(atm, log_rate_step=.5)


def test_layer_api_works_outside_repository_with_supplied_atmosphere(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    program = f'''
import sys
sys.path.insert(0, {str(repo)!r})
import numpy as np
from soil_diskin.layered_lognormal import InputAllocation, LayerLognormal
from soil_diskin.radiocarbon_utils import AtmC14
model = LayerLognormal(AtmC14(np.array([0.]), np.array([1.]), 1.))
assert model.predict(-1., 2., InputAllocation().layer_inputs(.5)[0], (20.,)).stock > 0
'''
    result = subprocess.run([sys.executable, '-c', program], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
