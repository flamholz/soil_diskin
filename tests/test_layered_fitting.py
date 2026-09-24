"""Each layer is a separate two-parameter fit, without f_new targets."""
import numpy as np

from soil_diskin.layered_lognormal import LayerLognormal, fit_layer
from soil_diskin.radiocarbon_utils import AtmC14


def test_fit_recovers_layer_observables_reproducibly():
    model = LayerLognormal(AtmC14(np.array([0.]), np.array([1.]), 1.))
    truth = model.predict(-1., 2.5, .04)
    fits = [fit_layer(model, truth.stock, truth.fm, .04) for _ in range(2)]
    for fit in fits:
        assert len(fit) == 3
        assert fit[0].success
        assert fit[0].objective < 1e-12
        assert fit[0].jacobian_rank == 2
        np.testing.assert_allclose(fit[0].stock_pred_kg_m2, truth.stock, rtol=1e-7)
        np.testing.assert_allclose(fit[0].fm_pred, truth.fm, atol=1e-7)
    assert fits[0] == fits[1]


def test_alternative_solutions_and_failed_starts_are_retained():
    model = LayerLognormal(AtmC14(np.array([0.]), np.array([0.]), 0.))
    fits = fit_layer(model, 1., 0., .02)
    near = [f for f in fits if f.near_best]
    assert len(near) == 3
    assert all(f.jacobian_rank == 1 for f in near)
    predictions = [model.predict(f.mu, f.sigma, .02, (20.,)).fnew[0] for f in near]
    assert np.ptp(predictions) > .01
    informative = LayerLognormal(AtmC14(np.array([0.]), np.array([1.]), 1.))
    limited = fit_layer(informative, 1., .8, .02, max_nfev=1)
    assert len(limited) == 3
    assert any(not f.success for f in limited)
    assert all(np.isfinite(f.objective) for f in limited)
