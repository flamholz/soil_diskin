"""Tests for notebooks/04_collect_continuum_model_predictions.py.

Script 04 turns calibrated continuum-model parameters (results/03_calibrate_models)
into predicted f_new, the fraction of carbon younger than each site's labeling
duration, i.e. the model's age CDF cdfA(Duration_labeling). It writes one
prediction table per model to results/04_model_predictions, plus a
depth-resolved lognormal table when layer calibrations exist.

The script is run end to end on small hand-made calibration files in a temporary
directory, and its outputs are checked against independent calculations:
  * lognormal predictions, for bulk sites and (optionally) layers, against a
    direct numerical integral, for the mean and q05/q95 stock scenarios;
  * the soft- and hard-cutoff alpha sweeps against each model's own cdfA;
  * generate_predictions keeps rows it cannot predict as NaN, in place.
"""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from soil_diskin import constants

SCRIPT = Path(__file__).resolve().parents[1]/'notebooks/04_collect_continuum_model_predictions.py'


@pytest.mark.parametrize('with_layers', [False, True])
def test_collect_bulk_and_layer_predictions(tmp_path, monkeypatch, with_layers):
    """Run script 04 on toy calibrations and check every prediction it writes.

    Parametrized over with_layers: without a depth-resolved lognormal calibration
    the script must skip the layer predictions (and not create their folder);
    with one, the layers must go through the same prediction path as bulk sites.
    """
    # f_new does not depend on radiocarbon; a constant atmosphere keeps model
    # construction independent of the real 14C record.
    monkeypatch.setattr(constants, 'INTERP_R_14C', lambda t: 1.)
    monkeypatch.chdir(tmp_path)
    calibration = tmp_path/'results/03_calibrate_models'
    predictions = tmp_path/'results/04_model_predictions'
    calibration.mkdir(parents=True)
    predictions.mkdir()
    # Three bulk sites; labeling durations span zero, decades and millennia.
    sites = pd.DataFrame({'Duration_labeling': [0., 20., 4000.], 'total_fnew': [.1, .2, .3]})
    sites.to_csv(tmp_path/'results/processed_balesdent_2018.csv', index=False)
    # The script reads these calibrations too. Their predictions are not checked
    # here; they only need to exist and be valid.
    for name, params in [
        ('powerlaw', {'t_min': 1., 't_max': 100.}),
        ('general_powerlaw', {'t_min': 1., 't_max': 100., 'alpha': .5}),
        ('weibull', {'k': .5, 'alpha': 10.}),
    ]:
        table = pd.DataFrame(params, index=sites.index)
        table.to_csv(calibration/f'{name}_model_optimization_results.csv', index=False)
        if name == 'general_powerlaw':
            table.to_csv(calibration/f'{name}_model_optimization_results_alpha_half.csv', index=False)
    # Long-format alpha sweep (one row per alpha and site), deliberately not in
    # site order. The same parameters serve the soft- and hard-cutoff sweeps.
    sweep = pd.DataFrame({'alpha': [.2, .2, .2, .7, .7, .7], 'site_index': [2, 0, 1, 1, 2, 0],
                          't_min': [.1, .2, .3, 1., 2., 3.], 't_max': [1e3, 2e3, 3e3, 1e4, 2e4, 3e4]})
    sweep.to_csv(calibration/'soft_cutoff_powerlaw_model_alpha_sweep.csv', index=False)
    sweep.to_csv(calibration/'general_powerlaw_model_alpha_sweep.csv', index=False)
    # Lognormal calibration (03b format). Site 0 is fast and site 1 slow; site 2 has
    # no fit. Only site 1 has q05/q95 stock scenarios. mu and sigma are set so the
    # model reproduces each (turnover, model_A) pair, inverting
    # T = exp(-mu + sigma^2/2) and A = T exp(sigma^2).
    params = pd.DataFrame({'turnover': [10., 2000., np.nan], 'model_A': [9000., 9000., np.nan],
        'turnover_q05': [np.nan, 1600., np.nan], 'model_A_05': [np.nan, 8000., np.nan],
        'turnover_q95': [np.nan, 2400., np.nan], 'model_A_95': [np.nan, 10000., np.nan]})
    for suffix, column in [('', 'turnover'), ('_05', 'turnover_q05'), ('_95', 'turnover_q95')]:
        params['sigma'+suffix] = np.sqrt(np.log(params['model_A'+suffix]/params[column]))
        params['mu'+suffix] = params['sigma'+suffix]**2/2-np.log(params[column])
    # A compromise has fitted turnover different from observed stock/input.
    # Predictions must use fitted parameters, never reconstruct them from that input.
    params.loc[1, ['mu', 'sigma']] = [-4., .01]
    filename = '03b_lognormal_predictions_calcurve_python.csv'
    params.to_csv(calibration/filename, index=False)
    # The layer calibration reuses the same parameters, with layer metadata. The
    # zero-padded profile_id checks that string IDs survive the round trip.
    if with_layers:
        (calibration/'depth_resolved').mkdir()
        layers = params.assign(profile_id='001', z_top_cm=[0., 10., 20.],
                               z_bottom_cm=[10., 20., 30.], Duration_labeling=20., fnew_obs=.2)
        layers.to_csv(calibration/'depth_resolved'/filename, index=False)

    # Run the script as `python 04_...py` would; namespace holds its globals.
    namespace = runpy.run_path(str(SCRIPT), run_name='__main__')

    # Lognormal predictions, bulk and (if present) layers.
    for folder in ['', 'depth_resolved'] if with_layers else ['']:
        result = pd.read_csv(predictions/folder/'lognormal_model_predictions.csv', dtype={'profile_id': str})
        observations = layers if folder else sites
        # The output carries the observation table through unchanged ...
        pd.testing.assert_frame_equal(result[observations.columns], observations)
        # ... with a prediction wherever there is a fit, and NaN (row kept) where not.
        assert result.predicted_fnew.iloc[:2].notna().all()
        assert np.isnan(result.predicted_fnew.iloc[2])
        assert result.loc[[0, 2], ['predicted_fnew_05', 'predicted_fnew_95']].isna().all().all()
        for suffix, column in [('', 'turnover'), ('_05', 'turnover_q05'), ('_95', 'turnover_q95')]:
            for i in ([0, 1] if not suffix else [1]):
                mu, sigma = params.loc[i, ['mu'+suffix, 'sigma'+suffix]]
                duration = 20. if folder else sites.Duration_labeling.iloc[i]
                # Independently integrate the stock-weighted log-rate distribution.
                # The stock is weighted by 1/k relative to inputs, so if ln k ~ N(mu, sigma)
                # for inputs, the stock's ln k ~ N(mu - sigma^2, sigma), and the fraction
                # younger than t is E[1 - exp(-k t)] over that distribution.
                expected, _ = quad(lambda z: norm.pdf(z)*(-np.expm1(
                    -np.exp(mu-sigma**2+sigma*z)*duration)), -10., 10.)
                assert result.loc[i, 'predicted_fnew'+suffix] == pytest.approx(expected, abs=1e-10)
    if not with_layers:
        assert not (predictions/'depth_resolved').exists()
    # Alpha sweeps: output is sorted by (alpha, site_index) whatever the input order,
    # and each prediction is the model's cdfA at that site's labeling duration.
    for fname, model_class in [('soft_cutoff_power_law', 'SoftCutoffPowerLawDisKin'),
                               ('general_power_law', 'GeneralPowerLawDisKin')]:
        swept = pd.read_csv(predictions/f'{fname}_model_predictions_alpha_sweep.csv')
        assert swept[['alpha', 'site_index']].values.tolist() == [[.2, 0], [.2, 1], [.2, 2], [.7, 0], [.7, 1], [.7, 2]]
        for _, row in sweep.iterrows():
            model = namespace[model_class](row.t_min, row.t_max, row.alpha)
            duration = sites.Duration_labeling.iloc[int(row.site_index)]
            predicted = swept.set_index(['alpha', 'site_index']).loc[(row.alpha, row.site_index), 'predicted_fnew']
            assert predicted == pytest.approx(model.cdfA(duration))
    # Missing calibration/duration rows remain aligned instead of being dropped.
    # A non-contiguous index checks alignment by label, not position. Only the first
    # row is predictable: the others have a missing duration, a negative duration,
    # and a missing mu.
    data = pd.DataFrame({'Duration_labeling': [20., np.nan, -1., 4000.]}, index=[3, 5, 8, 9])
    fitted = pd.DataFrame({'mu': [-2., -2., -2., np.nan], 'sigma': 1.}, index=data.index)
    result = namespace['generate_predictions'](namespace['LognormalDisKin'], fitted, ['mu', 'sigma'], data)
    assert result.index.equals(data.index)
    assert result.predicted_fnew.notna().tolist() == [True, False, False, False]
