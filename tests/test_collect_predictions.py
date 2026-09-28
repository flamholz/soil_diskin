"""Script 04 uses the same prediction path for bulk observations and layers."""
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
    monkeypatch.setattr(constants, 'INTERP_R_14C', lambda t: 1.)
    monkeypatch.chdir(tmp_path)
    calibration = tmp_path/'results/03_calibrate_models'
    predictions = tmp_path/'results/04_model_predictions'
    calibration.mkdir(parents=True)
    predictions.mkdir()
    sites = pd.DataFrame({'Duration_labeling': [0., 20., 4000.], 'total_fnew': [.1, .2, .3]})
    sites.to_csv(tmp_path/'results/processed_balesdent_2018.csv', index=False)
    for name, params in [
        ('powerlaw', {'t_min': 1., 't_max': 100.}),
        ('general_powerlaw', {'t_min': 1., 't_max': 100., 'beta': .5}),
        ('weibull', {'k': .5, 'alpha': 10.}),
    ]:
        table = pd.DataFrame(params, index=sites.index)
        table.to_csv(calibration/f'{name}_model_optimization_results.csv', index=False)
        if name == 'general_powerlaw':
            table.to_csv(calibration/f'{name}_model_optimization_results_beta_half.csv', index=False)
    params = pd.DataFrame({'turnover': [10., 2000., np.nan], 'pred': [9000., 9000., np.nan],
        'turnover_q05': [np.nan, 1600., np.nan], 'pred_05': [np.nan, 8000., np.nan],
        'turnover_q95': [np.nan, 2400., np.nan], 'pred_95': [np.nan, 10000., np.nan]})
    for suffix, column in [('', 'turnover'), ('_05', 'turnover_q05'), ('_95', 'turnover_q95')]:
        params['sigma'+suffix] = np.sqrt(np.log(params['pred'+suffix]/params[column]))
        params['mu'+suffix] = params['sigma'+suffix]**2/2-np.log(params[column])
    # A compromise has fitted turnover different from observed stock/input.
    # Predictions must use fitted parameters, never reconstruct them from that input.
    params.loc[1, ['mu', 'sigma']] = [-4., .01]
    filename = '03b_lognormal_predictions_calcurve_python.csv'
    params.to_csv(calibration/filename, index=False)
    if with_layers:
        (calibration/'depth_resolved').mkdir()
        layers = params.assign(profile_id='001', z_top_cm=[0., 10., 20.],
                               z_bottom_cm=[10., 20., 30.], Duration_labeling=20., fnew_obs=.2)
        layers.to_csv(calibration/'depth_resolved'/filename, index=False)

    namespace = runpy.run_path(str(SCRIPT), run_name='__main__')
    for folder in ['', 'depth_resolved'] if with_layers else ['']:
        result = pd.read_csv(predictions/folder/'lognormal_model_predictions.csv', dtype={'profile_id': str})
        observations = layers if folder else sites
        pd.testing.assert_frame_equal(result[observations.columns], observations)
        assert result.predicted_fnew.iloc[:2].notna().all()
        assert np.isnan(result.predicted_fnew.iloc[2])
        assert result.loc[[0, 2], ['predicted_fnew_05', 'predicted_fnew_95']].isna().all().all()
        for suffix, column in [('', 'turnover'), ('_05', 'turnover_q05'), ('_95', 'turnover_q95')]:
            for i in ([0, 1] if not suffix else [1]):
                mu, sigma = params.loc[i, ['mu'+suffix, 'sigma'+suffix]]
                duration = 20. if folder else sites.Duration_labeling.iloc[i]
                # Independently integrate the stock-weighted log-rate distribution.
                expected, _ = quad(lambda z: norm.pdf(z)*(-np.expm1(
                    -np.exp(mu-sigma**2+sigma*z)*duration)), -10., 10.)
                assert result.loc[i, 'predicted_fnew'+suffix] == pytest.approx(expected, abs=1e-10)
    if not with_layers:
        assert not (predictions/'depth_resolved').exists()
    # Missing calibration/duration rows remain aligned instead of being dropped.
    data = pd.DataFrame({'Duration_labeling': [20., np.nan, -1., 4000.]}, index=[3, 5, 8, 9])
    fitted = pd.DataFrame({'mu': [-2., -2., -2., np.nan], 'sigma': 1.}, index=data.index)
    result = namespace['generate_predictions'](namespace['LognormalDisKin'], fitted, ['mu', 'sigma'], data)
    assert result.index.equals(data.index)
    assert result.predicted_fnew.notna().tolist() == [True, False, False, False]
