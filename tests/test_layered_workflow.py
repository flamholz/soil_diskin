"""Readable end-to-end pipeline, evaluation isolation, and output protection."""
import json
import numpy as np
import pandas as pd
import pytest

from soil_diskin.layered_data import PreparedProfiles
from soil_diskin.layered_lognormal import LayerLognormal, input_weights
from soil_diskin.layered_workflow import run_profiles
from soil_diskin.radiocarbon_utils import AtmC14


def test_pipeline_keeps_layer_identity_and_fnew_out_of_fitting(tmp_path, monkeypatch):
    atm = AtmC14(np.array([0.]), np.array([1.]), 1.)
    model = LayerLognormal(atm)
    truth = [model.predict(-1., 2.5, rate) for rate in .5*input_weights(30)]
    profiles = pd.DataFrame({'profile_id': ['synthetic']*10, 'layer': np.arange(10),
        'z_top_cm': np.arange(10)*10, 'z_bottom_cm': np.arange(1,11)*10,
        'stock_kg_m2': [p.stock for p in truth], 'fm_obs': [p.fm for p in truth],
        'npp_kg_m2_yr': .5, 'duration_years': 20., 'fnew_obs': np.linspace(.1, .2, 10)})
    prepared = PreparedProfiles(profiles, pd.DataFrame(columns=['profile_id', 'reason']))
    outputs = [tmp_path/'original', tmp_path/'changed_evaluation']
    for output in outputs:
        run_profiles(prepared, atm, output, input_depth=30, times=(1., 100.))
        prepared.profiles['fnew_obs'] = .9
    a, b = [pd.read_csv(o/'layers.csv') for o in outputs]
    np.testing.assert_array_equal(a[['mu','sigma']], b[['mu','sigma']])
    assert len(a) == 10 and a.success.all() and a.quadrature_ok.all()
    pred = pd.read_csv(outputs[0]/'predictions.csv')
    assert set(pred.time_years) == {1., 20., 100.}
    assert pred.loc[pred.time_years == 20, 'fnew_obs'].notna().all()
    assert pred.loc[pred.time_years != 20, 'fnew_obs'].isna().all()
    assert len(pd.read_csv(outputs[0]/'fits.csv')) == 30
    metadata = json.loads((outputs[0]/'run.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['input_depth_cm'] == 30
    assert metadata['evaluation_used_for_parameter_fitting'] is False
    with pytest.raises(FileExistsError):
        run_profiles(prepared, atm, outputs[0])

    def failed_plot(*args):
        raise RuntimeError('plot failed')

    monkeypatch.setattr('soil_diskin.layered_workflow.plot_comparison', failed_plot)
    failed = tmp_path/'failed_plot'
    with pytest.raises(RuntimeError, match='plot failed'):
        run_profiles(prepared, atm, failed)
    assert pd.read_csv(failed/'layers.csv').success.all()
    assert json.loads((failed/'run.json').read_text())['status'] == 'interrupted_or_failed'
