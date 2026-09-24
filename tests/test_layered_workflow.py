"""Readable end-to-end pipeline, evaluation isolation, and output protection."""
import json
import numpy as np
import pandas as pd
import pytest

from soil_diskin.layered_data import PreparedProfiles, allocate_inputs
from soil_diskin.layered_lognormal import InputAllocation, layer_model
from soil_diskin.layered_workflow import run_profiles
from soil_diskin.radiocarbon_utils import AtmC14


def run_pipeline(prepared, atmosphere, output, *, allocation=InputAllocation(), **kwargs):
    inputs = allocate_inputs(prepared.profiles, allocation)
    return run_profiles(PreparedProfiles(inputs, prepared.excluded, prepared.metadata), atmosphere, output, **kwargs)


@pytest.mark.parametrize('surface_fraction', [0., .5])
@pytest.mark.parametrize('soil_npp_fraction', [1., .5])
def test_pipeline_keeps_layer_identity_and_fnew_out_of_fitting(tmp_path, monkeypatch, surface_fraction,
                                                             soil_npp_fraction):
    atm = AtmC14(np.array([0.]), np.array([1.]), 1.)
    model = layer_model(atm)
    weights = InputAllocation(30, surface_fraction).soil_input_fractions
    truth = [model.predict(-1., 2.5, rate) for rate in .5*soil_npp_fraction*weights]
    profiles = pd.DataFrame({'profile_id': ['synthetic']*10, 'layer': np.arange(10),
        'z_top_cm': np.arange(10)*10, 'z_bottom_cm': np.arange(1,11)*10,
        'stock_kg_m2': [p.stock for p in truth], 'fm_obs': [p.fm for p in truth],
        'npp_kg_m2_yr': .5, 'duration_years': 20., 'fnew_obs': np.linspace(.1, .2, 10)})
    prepared = PreparedProfiles(profiles, pd.DataFrame(columns=['profile_id', 'reason']))
    outputs = [tmp_path/'original', tmp_path/'changed_evaluation']
    for output in outputs:
        run_pipeline(prepared, atm, output, allocation=InputAllocation(30, surface_fraction, soil_npp_fraction),
                     times=(1., 100.))
        prepared.profiles['fnew_obs'] = .9
    a, b = [pd.read_csv(o/'layers.csv') for o in outputs]
    np.testing.assert_array_equal(a[['mu','sigma']], b[['mu','sigma']])
    assert len(a) == 10 and a.success.all() and a.quadrature_ok.all()
    np.testing.assert_allclose(a.input_kg_m2_yr, .5*soil_npp_fraction*weights)
    np.testing.assert_allclose(a.input_kg_m2_yr.sum(), .5*soil_npp_fraction)
    np.testing.assert_allclose(a.npp_kg_m2_yr, .5)  # Preserve original site NPP.
    np.testing.assert_allclose(a.observed_turnover_years, a.stock_kg_m2/a.input_kg_m2_yr)
    pred = pd.read_csv(outputs[0]/'predictions.csv')
    assert set(pred.time_years) == {1., 20., 100.}
    assert pred.loc[pred.time_years == 20, 'fnew_obs'].notna().all()
    assert pred.loc[pred.time_years != 20, 'fnew_obs'].isna().all()
    assert len(pd.read_csv(outputs[0]/'fits.csv')) == 30
    metadata = json.loads((outputs[0]/'run.json').read_text())
    assert metadata['status'] == 'complete'
    from pathlib import Path
    from soil_diskin.layered_data import file_digest
    for source, digest in metadata['source_code_sha256'].items():
        assert Path(source).is_file() and file_digest(source) == digest
    assert metadata['input_depth_cm'] == 30
    assert metadata['surface_fraction'] == surface_fraction
    assert metadata['soil_npp_fraction'] == soil_npp_fraction
    np.testing.assert_allclose(metadata['layer_input_weights'], weights)
    np.testing.assert_allclose(metadata['layer_npp_fractions'], soil_npp_fraction*weights)
    assert metadata['evaluation_used_for_parameter_fitting'] is False

    # Removing layers must not reindex depths or redistribute their NPP inputs.
    sparse = profiles.iloc[[2, 8]].copy()
    sparse.loc[sparse.layer == 8, 'fnew_obs'] = np.nan
    partial_output = tmp_path/'partial'
    run_pipeline(PreparedProfiles(sparse, prepared.excluded), atm, partial_output,
                 allocation=InputAllocation(30, surface_fraction, soil_npp_fraction))
    partial = pd.read_csv(partial_output/'layers.csv').set_index('layer')
    reference = a.set_index('layer').loc[[2, 8]]
    assert partial.index.tolist() == [2, 8]
    np.testing.assert_allclose(partial[['mu', 'sigma', 'input_kg_m2_yr', 'fnew_pred']],
                               reference[['mu', 'sigma', 'input_kg_m2_yr', 'fnew_pred']])
    assert partial.input_kg_m2_yr.sum() < .5*soil_npp_fraction
    assert len(partial) == 2 and partial.fnew_pred.notna().all()
    assert pd.read_csv(partial_output/'metrics.csv').n_layer_pairs.iloc[0] == 1

    with pytest.raises(FileExistsError):
        run_pipeline(prepared, atm, outputs[0])

    def failed_plot(*args):
        raise RuntimeError('plot failed')

    monkeypatch.setattr('soil_diskin.layered_workflow.plot_comparison', failed_plot)
    failed = tmp_path/'failed_plot'
    with pytest.raises(RuntimeError, match='plot failed'):
        run_pipeline(prepared, atm, failed)
    assert pd.read_csv(failed/'layers.csv').success.all()
    assert json.loads((failed/'run.json').read_text())['status'] == 'interrupted_or_failed'


@pytest.mark.parametrize('fraction', [0., -.1, 1.1, np.nan, np.inf])
def test_invalid_soil_npp_fraction_rejected_before_outputs(tmp_path, fraction):
    from notebooks.compare_jackson_inputs import run_comparison

    prepared = PreparedProfiles(pd.DataFrame({'profile_id': ['synthetic']}), pd.DataFrame())
    atmosphere = AtmC14(np.array([0.]), np.array([1.]), 1.)
    output = tmp_path/'invalid'
    with pytest.raises(ValueError, match='soil_npp_fraction'):
        run_pipeline(prepared, atmosphere, output, allocation=InputAllocation(soil_npp_fraction=fraction))
    with pytest.raises(ValueError, match='soil_npp_fraction'):
        run_comparison(output, soil_npp_fraction=fraction)
    assert not output.exists()


@pytest.mark.parametrize('duration', [20., np.nan])
def test_missing_evaluation_still_writes_metrics_and_plot(tmp_path, duration):
    atmosphere = AtmC14(np.array([0.]), np.array([1.]), 1.)
    profiles = pd.DataFrame({'profile_id': ['no-evaluation'], 'layer': [0],
        'z_top_cm': [0.], 'z_bottom_cm': [10.], 'stock_kg_m2': [1.], 'fm_obs': [.9],
        'npp_kg_m2_yr': [.5], 'duration_years': [duration], 'fnew_obs': [np.nan]})
    run_pipeline(PreparedProfiles(profiles, pd.DataFrame()), atmosphere, tmp_path)
    metrics = pd.read_csv(tmp_path/'metrics.csv').iloc[0]
    assert metrics.n_layer_pairs == 0
    assert metrics.evaluation_status == 'no_evaluable_observations'
    assert np.isnan(metrics.rmse) and np.isnan(metrics.kge_2012)
    assert (tmp_path/'fnew_scatter.png').is_file() and (tmp_path/'fnew_scatter.pdf').is_file()
    layers = pd.read_csv(tmp_path/'layers.csv')
    np.testing.assert_allclose(layers.implied_turnover_years, layers.stock_kg_m2/layers.input_kg_m2_yr)
    np.testing.assert_allclose(layers.model_turnover_years, layers.stock_pred_kg_m2/layers.input_kg_m2_yr)


def test_failed_fits_keep_complete_schema_and_invalidate_evaluation(tmp_path, monkeypatch):
    from soil_diskin.layered_evaluation import score
    from soil_diskin.layered_lognormal import InputAllocation

    def failed_fit(*args, **kwargs):
        raise FloatingPointError('synthetic integration failure')

    monkeypatch.setattr('soil_diskin.layered_workflow.fit_layer', failed_fit)
    atmosphere = AtmC14(np.array([0.]), np.array([1.]), 1.)
    profiles = pd.DataFrame({'profile_id': ['failed'], 'latitude': [0.], 'longitude': [0.],
        'layer': [4], 'z_top_cm': [40.], 'z_bottom_cm': [50.], 'stock_kg_m2': [1.],
        'fm_obs': [.9], 'npp_kg_m2_yr': [.5], 'duration_years': [20.], 'fnew_obs': [.2]})
    run_pipeline(PreparedProfiles(profiles, pd.DataFrame()), atmosphere, tmp_path,
                 allocation=InputAllocation(10., .5, .5))
    layers = pd.read_csv(tmp_path/'layers.csv')
    assert layers.message.iloc[0] == 'synthetic integration failure'
    assert layers[['mu', 'sigma', 'stock_pred_kg_m2', 'fm_pred', 'model_turnover_years']].isna().all().all()
    evaluation = score(layers)
    assert not evaluation['eligible'] and evaluation['n_layer_pairs'] == 1
    assert evaluation['evaluation_status'] == 'nonfinite_values'
    metrics = pd.read_csv(tmp_path/'metrics.csv').iloc[0]
    assert metrics.unconverged_layer_pairs == metrics.unchecked_layer_pairs == 1
    assert metrics.n_layer_pairs == 1 and np.isnan(metrics.rmse)
    assert (tmp_path/'fnew_scatter.png').is_file()


def test_fit_interruption_preserves_completed_layers(tmp_path, monkeypatch):
    from soil_diskin.layered_lognormal import fit_layer

    calls = 0

    def interrupt_second_layer(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return fit_layer(*args, **kwargs)

    monkeypatch.setattr('soil_diskin.layered_workflow.fit_layer', interrupt_second_layer)
    atmosphere = AtmC14(np.array([0.]), np.array([1.]), 1.)
    profiles = pd.DataFrame({'profile_id': ['interrupted']*2, 'layer': [0, 1],
        'z_top_cm': [0., 10.], 'z_bottom_cm': [10., 20.], 'stock_kg_m2': [1., 1.],
        'fm_obs': [.9, .9], 'npp_kg_m2_yr': .5, 'duration_years': 20., 'fnew_obs': .2})
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(PreparedProfiles(profiles, pd.DataFrame()), atmosphere, tmp_path)
    assert pd.read_csv(tmp_path/'layers.csv').layer.tolist() == [0]
    assert json.loads((tmp_path/'run.json').read_text())['status'] == 'interrupted_or_failed'
