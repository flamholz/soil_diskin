"""The figure-only sensitivity run preserves allocations and stock scenarios."""
import runpy
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from notebooks import figS7 as analysis
from soil_diskin.radiocarbon_utils import AtmC14


def test_jackson_mapping_without_comparison_script(monkeypatch):
    monkeypatch.setitem(sys.modules, 'notebooks.compare_jackson_inputs', None)
    standalone = runpy.run_path(analysis.__file__)
    sites = pd.DataFrame({
        'land_use': ['CROP', 'GRASSLAND', 'FOREST', 'FOREST', 'FOREST', 'GRASSLAND', 'GRASSLAND', None],
        'vegetation': ['maize', 'pasture', 'pine', 'shrub', 'sylvopastoral', 'savanna', 'Trifolium', None],
        'npp_kg_m2_yr': .6, 'z_top_cm': 7., 'z_bottom_cm': 23.})
    beta = np.array([.961, .952, .970, .978]+[.966]*4)
    roots = (beta**7-beta**23)/(1-beta**100)
    for scheme in ['jackson_vegetation', 'jackson_vegetation_surface']:
        weights = roots if scheme == 'jackson_vegetation' else .5*roots+.5*3/10
        full = standalone['layer_inputs'](sites, scheme, 1.)
        half = standalone['layer_inputs'](sites, scheme, .5)
        np.testing.assert_allclose(full, .6*weights, rtol=1e-12)
        np.testing.assert_allclose(half, full/2, rtol=1e-12)


def test_refits_in_memory_and_writes_only_one_figure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # No config files or prepared bulk data here.
    atmosphere = AtmC14(np.array([0.]), np.array([1.]), 1.)
    monkeypatch.setattr(analysis, 'load_atm14c', lambda path: atmosphere)
    sites = pd.DataFrame({'profile_id': ['001', '001', 'forest', 'forest'], 'layer': [0, 1, 3, 4],
        'z_top_cm': [0., 7., 60., 80.], 'z_bottom_cm': [7., 23., 80., 100.],
        'land_use': ['CROP', 'CROP', 'FOREST', 'FOREST'], 'vegetation': ['maize', 'maize', 'pine', 'pine'],
        'npp_kg_m2_yr': .6, 'stock_kg_m2': [.2, .8, .4, np.nan], 'fm': .9,
        'stock_kg_m2_q05': [np.nan, .5, np.nan, np.nan],
        'stock_kg_m2_q95': [np.nan, 1.2, np.nan, np.nan],
        'Duration_labeling': [12., 20., 30., 20.], 'fnew_obs': [.12, .2, .3, .4],
        'stock_source': ['Balesdent Layers', 'SoilGrids backfill', 'Balesdent Layers', 'Balesdent Layers']})
    sites['input_kg_m2_yr'] = analysis.layer_inputs(sites, analysis.BASELINE_SCHEME, 1.)
    fm_evaluator = analysis.cached_radiocarbon(atmosphere)
    baseline = analysis.refit(sites, sites.input_kg_m2_yr, atmosphere, fm_evaluator, n_jobs=1)
    source = tmp_path/'baseline.csv'
    sites.join(baseline.drop(columns=sites.columns, errors='ignore')).iloc[::-1].to_csv(source, index=False)
    figures, captured = [], {}
    plot = analysis.plot_comparison

    def capture(tables, exclude_layered_soilgrids=False):
        captured.update(tables)
        fig = plot(tables, exclude_layered_soilgrids)
        figures.append(fig)
        return fig

    monkeypatch.setattr(analysis, 'plot_comparison', capture)
    output = tmp_path/'comparison.png'
    analysis.main(input=source, output=output, n_jobs=1)
    assert set(tmp_path.iterdir()) == {source, output} and output.stat().st_size > 0
    assert len(captured) == 10
    assert all(ax.texts[0].get_text().startswith('N = 3\n') for ax in figures[0].axes)
    prediction_columns = ['predicted_fnew', 'predicted_fnew_05', 'predicted_fnew_95']
    pd.testing.assert_frame_equal(captured[(1., analysis.BASELINE_SCHEME)][prediction_columns], baseline[prediction_columns])
    for table in captured.values():
        assert table.predicted_fnew.notna().tolist() == [True, True, True, False]
        assert table.predicted_fnew_05.notna().tolist() == [False, True, False, False]
        assert table.predicted_fnew_95.notna().tolist() == [False, True, False, False]
    assert not np.allclose(captured[(.5, analysis.BASELINE_SCHEME)].predicted_fnew[:3], baseline.predicted_fnew[:3])
    # A missing scenario stock must not reuse baseline parameters from the input.
    missing = analysis.refit(sites.drop(columns='stock_kg_m2_q95'), sites.input_kg_m2_yr,
                             atmosphere, fm_evaluator, n_jobs=1)
    assert missing.predicted_fnew_95.isna().all()
    # A failed alternative excludes that pair from every panel.
    captured[(.5, 'jackson_global')].loc[0, 'predicted_fnew'] = np.nan
    fig = plot(captured)
    assert all(ax.texts[0].get_text().startswith('N = 2\n') for ax in fig.axes)
    plt.close(fig)
    captured[(.5, 'jackson_global')].loc[0, 'predicted_fnew'] = .2
    fig = plot(captured, exclude_layered_soilgrids=True)
    assert all(ax.texts[0].get_text().startswith('N = 2\n') for ax in fig.axes)
    plt.close(fig)
    assert set(tmp_path.iterdir()) == {source, output}
