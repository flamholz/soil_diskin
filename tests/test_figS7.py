"""Tests for notebooks/figS7.py, the layered input-allocation sensitivity figure.

Figure S7 asks how the depth-resolved lognormal f_new predictions (Fig. 4B)
depend on how NPP inputs are allocated across soil layers. It reads the saved
Fig. 4B table, recomputes each layer's input under five allocation schemes
(Jackson et al. 1996 root profiles, global or per vegetation, with or without a
50% surface share, and an exponential with h = 10 cm), each at 100% and 50% of
NPP entering the soil. It refits the lognormal model per layer in memory for
every (fraction, scheme) pair and plots the ten resulting prediction sets.

The tests check that:
  * layer_inputs integrates the Jackson root profile correctly, maps land use and
    vegetation to the right Jackson group, and scales linearly with the soil NPP
    fraction (test_jackson_mapping_without_comparison_script);
  * main() reproduces the saved baseline, refits all ten tables without writing
    any intermediate files, keeps mean and q05/q95 stock scenarios independent,
    and compares every panel on the same set of observation/prediction pairs
    (test_refits_in_memory_and_writes_only_one_figure).
"""
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
    """layer_inputs matches Jackson's closed form for every vegetation group.

    Jackson et al. (1996) give the cumulative root fraction above depth d (cm) as
    Y(d) = 1 - beta**d. Normalized over 0-100 cm, a 7-23 cm layer receives
    (beta**7 - beta**23) / (1 - beta**100) of the root-allocated NPP. The
    *_surface schemes put half of NPP uniformly into 0-10 cm, of which the 7-23 cm
    layer overlaps 3 cm, hence 0.5 * roots + 0.5 * 3/10.
    """
    # Blocking the import (a None entry in sys.modules makes `import` fail) and
    # running the file as a script shows figS7 works without the older
    # compare_jackson_inputs analysis. run_path does not call main().
    monkeypatch.setitem(sys.modules, 'notebooks.compare_jackson_inputs', None)
    standalone = runpy.run_path(analysis.__file__)
    # One 7-23 cm layer per row, chosen to hit each branch of the group mapping:
    # crop, grass, tree and shrub use their own beta; sylvopastoral, savanna and
    # Trifolium (matched case-insensitively) and missing labels fall back to global.
    sites = pd.DataFrame({
        'land_use': ['CROP', 'GRASSLAND', 'FOREST', 'FOREST', 'FOREST', 'GRASSLAND', 'GRASSLAND', None],
        'vegetation': ['maize', 'pasture', 'pine', 'shrub', 'sylvopastoral', 'savanna', 'Trifolium', None],
        'npp_kg_m2_yr': .6, 'z_top_cm': 7., 'z_bottom_cm': 23.})
    # Expected beta per row, from JACKSON_BETA: crop, grass, tree, shrub, then global x4.
    beta = np.array([.961, .952, .970, .978]+[.966]*4)
    roots = (beta**7-beta**23)/(1-beta**100)
    for scheme in ['jackson_vegetation', 'jackson_vegetation_surface']:
        weights = roots if scheme == 'jackson_vegetation' else .5*roots+.5*3/10
        full = standalone['layer_inputs'](sites, scheme, 1.)
        half = standalone['layer_inputs'](sites, scheme, .5)
        # Inputs are NPP (0.6 kg C/m2/yr) times the layer's share ...
        np.testing.assert_allclose(full, .6*weights, rtol=1e-12)
        # ... and halving the fraction of NPP entering the soil halves every input.
        np.testing.assert_allclose(half, full/2, rtol=1e-12)


def test_refits_in_memory_and_writes_only_one_figure(tmp_path, monkeypatch):
    """main() refits all ten (fraction, scheme) tables in memory and saves one figure.

    Four layers from two profiles exercise the edge cases: profile 001 has a
    measured surface layer and a SoilGrids-backfilled layer (the only row with
    q05/q95 stocks); the forest profile has a measured layer and one with a
    missing stock, which cannot be fit. So 3 of 4 rows get mean predictions and
    only row 1 gets q05/q95 predictions, in every table.
    """
    monkeypatch.chdir(tmp_path)  # No config files or prepared bulk data here.
    # A constant atmosphere keeps the fits fast and independent of the real 14C record.
    atmosphere = AtmC14(np.array([0.]), np.array([1.]), 1.)
    monkeypatch.setattr(analysis, 'load_atm14c', lambda path: atmosphere)
    sites = pd.DataFrame({'profile_id': ['001', '001', 'forest', 'forest'], 'layer': [0, 1, 3, 4],
        'z_top_cm': [0., 7., 60., 80.], 'z_bottom_cm': [7., 23., 80., 100.],
        'land_use': ['CROP', 'CROP', 'FOREST', 'FOREST'], 'vegetation': ['maize', 'maize', 'pine', 'pine'],
        'npp_kg_m2_yr': .6, 'stock_kg_m2': [.2, .8, .4, np.nan], 'fm_obs': .9,
        'stock_kg_m2_q05': [np.nan, .5, np.nan, np.nan],
        'stock_kg_m2_q95': [np.nan, 1.2, np.nan, np.nan],
        'Duration_labeling': [12., 20., 30., 20.], 'fnew_obs': [.12, .2, .3, .4],
        'stock_source': ['Balesdent Layers', 'SoilGrids backfill', 'Balesdent Layers', 'Balesdent Layers']})
    # Build a stand-in for the saved Fig. 4B table: baseline inputs and fits. It is
    # written in reverse row order to check that main() re-sorts by profile and layer.
    sites['input_kg_m2_yr'] = analysis.layer_inputs(sites, analysis.BASELINE_SCHEME, 1.)
    fm_evaluator = analysis.cached_radiocarbon(atmosphere)
    baseline = analysis.refit(sites, sites.input_kg_m2_yr, atmosphere, fm_evaluator, n_jobs=1)
    source = tmp_path/'baseline.csv'
    sites.join(baseline.drop(columns=sites.columns, errors='ignore')).iloc[::-1].to_csv(source, index=False)
    figures, captured = [], {}
    plot = analysis.plot_comparison

    # Wrap plot_comparison to record the tables main() builds and the figure it draws.
    def capture(tables, exclude_layered_soilgrids=False):
        captured.update(tables)
        fig = plot(tables, exclude_layered_soilgrids)
        figures.append(fig)
        return fig

    monkeypatch.setattr(analysis, 'plot_comparison', capture)
    output = tmp_path/'comparison.png'
    analysis.main(input=source, output=output, n_jobs=1)
    # Refits stay in memory: the only new file is the figure.
    assert set(tmp_path.iterdir()) == {source, output} and output.stat().st_size > 0
    # 5 schemes x 2 soil NPP fractions.
    assert len(captured) == 10
    # Each panel is annotated with its number of plotted pairs: the 3 fittable rows.
    assert all(ax.texts[0].get_text().startswith('N = 3\n') for ax in figures[0].axes)
    # The 100% baseline panel is the saved table, restored to the original row order.
    prediction_columns = ['predicted_fnew', 'predicted_fnew_05', 'predicted_fnew_95']
    pd.testing.assert_frame_equal(captured[(1., analysis.BASELINE_SCHEME)][prediction_columns], baseline[prediction_columns])
    # Mean and q05/q95 stock scenarios are fit independently, so their coverage
    # follows each scenario's stocks, the same in every table.
    for table in captured.values():
        assert table.predicted_fnew.notna().tolist() == [True, True, True, False]
        assert table.predicted_fnew_05.notna().tolist() == [False, True, False, False]
        assert table.predicted_fnew_95.notna().tolist() == [False, True, False, False]
    # The 50% panels really are refit with the smaller inputs.
    assert not np.allclose(captured[(.5, analysis.BASELINE_SCHEME)].predicted_fnew[:3], baseline.predicted_fnew[:3])
    # A missing scenario stock must not reuse baseline parameters from the input.
    missing = analysis.refit(sites.drop(columns='stock_kg_m2_q95'), sites.input_kg_m2_yr,
                             atmosphere, fm_evaluator, n_jobs=1)
    assert missing.predicted_fnew_95.isna().all()
    # Every panel shows the same pairs: a failed fit in any one table drops that
    # row from all ten panels.
    captured[(.5, 'jackson_global')].loc[0, 'predicted_fnew'] = np.nan
    fig = plot(captured)
    assert all(ax.texts[0].get_text().startswith('N = 2\n') for ax in fig.axes)
    plt.close(fig)
    # Excluding SoilGrids-backfilled layers drops row 1, again from every panel.
    captured[(.5, 'jackson_global')].loc[0, 'predicted_fnew'] = .2
    fig = plot(captured, exclude_layered_soilgrids=True)
    assert all(ax.texts[0].get_text().startswith('N = 2\n') for ax in fig.axes)
    plt.close(fig)
    # plot_comparison only returns figures; it never saves them.
    assert set(tmp_path.iterdir()) == {source, output}
