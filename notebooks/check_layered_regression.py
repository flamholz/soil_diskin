"""Refit frozen historical inputs and compare with the saved D=v=0 run.

Run from the repository root. Current raw-data changes must not alter the frozen
500-layer calibration targets. Optionally check all saved corrected-target
predictions with --current-refits results/layered_radiocarbon_refit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from soil_diskin.layered_data import PreparedProfiles, allocate_inputs, file_digest
from soil_diskin.layered_lognormal import InputAllocation, layer_model
from soil_diskin.layered_workflow import run_profiles, source_hashes
from soil_diskin.radiocarbon_utils import load_atm14c
from soil_diskin.run_output import require_empty_output, run_record

KEYS = ['profile_id', 'layer']
CALIBRATION_COLUMNS = KEYS + ['latitude', 'longitude', 'z_top_cm', 'z_bottom_cm',
    'stock_kg_m2', 'fm_obs', 'npp_kg_m2_yr', 'duration_years', 'fnew_obs', 'fnew_observation_valid']
FIT_COLUMNS = ['mu', 'sigma', 'stock_pred_kg_m2', 'fm_pred', 'fnew_pred']


def compare(actual: pd.DataFrame, expected: pd.DataFrame, columns: list[str]) -> dict:
    actual, expected = [frame.set_index(KEYS).sort_index() for frame in (actual, expected)]
    assert actual.index.is_unique and expected.index.is_unique, 'duplicate profile/layer keys'
    pd.testing.assert_index_equal(actual.index, expected.index)
    differences = {}
    for column in columns:
        # Much tighter than the production quadrature tolerances; accommodates
        # solver/platform roundoff rather than changing the scientific target.
        np.testing.assert_allclose(actual[column], expected[column], rtol=1e-8, atol=1e-9,
                                   err_msg=column, equal_nan=False)
        differences[column] = float(np.max(np.abs(actual[column]-expected[column])))
    return differences


def run_regression(output: Path, reference: Path, coupled: Path, current: Path | None = None) -> dict:
    require_empty_output(output)
    frozen = pd.read_csv(reference/'layers.csv', float_precision='round_trip')
    settings = json.loads((reference/'run.json').read_text())
    assert len(frozen) == 500 and frozen.profile_id.nunique() == 50
    assert frozen.groupby('profile_id').layer.apply(lambda x: set(x) == set(range(10))).all()
    atmosphere_path = Path('data/14C_atm_annot.csv')
    assert file_digest(atmosphere_path) == settings['data']['atmosphere']['sha256'], 'atmosphere changed'
    atmosphere = load_atm14c(str(atmosphere_path))
    output.mkdir(parents=True, exist_ok=True)
    report = {'reference': str(reference), 'coupled_reference': str(coupled),
              'n_profiles': 50, 'n_layers': 500, 'rtol': 1e-8, 'atol': 1e-9,
              'source_sha256': {**source_hashes(), str(Path(__file__)): file_digest(__file__)},
              'frozen_sha256': {str(path): file_digest(path) for path in
                  [reference/'layers.csv', reference/'run.json', coupled/'parameters.csv',
                   coupled/'predictions.csv', atmosphere_path]}}
    with run_record(output/'regression.json', report):
        # No raw workbook/raster reload: these are the original inputs, before
        # coordinate recovery and the radiocarbon spatial-filling correction.
        inputs = allocate_inputs(frozen[CALIBRATION_COLUMNS], InputAllocation(settings['input_depth_cm']))
        prepared = PreparedProfiles(inputs, pd.DataFrame(), settings['data'])
        run_profiles(prepared, atmosphere, output/'refit',
                     times=tuple(settings['requested_times_years']),
                     max_nfev=settings['max_nfev_per_start'], log_rate_step=settings['log_rate_step'])
        fitted = pd.read_csv(output/'refit/layers.csv', float_precision='round_trip')
        assert fitted.success.all() and fitted.quadrature_ok.all()
        report['versus_saved_independent_max_abs'] = compare(fitted, frozen, FIT_COLUMNS)
        coupled_parameters = pd.read_csv(coupled/'parameters.csv', float_precision='round_trip')
        zero = coupled_parameters.query('D_cm2_yr == 0 and v_cm_yr == 0 and h_cm == 30 and candidate_id == 0')
        compare(frozen, zero, ['stock_kg_m2', 'fm_obs', 'npp_kg_m2_yr', 'duration_years', 'fnew_obs'])
        coupled_predictions = pd.read_csv(coupled/'predictions.csv', float_precision='round_trip')
        primary_predictions = coupled_predictions.query(
            'D_cm2_yr == 0 and v_cm_yr == 0 and h_cm == 30 and candidate_id == 0 and at_label_duration')
        zero = zero.merge(primary_predictions[KEYS+['fnew_pred']], on=KEYS, validate='one_to_one')
        report['versus_saved_zero_transport_max_abs'] = compare(fitted, zero, FIT_COLUMNS)
        # Verify requested prediction times as well as the labeling duration.
        new_predictions = pd.read_csv(output/'refit/predictions.csv', float_precision='round_trip')
        new_predictions = new_predictions.query('candidate_id == 0').set_index(KEYS+['time_years']).sort_index()
        old_predictions = coupled_predictions.query(
            'D_cm2_yr == 0 and v_cm_yr == 0 and h_cm == 30 and candidate_id == 0')
        old_predictions = old_predictions.set_index(KEYS+['time_years']).sort_index()
        pd.testing.assert_index_equal(new_predictions.index, old_predictions.index)
        np.testing.assert_allclose(new_predictions.fnew_pred, old_predictions.fnew_pred, rtol=1e-8, atol=1e-9)
        report['requested_fnew_max_abs'] = float(np.max(np.abs(new_predictions.fnew_pred-old_predictions.fnew_pred)))
        if current is not None:
            # Audit the refactored forward model against every saved input scheme.
            paths = sorted(current.glob('npp*/*/layers.csv'))
            if not paths:
                raise ValueError(f'no saved corrected-target fits under {current}')
            model = layer_model(atmosphere)
            checks = []
            for path in paths:
                saved = pd.read_csv(path, float_precision='round_trip')
                calculated = []
                for row in saved.itertuples():
                    allocation = InputAllocation(row.input_depth_cm, row.surface_fraction, row.soil_npp_fraction)
                    layer_input = allocation.layer_inputs(row.npp_kg_m2_yr)[row.layer]
                    np.testing.assert_allclose(layer_input, row.input_kg_m2_yr, rtol=1e-14)
                    prediction = model.predict(row.mu, row.sigma, layer_input, (row.duration_years,))
                    calculated.append({'profile_id': row.profile_id, 'layer': row.layer,
                        'stock_pred_kg_m2': prediction.stock, 'fm_pred': prediction.fm, 'fnew_pred': prediction.fnew[0]})
                errors = compare(pd.DataFrame(calculated), saved, FIT_COLUMNS[2:])
                checks.append({'path': str(path), 'sha256': file_digest(path), 'n_layers': len(saved), **errors})
            report['current_forward_checks'] = checks
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/layered_review_regression'))
    parser.add_argument('--reference', type=Path, default=Path('results/layered_no_transport'))
    parser.add_argument('--coupled-reference', type=Path, default=Path('results/layered_lognormal_all_profiles'))
    parser.add_argument('--current-refits', type=Path)
    args = parser.parse_args()
    run_regression(args.output_dir, args.reference, args.coupled_reference, args.current_refits)
