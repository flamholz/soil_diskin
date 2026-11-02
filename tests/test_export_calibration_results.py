"""Tests for notebooks/04_export_calibration_results.py."""
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

from soil_diskin.constants import LAMBDA_14C

SCRIPT = Path(__file__).resolve().parents[1]/'notebooks/04_export_calibration_results.py'
main = runpy.run_path(str(SCRIPT))['main']


def test_exports_real_and_permuted_calibrations(tmp_path):
    sites = pd.DataFrame({'Latitude': [10., 20., 30.], 'Longitude': [-1., -2., -3.]})
    sites.to_csv(tmp_path/'sites.csv', index=False)
    calibration = tmp_path/'calibration'
    calibration.mkdir()
    real = pd.DataFrame({'turnover': [10., 20., 30.], 'fm_obs': [.9, 1.1, .8],
                         't_min': [1., 2., 3.], 't_max': [1e3, 2e3, 3e3], 'mu': [-1., -2., -3.],
                         'sigma': [1., 1.5, 2.], 'model_T': [10., 20., 30.], 'model_A': [100., 200., 300.]})
    # Permuted rows name their site in site_index; fm_obs is another site's.
    permuted = real.iloc[[2, 0, 1]].assign(replicate=0, site_index=[2, 0, 1], fm_obs=[.9, 1.1, .8],
                                           fm_obs_original=[.8, .9, 1.1])
    for model in ['powerlaw', 'lognormal']:
        real.to_csv(calibration/f'{model}_model_optimization_results.csv', index=False)
        permuted.to_csv(calibration/f'{model}_model_optimization_results_permuted.csv', index=False)

    main(tmp_path/'sites.csv', calibration, tmp_path/'out')

    out = pd.read_csv(tmp_path/'out'/'lognormal_calibration_results_balesdent.csv')
    assert list(out.columns) == ['source_row', 'latitude', 'longitude', 'turnover', 'fm_obs', 'A_r',
                                 'mu', 'sigma', 'model_T', 'model_A']
    out = pd.read_csv(tmp_path/'out'/'powerlaw_calibration_results_permuted_balesdent.csv')
    assert list(out.columns) == ['replicate', 'source_row', 'latitude', 'longitude', 'turnover', 'fm_obs',
                                 'fm_obs_original', 'A_r', 't_min', 't_max', 'model_T', 'model_A']
    assert out.source_row.tolist() == [2, 0, 1]
    assert out.latitude.tolist() == [30., 10., 20.]
    # A_r is the apparent age of the permuted fm the model was calibrated to.
    np.testing.assert_allclose(out.A_r, -np.log(out.fm_obs) / LAMBDA_14C)
    assert out.t_min.tolist() == [3., 1., 2.]


def test_misaligned_coordinates_are_rejected(tmp_path):
    sites = pd.DataFrame({'Latitude': [10., 20.], 'Longitude': [-1., -2.]})
    sites.to_csv(tmp_path/'sites.csv', index=False)
    calibration = tmp_path/'calibration'
    calibration.mkdir()
    rows = pd.DataFrame({'turnover': [10., 20.], 'fm_obs': [.9, 1.], 't_min': 1., 't_max': 1e3,
                         'mu': -1., 'sigma': 1., 'model_T': 1., 'model_A': 1.})
    for model in ['powerlaw', 'lognormal']:
        rows.to_csv(calibration/f'{model}_model_optimization_results.csv', index=False)
        # site_index points at the wrong sites for these coordinates.
        rows.assign(replicate=0, site_index=[1, 0], Latitude=[10., 20.]).to_csv(
            calibration/f'{model}_model_optimization_results_permuted.csv', index=False)
    with pytest.raises(ValueError, match='Latitude'):
        main(tmp_path/'sites.csv', calibration, tmp_path/'out')


def test_exports_israd_calibrations_with_identifiers(tmp_path):
    main_israd = runpy.run_path(str(SCRIPT))['main_israd']
    real = pd.DataFrame({'entry_name': ['A_2020', 'A_2020', 'B_2024'], 'site_name': ['s1', 's2', 's3'],
                         'pro_name': ['p1', 'p2', 'p3'], 'sampling_year': [1990, 2005, 2019],
                         'latitude': [10., 20., 30.], 'longitude': [-1., -2., -3.], 'stock_kg_m2': 5.,
                         'turnover': [10., 20., 30.], 'fm_obs': [.9, 1.1, .8],
                         't_min': [1., 2., np.nan], 't_max': [1e3, 2e3, np.nan], 'mu': [-1., -2., np.nan],
                         'sigma': [1., 1.5, np.nan], 'model_T': [10., 20., np.nan],
                         'model_A': [100., 200., np.nan], 'model_fm': .9,
                         'calibration_status': ['calibrated', 'calibrated', 'no_atmosphere']})
    # The null shuffles turnover among the profiles within the atmospheric record.
    permuted = real.iloc[:2].assign(replicate=0, site_index=[0, 1], turnover=[20., 10.],
                                    turnover_original=[10., 20.])
    for model in ['powerlaw', 'lognormal']:
        real.to_csv(tmp_path/f'{model}_model_optimization_results.csv', index=False)
        permuted.to_csv(tmp_path/f'{model}_model_optimization_results_permuted.csv', index=False)

    main_israd(tmp_path, tmp_path/'out')

    out = pd.read_csv(tmp_path/'out'/'lognormal_calibration_results_israd.csv')
    assert list(out.columns) == ['source_row', 'entry_name', 'site_name', 'pro_name', 'sampling_year',
                                 'latitude', 'longitude', 'turnover', 'fm_obs', 'A_r',
                                 'mu', 'sigma', 'model_T', 'model_A', 'calibration_status']
    assert out.calibration_status.tolist() == ['calibrated', 'calibrated', 'no_atmosphere']
    out = pd.read_csv(tmp_path/'out'/'powerlaw_calibration_results_permuted_israd.csv')
    assert list(out.columns) == ['replicate', 'source_row', 'entry_name', 'site_name', 'pro_name',
                                 'sampling_year', 'latitude', 'longitude', 'turnover', 'turnover_original',
                                 'fm_obs', 'A_r', 't_min', 't_max', 'model_T', 'model_A', 'calibration_status']
    assert out.turnover.tolist() == [20., 10.]
    # fm_obs, and so A_r, stay with the profile.
    np.testing.assert_allclose(out.A_r, -np.log([.9, 1.1]) / LAMBDA_14C)
