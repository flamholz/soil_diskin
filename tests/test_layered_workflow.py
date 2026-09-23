"""End-to-end public workflow seam using a known synthetic profile."""

import json

import numpy as np
import pandas as pd

from soil_diskin.layered_data import PreparedProfiles
from soil_diskin.layered_fitting import FitSettings
from soil_diskin.layered_lognormal import LayeredLognormal
from soil_diskin.layered_workflow import run_profiles
from soil_diskin.radiocarbon_utils import AtmC14


def test_batch_exports_link_fits_predictions_and_evaluation_times(tmp_path):
    atm = AtmC14(np.array([0.]), np.array([1.]), 1.)
    model = LayeredLognormal(0, 0, 30, atm)
    truth = model.predict(np.full(10, -1.), np.full(10, 2.5), 0.5)
    profiles = pd.DataFrame({'profile_id': ['synthetic']*10, 'layer': np.arange(10),
                             'z_top_cm': np.arange(10)*10, 'z_bottom_cm': np.arange(1,11)*10,
                             'stock_kg_m2': truth.stocks, 'fm_obs': truth.fm,
                             'npp_kg_m2_yr': [0.5]*10, 'duration_years': [20.]*10,
                             'fnew_obs': [0.123]*10})
    invalid = profiles.assign(profile_id='invalid', stock_kg_m2=np.nan)
    prepared = PreparedProfiles(pd.concat([profiles, invalid], ignore_index=True),
                                pd.DataFrame([{'profile_id': 'missing',
                                                        'reason': 'missing NPP'}]))
    output = tmp_path/'run'
    run_profiles(prepared, atm, [(0., 0., 30.)], output,
                 settings=FitSettings(n_starts=1, max_nfev=20), times=(1., 100.))
    fits = pd.read_csv(output/'fits.csv')
    predictions = pd.read_csv(output/'predictions.csv')
    parameters = pd.read_csv(output/'parameters.csv')
    assert len(fits) == 1 and bool(fits.success.iloc[0])
    assert bool(fits.quadrature_ok.iloc[0])
    assert len(parameters) == 10
    assert predictions.time_years.unique().tolist() == [1., 20., 100.]
    assert predictions.loc[predictions.time_years == 20, 'fnew_obs'].notna().all()
    assert predictions.loc[predictions.time_years != 20, 'fnew_obs'].isna().all()
    assert predictions.candidate_id.unique().tolist() == fits.candidate_id.tolist()
    assert (predictions.fnew_pred != 0.123).any()
    metadata = json.loads((output/'run.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['settings']['seed'] == 0
    assert metadata['evaluation_used_for_fitting'] is False
    assert metadata['profile_triples_without_candidates'] == 1
    assert metadata['profile_triples_with_candidates'] == 1
    assert pd.read_csv(output/'exclusions.csv').profile_id.tolist() == ['missing']
