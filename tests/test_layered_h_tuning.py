"""Hyperparameter selection must respect locations and validation isolation."""
import numpy as np
import pandas as pd
import pytest

from notebooks.tune_layered_input_depth import make_split, score, choose_h


def test_split_is_reproducible_grouped_and_independent_of_row_order():
    profiles = pd.DataFrame({'profile_id': [f'p{i}' for i in range(50)],
                             'latitude': np.arange(50)//2, 'longitude': 0.})
    a = make_split(profiles, seed=42).set_index('profile_id').sort_index()
    b = make_split(profiles.sample(frac=1, random_state=5), seed=42).set_index('profile_id').sort_index()
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby(['latitude','longitude']).split.nunique().eq(1).all()
    assert set(a.split) == {'train','validation','test'}
    assert a.groupby('split').size().to_dict() == {'test':10, 'train':30, 'validation':10}


def test_selection_uses_validation_only_and_keeps_failed_candidates_ineligible():
    scores = pd.DataFrame({'h_cm':[10,30,60,10,60],
        'split':['validation','validation','validation','test','test'],
        'rmse':[.1,.2,.05,99.,0.], 'kge_2012':[.7,.8,.9,-99.,1.],
        'eligible':[True,True,False,True,True]})
    assert choose_h(scores, 'rmse') == 10
    assert choose_h(scores, 'kge_2012') == 30
    frame = pd.DataFrame({'profile_id':['a','a'], 'latitude':0., 'longitude':0.,
        'fnew_obs':[.1,.3], 'fnew_pred':[.2,.4], 'success':[True,False],
        'quadrature_ok':[True,True], 'stock_kg_m2':1., 'stock_pred_kg_m2':1.,
        'fm_obs':1., 'fm_pred':1., 'mu_at_bound':False, 'sigma_at_bound':False})
    result = score(frame)
    assert result['n_layer_pairs'] == 2
    assert result['rmse'] == pytest.approx(.1)
    assert not result['eligible']
    frame.loc[1,'fnew_pred'] = np.nan
    assert not score(frame)['eligible']
    with pytest.raises(ValueError):
        choose_h(scores.assign(eligible=False), 'rmse')
