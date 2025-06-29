import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
process_baledant_data = __import__('notebooks.01_preprocess_baledant_data', fromlist=['process_baledant_data']).process_baledant_data

@pytest.fixture
def mock_raw_data():
    """
    Provides a mock DataFrame simulating the raw Baledant et al. 2018 data.
    """
    data = {
        'Latitude':         [10.0,  15.0,   20.0,   25.0,   30.0],
        'Longitude':        [100.0, 105.0,  110.0,  115.0,  120.0],
        'Duration_labeling':[10,    20,     10,     20,     100],
        'MAT_C':            [18.0,  19.0,   16.0,   20.0,   18],
        'PANN_mm':          [1200.0,1500.0, 900.0,  1100.0, 1300.0],
        'P to PET ratio':   [0.9,   0.95,   0.7,    0.85,   1.2],
        'Ctotal_0-10':      [10.0,  12.0,   8.0,    11.0,   15.0],   # Carbon in 0-10
        'Ctotal_0-20':      [18.0,  22.0,   14.0,   20.0,   25.0], # Cumulative carbon to 20cm
        'Ctotal_0-30':      [24.0,  30.0,   18.0,   26.0,   35.0], # Cumulative carbon to 30cm
        'Ctotal_0-100estim':[50.0,  60.0,   np.nan, 55.0,   np.nan],
        'f_0':              [0.5,   0.6,    0.4,    0.55,   0.1],
        'f_10':             [0.4,   0.5,    0.3,    0.45,   0.15],
        'f_20':             [0.3,   0.4,    0.2,    0.35,   0.4],
    }
    np.random.seed(42)  # For reproducibility
    for i in range(30, 110, 10):
        data[f'Ctotal_0-{i+10}'] = [data[f'Ctotal_0-{i}'][j] + np.random.rand() * 5 for j in range(len(list(data.values())[0]))]
        data[f'f_{i}'] = [np.random.rand() * 0.2 for _ in range(len(list(data.values())[0]))]
    return pd.DataFrame(data)

def test_process_baledant_data(mock_raw_data):
    """
    Tests the process_baledant_data function with mock data.
    """
    processed_data = process_baledant_data(mock_raw_data)

    # Assertions for the filtered data (site 3 should be removed)
    assert len(processed_data) == 4
    assert 20 not in processed_data['Latitude'].values
    
    # Test Ctotal_0-100estim NaN filling
    # Original site 2 (index 2 in mock_raw_data) had NaN for Ctotal_0-100estim.
    # After filtering, this becomes the second row (index 1) in the tropical_sites intermediate dataframe.
    # The mean of Ctotal_0-100estim from the *filtered* data (sites 0, 1, 3) should be used.
    # Mean = (50.0 + 60.0 + 55.0) / 3 = 165.0 / 3 = 55.0
    assert processed_data.loc[processed_data['Latitude'] == 15.0, 'Ctotal_0-100estim'].iloc[0] == 60.0 # Site 1
    assert processed_data.loc[processed_data['Latitude'] == 25.0, 'Ctotal_0-100estim'].iloc[0] == 55.0 # Site 3
    # The site that was originally index 2 (Latitude 20.0) is filtered out.
    # The site that was originally index 0 (Latitude 10.0) is still there.
    # The site that was originally index 1 (Latitude 15.0) is still there.
    # The site that was originally index 3 (Latitude 25.0) is still there.
    # The NaN filling happens *before* groupby.
    # So, the mean should be calculated from the filtered `tropical_sites` dataframe.
    # The `tropical_sites` dataframe will contain original indices 0, 1, 3.
    # Ctotal_0-100estim values for these are 50.0, 60.0, 55.0.
    # Mean = (50.0 + 60.0 + 55.0) / 3 = 55.0
    # So, if any of these had NaN, they would be filled with 55.0.
    # In my mock data, only the filtered-out site (index 2) had NaN.
    # So, after filtering, there should be no NaNs in Ctotal_0-100estim.
    assert processed_data['Ctotal_0-100estim'].isnull().sum() == 0
    assert processed_data.loc[3, 'Ctotal_0-100estim'] == 55.0 # Site 3 (original index 3)

    # Assertions for weight columns and total_fnew
    # This requires detailed calculation based on the mock data.
    # Let's check the sum of weights for a site. It should be close to 1.0.
    # For site 0 (original index 0, now index 0 in processed_data):
    # C_dens: [8.0, 6.0, ...] (from diff of Ctotal_10-20 - Ctotal_0-10, Ctotal_20-30 - Ctotal_10-20)
    # Sum of C_dens for site 0: 8.0 + 6.0 + ...
    # Weights for site 0: C_dens / sum(C_dens)
    # This is getting complex due to the dynamic number of Ctotal_ columns.
    # I will assert the presence of weight columns and total_fnew, and check their types/ranges.

    expected_weight_columns = [f'weight_{i*10}' for i in range(1, 10)] # Ctotal_0-10 is not part of diff
    for col in expected_weight_columns:
        assert col in processed_data.columns
        assert processed_data[col].dtype == np.float64
        assert (processed_data[col] >= 0).all() # Weights should be non-negative

    assert 'total_fnew' in processed_data.columns
    assert processed_data['total_fnew'].dtype == np.float64
    assert (processed_data['total_fnew'] >= 0).all()
    assert (processed_data['total_fnew'] <= 1).all() # Fraction of new C should be between 0 and 1

    # For 

    # Check if grouping by Latitude, Longitude, Duration_labeling works
    assert len(processed_data.groupby(['Latitude','Longitude','Duration_labeling'])) == 4 # Should be 4 unique combinations

    # Test case for missing values in deeper layers (not explicitly covered by current mock data, but the fillna handles Ctotal_0-100estim)
    # The `site_C_weights[np.nansum(site_C_weights,axis=1) == 0,:] = mean_C_weights` line handles cases where a site has no C density data.
    # Let's add a test case for this.

    # Test with a site having all Ctotal_ values as NaN (after filtering)
    mock_data_with_nan_Ctotal = {
        'Latitude': [10.0, 15.0],
        'Longitude': [100.0, 105.0],
        'Duration_labeling': [10, 20],
        'MAT_C': [18.0, 19.0],
        'PANN_mm': [1200.0, 1500.0],
        'P to PET ratio': [0.9, 0.95],
        'Ctotal_0-10': [10.0, np.nan],
        'Ctotal_10-20': [18.0, np.nan],
        'Ctotal_20-30': [24.0, np.nan],
        'Ctotal_0-100estim': [50.0, 60.0],
        'f_0': [0.5, 0.6],
        'f_10': [0.4, 0.5],
        'f_20': [0.3, 0.4],
    }
    for i in range(30, 100, 10):
        mock_data_with_nan_Ctotal[f'Ctotal_{i}-{i+10}'] = [mock_data_with_nan_Ctotal[f'Ctotal_{i-10}-{i}'][j] + np.random.rand() * 5 if not np.isnan(mock_data_with_nan_Ctotal[f'Ctotal_{i-10}-{i}'][j]) else np.nan for j in range(2)]
        mock_data_with_nan_Ctotal[f'f_{i}'] = [np.random.rand() * 0.2 for _ in range(2)]

    mock_raw_data_with_nan_Ctotal_df = pd.DataFrame(mock_data_with_nan_Ctotal)
    processed_data_nan_Ctotal = process_baledant_data(mock_raw_data_with_nan_Ctotal_df)

    # For the second site (index 1), C_dens will be all NaN.
    # So, site_C_weights for this site should be replaced by mean_C_weights.
    # This is hard to assert directly without re-calculating mean_C_weights in the test.
    # Instead, I will check if the weights for the second site are not all NaN.
    # The sum of weights for the second site should be 1.0 (or very close).
    
    # Get the weight columns from the processed data
    weight_cols_processed = [col for col in processed_data_nan_Ctotal.columns if col.startswith('weight_')]
    
    # Check sum of weights for the site with NaN Ctotal (original index 1, now index 1 in processed_data_nan_Ctotal)
    assert np.isclose(processed_data_nan_Ctotal.loc[1, weight_cols_processed].sum(), 1.0)
