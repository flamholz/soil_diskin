import pandas as pd
import numpy as np
from unittest import TestCase

from soil_diskin.data_wrangling import process_balesdent_data
pbd = process_balesdent_data

# Column with index 2 will have all NaN values for Ctotal_0-...
# The last two columns have the same Latitude, Longitude, and Duration_labeling
# and will be averaged together.
# The column with index 3 has a NaN for Ctotal_0-100estim that should be
# but in the raw data this case does not occur, so it will remain as is. 
MOCK_DATA = {
    'Latitude':         [10.0,  15.0,   20.0,   25.0,   30.0,   30.0],
    'Longitude':        [100.0, 105.0,  110.0,  115.0,  120.0,  120.0],
    'Duration_labeling':[10,    20,     10,     20,     100,    100],
    'MAT_C':            [18.0,  19.0,   16.0,   20.0,   18,     18],
    'PANN_mm':          [1200.0,1500.0, 900.0,  1100.0, 1300.0, 1300.0],
    'P to PET ratio':   [0.9,   0.95,   0.7,    0.85,   1.2,    1.2],
    # this row is here in the actual data as a convenience for difference calculations
    'Ctotal_0-0':       [0.0,   0.0,    np.nan,    0.0,    0.0, 0.0],  # Carbon in 0-0 
    'Ctotal_0-10':      [10.0,  12.0,   np.nan, 11.0,   15.0,   20.0], # Carbon in 0-10
    'Ctotal_0-20':      [18.0,  22.0,   np.nan, 20.0,   25.0,   30.0], # Cumulative carbon to 20cm
    'Ctotal_0-30':      [24.0,  30.0,   np.nan, 26.0,   35.0,   40.0], # Cumulative carbon to 30cm
    'Ctotal_0-100estim':[50.0,  60.0,   np.nan, np.nan, 35.0,   40.0],
    'f_0':              [0.5,   0.6,    0.4,    0.55,   0.1,    0.1],
    'f_10':             [0.4,   0.5,    0.3,    0.45,   0.15,   0.15],
    'f_20':             [0.3,   0.4,    0.2,    0.35,   0.4,    0.4],
}

def mock_raw_data():
    """
    Provides a mock DataFrame simulating the raw Baledant et al. 2018 data.

    Returns:
    """
    data = MOCK_DATA.copy()
    np.random.seed(42)  # For reproducibility
    for i in range(30, 110, 10):
        data[f'Ctotal_0-{i+10}'] = [data[f'Ctotal_0-{i}'][j] + np.random.rand() * 5 for j in range(len(list(data.values())[0]))]
        data[f'f_{i}'] = [np.random.rand() * 0.2 for _ in range(len(list(data.values())[0]))]
    return pd.DataFrame(data)


class TestProcessBalesdentData(TestCase):
    """
    Tests for the process_balesdent_data function.
    """

    def setUp(self):
        self.mock_data = mock_raw_data()
        self.processed_data = pbd(self.mock_data)
        # weight_cols = [f'weight_{i*10}' for i in range(0, 10)]
        # print(self.processed_data[weight_cols])
        
    def test_columns(self):
        """
        Test that the processed data contains the expected columns.
        """
        expected_columns = ['Latitude', 'Longitude', 'Duration_labeling', 'total_fnew', 
                            'weight_0', 'weight_10', 'weight_20', 'weight_30', 'weight_40', 
                            'weight_50', 'weight_60', 'weight_70', 'weight_80', 
                            'weight_90', 'Ctotal_0-100estim']
        column_set = set(self.processed_data.columns)
        self.assertTrue(set(expected_columns).issubset(column_set))

    def test_sites(self):
        """
        Test that the correct number of unique sites are processed.
        """
        # We expect the third column to be removed due to filtering (Latitude 20.0)
        # and the last one (index 5) to be averaged with the prior
        # (index 4, Latitude 30.0, Duration 100)
        self.assertEqual(len(self.processed_data), 3)

        filtered_site_idx = 2  # The index of the site that should be filtered out
        site_lat = MOCK_DATA['Latitude'][filtered_site_idx]
        site_lng = MOCK_DATA['Longitude'][filtered_site_idx]
        site_duration = MOCK_DATA['Duration_labeling'][filtered_site_idx]

        # Make sure a site with this combination does not exist (it was filtered out)
        mask = (self.processed_data['Latitude'] == site_lat) & \
               (self.processed_data['Longitude'] == site_lng) & \
               (self.processed_data['Duration_labeling'] == site_duration)
        self.assertFalse(mask.any())

        # The site with Ctotal_0-100estim should be removed even though 
        # such a case does not occur in the real data.
        filtered_site_idx = 3  # The index of the site that should be 
        site_lat = MOCK_DATA['Latitude'][filtered_site_idx]
        site_lng = MOCK_DATA['Longitude'][filtered_site_idx]
        site_duration = MOCK_DATA['Duration_labeling'][filtered_site_idx]

        # Make sure a site with this combination does not exist (it was filtered out)
        mask = (self.processed_data['Latitude'] == site_lat) & \
               (self.processed_data['Longitude'] == site_lng) & \
               (self.processed_data['Duration_labeling'] == site_duration)
        self.assertFalse(mask.any())

        filtered_site_idx = 5  # The index of the site that should be averaged with the prior
        site_lat = MOCK_DATA['Latitude'][filtered_site_idx]
        site_lng = MOCK_DATA['Longitude'][filtered_site_idx]
        site_duration = MOCK_DATA['Duration_labeling'][filtered_site_idx]

        # Find the rows with this combination
        mask = (self.processed_data['Latitude'] == site_lat) & \
               (self.processed_data['Longitude'] == site_lng) & \
               (self.processed_data['Duration_labeling'] == site_duration)
        self.assertEqual(mask.sum(), 1)  # There should be only one row after averaging

        # Make sure that some averaging happened.
        original_values = self.mock_data.iloc[[4, 5]]
        expected_Ctotal = original_values['Ctotal_0-100estim'].mean()
        processed_Ctotal = self.processed_data.loc[mask, 'Ctotal_0-100estim'].values[0]
        self.assertAlmostEqual(expected_Ctotal, processed_Ctotal)

        # For the other sites, they should remain unchanged.
        for idx in [0, 1]:
            site_lat = MOCK_DATA['Latitude'][idx]
            site_lng = MOCK_DATA['Longitude'][idx]
            site_duration = MOCK_DATA['Duration_labeling'][idx]

            mask = (self.processed_data['Latitude'] == site_lat) & \
                   (self.processed_data['Longitude'] == site_lng) & \
                   (self.processed_data['Duration_labeling'] == site_duration)
            self.assertEqual(mask.sum(), 1)  # Should exist

            # Check that Ctotal_0-100estim is unchanged, even if they are NaN
            processed_Ctotal = self.processed_data.loc[mask, 'Ctotal_0-100estim'].values[0]
            expected_Ctotal = MOCK_DATA['Ctotal_0-100estim'][idx]
            if pd.isna(expected_Ctotal):
                self.assertTrue(pd.isna(processed_Ctotal))
            else:
                self.assertEqual(expected_Ctotal, processed_Ctotal)

    def test_weights_sum_to_one(self):
        """
        Test that the weights for each site sum to 1.
        """
        weight_columns = [f'weight_{i*10}' for i in range(0, 10)]
        for _, row in self.processed_data.iterrows():
            weight_sum = row[weight_columns].sum()
            self.assertAlmostEqual(weight_sum, 1.0, places=5)

    def test_keep_missing_soc_false(self):
        """
        Test that sites with missing SOC are removed when keep_missing_soc=False (default).
        """
        processed_data = pbd(self.mock_data, keep_missing_soc=False)
        
        # Sites with indices 2 and 3 should be filtered out and 
        # sites with indices 4 and 5 merged into one row as they
        # have the same Latitude, Longitude, and Duration_labeling.
        # Since we start with 6 rows, we expect 3 rows remaining.
        # Index 2: all Ctotal columns are NaN
        # Index 3: Ctotal_0-100estim is NaN
        self.assertEqual(len(processed_data), 3)
        
        # Verify site at index 2 (Latitude 20.0) is not present
        mask = (processed_data['Latitude'] == 20.0) & \
               (processed_data['Longitude'] == 110.0) & \
               (processed_data['Duration_labeling'] == 10)
        self.assertFalse(mask.any())
        
        # Verify site at index 3 (Latitude 25.0) is not present
        mask = (processed_data['Latitude'] == 25.0) & \
               (processed_data['Longitude'] == 115.0) & \
               (processed_data['Duration_labeling'] == 20)
        self.assertFalse(mask.any())
        
        # All remaining sites should have non-null Ctotal_0-100estim
        self.assertTrue(processed_data['Ctotal_0-100estim'].notna().all())

    def test_keep_missing_soc_true(self):
        """
        Test that sites with missing SOC are kept when keep_missing_soc=True.
        """
        processed_data = pbd(self.mock_data, keep_missing_soc=True)
        
        # With keep_missing_soc=True, we expect more sites to be retained
        # Index 2 and 3 should be kept -- missing C data is allowed.
        # Indices 4 and 5: merged into one row
        # Since we started with 6 sites, expect 5 rows remaining (0, 1, 3, merged 4+5)
        self.assertEqual(len(processed_data), 5)
        
        # Verify site at index 2 (all NaN Ctotal) is still filtered out
        mask = (processed_data['Latitude'] == 20.0) & \
               (processed_data['Longitude'] == 110.0) & \
               (processed_data['Duration_labeling'] == 10)
        self.assertTrue(mask.any(), "Site with all NaN Ctotal should still be filtered")
        
        # Verify site at index 3 (only Ctotal_0-100estim is NaN) IS present
        mask = (processed_data['Latitude'] == 25.0) & \
               (processed_data['Longitude'] == 115.0) & \
               (processed_data['Duration_labeling'] == 20)
        self.assertTrue(mask.any(), "Site with NaN Ctotal_0-100estim should be kept when keep_missing_soc=True")
        
        # Verify that this site has NaN for Ctotal_0-100estim
        if mask.any():
            ctotal_value = processed_data.loc[mask, 'Ctotal_0-100estim'].values[0]
            self.assertTrue(pd.isna(ctotal_value), "Kept site should have NaN Ctotal_0-100estim")
        
        # Verify sites with valid SOC are still present
        mask = (processed_data['Latitude'] == 10.0) & \
               (processed_data['Longitude'] == 100.0) & \
               (processed_data['Duration_labeling'] == 10)
        self.assertTrue(mask.any())
        self.assertEqual(processed_data.loc[mask, 'Ctotal_0-100estim'].values[0], 50.0)

