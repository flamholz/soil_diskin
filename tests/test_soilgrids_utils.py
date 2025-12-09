"""
Unit tests for soilgrids_utils module with mocked Earth Engine API.

These tests verify the SoilGrids data processing logic without requiring
actual Google Earth Engine authentication or internet connectivity.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd

from soil_diskin.soilgrids_utils import (
    initialize_earth_engine,
    get_soc_at_point,
    calculate_total_soc_0_100,
    get_soc_with_bulk_density,
    backfill_missing_soc
)


class TestInitializeEarthEngine(unittest.TestCase):
    """Test Earth Engine initialization."""
    
    @patch('soil_diskin.soilgrids_utils.ee')
    def test_initialize_success(self, mock_ee):
        """Test successful initialization."""
        mock_ee.Initialize.return_value = None
        
        initialize_earth_engine()
        
        mock_ee.Authenticate.assert_called_once()
        mock_ee.Initialize.assert_called()
    
    @patch('soil_diskin.soilgrids_utils.ee')
    def test_initialize_with_exception(self, mock_ee):
        """Test initialization with exception and retry."""
        mock_ee.Initialize.side_effect = [Exception("Not initialized"), None]
        
        initialize_earth_engine()
        
        # Should authenticate and initialize twice
        assert mock_ee.Authenticate.call_count == 2
        assert mock_ee.Initialize.call_count == 2


class TestGetSocAtPoint(unittest.TestCase):
    """Test getting SOC at a single point."""
    
    def setUp(self):
        """Set up mock Earth Engine components."""
        self.mock_ee_patcher = patch('soil_diskin.soilgrids_utils.ee')
        self.mock_ee = self.mock_ee_patcher.start()
        
        # Mock SOC values (in dg/kg, need to divide by 10)
        self.mock_soc_values = {
            'soc_0-5cm_mean': 150,    # 15 g/kg
            'soc_5-15cm_mean': 120,   # 12 g/kg
            'soc_15-30cm_mean': 100,  # 10 g/kg
            'soc_30-60cm_mean': 80,   # 8 g/kg
            'soc_60-100cm_mean': 60,  # 6 g/kg
            'soc_100-200cm_mean': 40  # 4 g/kg
        }
    
    def tearDown(self):
        """Clean up patches."""
        self.mock_ee_patcher.stop()
    
    def test_get_soc_at_point_success(self):
        """Test successful SOC retrieval."""
        # Set up mock image and sample
        mock_sample = Mock()
        mock_sample.get = lambda band: Mock(getInfo=lambda: self.mock_soc_values.get(band))
        
        mock_collection = Mock()
        mock_collection.first.return_value = mock_sample
        
        mock_image = Mock()
        mock_image.sample.return_value = mock_collection
        
        self.mock_ee.Image.return_value = mock_image
        self.mock_ee.Geometry.Point.return_value = Mock()
        
        # Call function
        result = get_soc_at_point(48.8566, 2.3522)
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(result['0-5cm'], 15.0)
        self.assertEqual(result['5-15cm'], 12.0)
        self.assertEqual(result['15-30cm'], 10.0)
        self.assertEqual(result['30-60cm'], 8.0)
        self.assertEqual(result['60-100cm'], 6.0)
        self.assertEqual(result['100-200cm'], 4.0)
    
    def test_get_soc_at_point_none_sample(self):
        """Test when sample returns None."""
        mock_collection = Mock()
        mock_collection.first.return_value = None
        
        mock_image = Mock()
        mock_image.sample.return_value = mock_collection
        
        self.mock_ee.Image.return_value = mock_image
        self.mock_ee.Geometry.Point.return_value = Mock()
        
        result = get_soc_at_point(48.8566, 2.3522)
        
        self.assertIsNone(result)
    
    def test_get_soc_at_point_missing_band(self):
        """Test when some bands are missing."""
        mock_values = self.mock_soc_values.copy()
        mock_values['soc_30-60cm_mean'] = None
        
        mock_sample = Mock()
        mock_sample.get = lambda band: Mock(getInfo=lambda: mock_values.get(band))
        
        mock_collection = Mock()
        mock_collection.first.return_value = mock_sample
        
        mock_image = Mock()
        mock_image.sample.return_value = mock_collection
        
        self.mock_ee.Image.return_value = mock_image
        self.mock_ee.Geometry.Point.return_value = Mock()
        
        result = get_soc_at_point(48.8566, 2.3522)
        
        self.assertIsNone(result['30-60cm'])


class TestCalculateTotalSoc(unittest.TestCase):
    """Test total SOC calculation."""
    
    def test_calculate_total_soc_complete(self):
        """Test calculation with complete data."""
        soc_dict = {
            '0-5cm': 15.0,
            '5-15cm': 12.0,
            '15-30cm': 10.0,
            '30-60cm': 8.0,
            '60-100cm': 6.0
        }
        
        total = calculate_total_soc_0_100(soc_dict)
        
        # Expected: (15*5 + 12*10 + 10*15 + 8*30 + 6*40) * 1.3 / (100)
        # = (75 + 120 + 150 + 240 + 240) * 1.3 / 100
        # = 825 * 1.3 / 100 = 10.725
        expected = (15*5 + 12*10 + 10*15 + 8*30 + 6*40) * 1.3 / 100
        
        self.assertAlmostEqual(total, expected, places=4)
    
    def test_calculate_total_soc_none_input(self):
        """Test with None input."""
        result = calculate_total_soc_0_100(None)
        self.assertIsNone(result)
    
    def test_calculate_total_soc_missing_layer(self):
        """Test with missing layer data."""
        soc_dict = {
            '0-5cm': 15.0,
            '5-15cm': None,  # Missing
            '15-30cm': 10.0,
            '30-60cm': 8.0,
            '60-100cm': 6.0
        }
        
        result = calculate_total_soc_0_100(soc_dict)
        self.assertIsNone(result)


class TestGetSocWithBulkDensity(unittest.TestCase):
    """Test SOC calculation with bulk density."""
    
    def setUp(self):
        """Set up mock Earth Engine components."""
        self.mock_ee_patcher = patch('soil_diskin.soilgrids_utils.ee')
        self.mock_ee = self.mock_ee_patcher.start()
        
        # Mock SOC values (in dg/kg)
        self.mock_soc_values = {
            'soc_0-5cm_mean': 150,
            'soc_5-15cm_mean': 120,
            'soc_15-30cm_mean': 100,
            'soc_30-60cm_mean': 80,
            'soc_60-100cm_mean': 60
        }
        
        # Mock bulk density values (in cg/cm³)
        self.mock_bdod_values = {
            'bdod_0-5cm_mean': 130,   # 1.3 kg/dm³
            'bdod_5-15cm_mean': 140,  # 1.4 kg/dm³
            'bdod_15-30cm_mean': 145, # 1.45 kg/dm³
            'bdod_30-60cm_mean': 150, # 1.5 kg/dm³
            'bdod_60-100cm_mean': 155 # 1.55 kg/dm³
        }
    
    def tearDown(self):
        """Clean up patches."""
        self.mock_ee_patcher.stop()
    
    def test_get_soc_with_bulk_density_success(self):
        """Test successful SOC calculation with bulk density."""
        # Create mock samples
        mock_soc_sample = Mock()
        mock_soc_sample.get = lambda band: Mock(getInfo=lambda: self.mock_soc_values.get(band))
        
        mock_bdod_sample = Mock()
        mock_bdod_sample.get = lambda band: Mock(getInfo=lambda: self.mock_bdod_values.get(band))
        
        # Create mock collections
        mock_soc_collection = Mock()
        mock_soc_collection.first.return_value = mock_soc_sample
        
        mock_bdod_collection = Mock()
        mock_bdod_collection.first.return_value = mock_bdod_sample
        
        # Create mock images
        mock_soc_image = Mock()
        mock_soc_image.sample.return_value = mock_soc_collection
        
        mock_bdod_image = Mock()
        mock_bdod_image.sample.return_value = mock_bdod_collection
        
        # Set up Image to return different mocks
        self.mock_ee.Image.side_effect = [mock_soc_image, mock_bdod_image]
        self.mock_ee.Geometry.Point.return_value = Mock()
        
        # Call function
        result = get_soc_with_bulk_density(48.8566, 2.3522)
        
        # Verify result is a positive number
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)
        
        # Calculate expected value
        # Layer calculations: (SOC_gkg / 1000) * BD_kgdm * thickness_cm
        expected = 0
        soc_vals = [15.0, 12.0, 10.0, 8.0, 6.0]  # g/kg
        bd_vals = [1.3, 1.4, 1.45, 1.5, 1.55]    # kg/dm³
        thicknesses = [5, 10, 15, 30, 40]        # cm
        
        for soc, bd, thick in zip(soc_vals, bd_vals, thicknesses):
            expected += (soc / 100.0) * bd * thick
        
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_get_soc_with_bulk_density_none_sample(self):
        """Test when samples return None."""
        mock_collection = Mock()
        mock_collection.first.return_value = None
        
        mock_image = Mock()
        mock_image.sample.return_value = mock_collection
        
        self.mock_ee.Image.return_value = mock_image
        self.mock_ee.Geometry.Point.return_value = Mock()
        
        result = get_soc_with_bulk_density(48.8566, 2.3522)
        
        self.assertIsNone(result)


class TestBackfillMissingSoc(unittest.TestCase):
    """Test backfilling missing SOC values in a DataFrame."""
    
    @patch('soil_diskin.soilgrids_utils.initialize_earth_engine')
    @patch('soil_diskin.soilgrids_utils.get_soc_with_bulk_density')
    def test_backfill_no_missing(self, mock_get_soc, mock_init):
        """Test when there are no missing values."""
        df = pd.DataFrame({
            'Latitude': [48.8566, 51.5074],
            'Longitude': [2.3522, -0.1278],
            'Ctotal_0-100estim': [5.2, 4.8],
            'C_data_source': ['Measured', 'Measured']
        })
        
        result_df, stats = backfill_missing_soc(df)
        
        # No calls to get_soc should be made
        mock_get_soc.assert_not_called()
        
        # Stats should show no missing
        self.assertEqual(stats['n_missing'], 0)
        self.assertEqual(stats['n_filled'], 0)
        self.assertEqual(stats['n_failed'], 0)
    
    @patch('soil_diskin.soilgrids_utils.initialize_earth_engine')
    @patch('soil_diskin.soilgrids_utils.get_soc_with_bulk_density')
    def test_backfill_with_missing(self, mock_get_soc, mock_init):
        """Test backfilling missing values."""
        # Mock successful SOC retrieval
        mock_get_soc.side_effect = [3.5, 4.2]
        
        df = pd.DataFrame({
            'Latitude': [48.8566, 51.5074, 40.7128],
            'Longitude': [2.3522, -0.1278, -74.0060],
            'Ctotal_0-100estim': [5.2, None, None],
            'C_data_source': ['Measured', 'Measured', 'Measured']
        })
        
        result_df, stats = backfill_missing_soc(df, use_bulk_density=True)
        
        # Should have called get_soc for the two missing locations
        self.assertEqual(mock_get_soc.call_count, 2)
        
        # Check stats
        self.assertEqual(stats['n_missing'], 2)
        self.assertEqual(stats['n_filled'], 2)
        self.assertEqual(stats['n_failed'], 0)
        self.assertEqual(stats['fill_rate'], 1.0)
        
        # Check filled values
        self.assertEqual(result_df.loc[1, 'Ctotal_0-100estim'], 3.5)
        self.assertEqual(result_df.loc[2, 'Ctotal_0-100estim'], 4.2)
        
        # Check source column updated
        self.assertEqual(result_df.loc[1, 'C_data_source'], 'SoilGrids backfill')
        self.assertEqual(result_df.loc[2, 'C_data_source'], 'SoilGrids backfill')
        self.assertEqual(result_df.loc[0, 'C_data_source'], 'Measured')
    
    @patch('soil_diskin.soilgrids_utils.initialize_earth_engine')
    @patch('soil_diskin.soilgrids_utils.get_soc_with_bulk_density')
    def test_backfill_with_failures(self, mock_get_soc, mock_init):
        """Test backfilling with some failures."""
        # Mock one success and one failure
        mock_get_soc.side_effect = [3.5, None]
        
        df = pd.DataFrame({
            'Latitude': [48.8566, 51.5074],
            'Longitude': [2.3522, -0.1278],
            'Ctotal_0-100estim': [None, None],
            'C_data_source': ['Measured', 'Measured']
        })
        
        result_df, stats = backfill_missing_soc(df, use_bulk_density=True)
        
        # Check stats
        self.assertEqual(stats['n_missing'], 2)
        self.assertEqual(stats['n_filled'], 1)
        self.assertEqual(stats['n_failed'], 1)
        self.assertEqual(stats['fill_rate'], 0.5)
        
        print(result_df)

        # Second row should be removed since we failed to fill it
        # and then filtered out missing data
        self.assertEqual(len(result_df), 1)

        # Check that the first value was filled
        self.assertEqual(result_df.loc[0, 'Ctotal_0-100estim'], 3.5)

    @patch('soil_diskin.soilgrids_utils.initialize_earth_engine')
    @patch('soil_diskin.soilgrids_utils.get_soc_at_point')
    @patch('soil_diskin.soilgrids_utils.calculate_total_soc_0_100')
    def test_backfill_without_bulk_density(self, mock_calc, mock_get_soc, mock_init):
        """Test backfilling without using bulk density."""
        mock_soc_dict = {'0-5cm': 15.0, '5-15cm': 12.0, '15-30cm': 10.0,
                        '30-60cm': 8.0, '60-100cm': 6.0}
        mock_get_soc.return_value = mock_soc_dict
        mock_calc.return_value = 3.5
        
        df = pd.DataFrame({
            'Latitude': [48.8566],
            'Longitude': [2.3522],
            'Ctotal_0-100estim': [None],
            'C_data_source': ['Measured']
        })
        
        result_df, stats = backfill_missing_soc(df, use_bulk_density=False)
        
        # Should have used the simple method
        mock_get_soc.assert_called_once()
        mock_calc.assert_called_once_with(mock_soc_dict)
        
        # Check result
        self.assertEqual(result_df.loc[0, 'Ctotal_0-100estim'], 3.5)
        self.assertEqual(stats['n_filled'], 1)


if __name__ == '__main__':
    unittest.main()
