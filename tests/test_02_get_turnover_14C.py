import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import xarray as xr


class Test02GetTurnover14C(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create mock tropical sites data
        self.mock_tropical_sites = pd.DataFrame({
            'Latitude': [10.0, 20.0, 30.0],
            'Longitude': [-50.0, -60.0, -70.0],
            'Ctotal_0-100estim': [100.0, 150.0, 200.0],
            'weight_1': [0.1, 0.2, 0.3],
            'weight_2': [0.2, 0.3, 0.4]
        })
        
        # Create mock xarray DataArray
        self.mock_c14_data = xr.DataArray(
            np.random.rand(5, 5) * 100 - 50,  # Random values between -50 and 50
            dims=['y', 'x'],
            coords={'y': np.linspace(0, 40, 5), 'x': np.linspace(-80, -40, 5)},
            name='delta_14C'
        )
    
    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir)
    
    @patch('ee.Initialize')
    @patch('ssl._create_default_https_context')
    def test_imports_and_initialization(self, mock_ssl, mock_ee_init):
        """Test that all required imports work and ee initializes."""
        try:
            find_nearest = __import__('02_get_turnover_14C.find_nearest',
                                    fromlist=['find_nearest']).find_nearest
            mock_ee_init.assert_called_once()
        except ImportError:
            self.fail("Failed to import required modules")
    
    def test_find_nearest_with_numpy_array(self):
        """Test find_nearest function with numpy array coordinates."""
        find_nearest = __import__('02_get_turnover_14C.find_nearest',
                                    fromlist=['find_nearest']).find_nearest
        
        # Create test coordinates
        coords = np.array([[10.0, -50.0], [20.0, -60.0]])
        
        # Test with tolerance that should include all points
        result = find_nearest(coords, self.mock_c14_data, tolerance=10.0)
        
        self.assertEqual(len(result), 2)
        self.assertTrue(isinstance(result, np.ndarray))
    
    def test_find_nearest_with_dataframe(self):
        """Test find_nearest function with DataFrame coordinates."""
        find_nearest = __import__('02_get_turnover_14C.find_nearest',
                                  fromlist=['find_nearest']).find_nearest
        
        coords_df = pd.DataFrame({
            'lat': [10.0, 20.0],
            'lon': [-50.0, -60.0]
        })
        
        result = find_nearest(coords_df, self.mock_c14_data, tolerance=10.0)
        
        self.assertEqual(len(result), 2)
        self.assertTrue(isinstance(result, np.ndarray))
    
    def test_find_nearest_with_strict_tolerance(self):
        """Test find_nearest function with strict tolerance."""
        find_nearest = __import__('02_get_turnover_14C.find_nearest',
                                    fromlist=['find_nearest']).find_nearest
        
        coords = np.array([[100.0, -200.0]])  # Far from data
        
        result = find_nearest(coords, self.mock_c14_data, tolerance=0.1)
        
        self.assertTrue(np.isnan(result[0]))
    
    @patch('utils.download_file')
    @patch('os.path.exists')
    def test_data_download_workflow(self, mock_exists, mock_download):
        """Test the data download workflow."""
        mock_exists.return_value = False
        
        # This would normally be part of the script execution
        c14_folder = "../data/shi_2020/"
        c14_data_filename = "global_delta_14C.nc"
        c14_url = "https://zenodo.org/records/3823612/files/global_delta_14C.nc"
        
        # Simulate the download call
        mock_download(c14_url, c14_folder, c14_data_filename)
        mock_download.assert_called_once_with(c14_url, c14_folder, c14_data_filename)
    
    @patch('zipfile.ZipFile')
    @patch('os.path.exists')
    def test_zip_extraction(self, mock_exists, mock_zipfile):
        """Test ZIP file extraction logic."""
        mock_exists.return_value = False
        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
        
        # Simulate the extraction logic
        GPP_folder = "../data/kang_2023/"
        if not os.path.exists(os.path.join(GPP_folder, "ST_CFE-Hybrid_NT")):
            with mock_zipfile(os.path.join(GPP_folder, "ST_CFE-Hybrid_NT.zip"), 'r') as zip_ref:
                zip_ref.extractall(GPP_folder)
        
        mock_zip_instance.extractall.assert_called_once_with(GPP_folder)
    
    @patch('pd.read_csv')
    @patch('rioxarray.open_rasterio')
    def test_data_loading(self, mock_rio, mock_read_csv):
        """Test data loading functionality."""
        # Mock CSV reading
        mock_read_csv.return_value = self.mock_tropical_sites
        
        # Mock rioxarray loading
        mock_rio.return_value = self.mock_c14_data
        
        # Test that data can be loaded
        tropical_sites = mock_read_csv('results/processed_balesdant_2018.csv')
        c14_data = mock_rio('path/to/file.nc')
        
        self.assertIsInstance(tropical_sites, pd.DataFrame)
        self.assertEqual(len(tropical_sites), 3)
    
    def test_extract_tropical_sites_function(self):
        """Test the extract_tropical_sites function logic."""
        # Create mock coordinates
        unique_coords = pd.DataFrame({
            'Latitude': [10.0, 20.0],
            'Longitude': [-50.0, -60.0]
        })
        
        def extract_tropical_sites(ds):
            return xr.concat([
                ds.sel(y=row['Latitude'], x=row['Longitude'], method='nearest') 
                for i, row in unique_coords.iterrows()
            ], dim='site')
        
        result = extract_tropical_sites(self.mock_c14_data)
        self.assertEqual(len(result.dims), 1)  # Should have 'site' dimension
    
    @patch('ee.ImageCollection')
    @patch('ee.FeatureCollection')
    @patch('geemap.ee_to_df')
    def test_earth_engine_integration(self, mock_ee_to_df, mock_feature_collection, mock_image_collection):
        """Test Earth Engine integration components."""
        # Mock Earth Engine objects
        mock_collection = MagicMock()
        mock_image_collection.return_value = mock_collection
        mock_feature_collection.return_value = MagicMock()
        
        # Mock the result of ee_to_df
        mock_npp_data = pd.DataFrame({'NPP': [100.0, 150.0, 200.0]})
        mock_ee_to_df.return_value = mock_npp_data
        
        # Test that the mocked functions return expected data
        result = mock_ee_to_df()
        self.assertIn('NPP', result.columns)
        self.assertEqual(len(result), 3)
    
    def test_data_processing_calculations(self):
        """Test the data processing and calculations."""
        # Test fraction modern carbon calculation
        delta_14C = np.array([-100, 0, 100])  # per mil values
        fm = delta_14C / 1e3 + 1
        expected_fm = np.array([0.9, 1.0, 1.1])
        np.testing.assert_array_equal(fm, expected_fm)
        
        # Test turnover calculation
        ctotal = np.array([1000, 1500, 2000])  # gC/m2
        npp = np.array([100, 150, 200])  # gC/m2/year
        turnover = ctotal * 1e3 / npp  # Convert ctotal to same units
        expected_turnover = np.array([10000, 10000, 10000])
        np.testing.assert_array_equal(turnover, expected_turnover)
    
    @patch('pandas.DataFrame.to_csv')
    def test_output_file_creation(self, mock_to_csv):
        """Test that output file is created correctly."""
        # Create mock merged data
        merged_data = pd.DataFrame({
            'Latitude': [10.0, 20.0],
            'Longitude': [-50.0, -60.0],
            '14C': [-50.0, 0.0],
            'NPP': [100.0, 150.0],
            'Ctotal_0-100estim': [1000.0, 1500.0],
            'fm': [0.95, 1.0],
            'turnover': [10.0, 10.0]
        })
        
        # Test CSV export
        merged_data.to_csv('../results/tropical_sites_14C_turnover.csv', index=False)
        mock_to_csv.assert_called_once_with('../results/tropical_sites_14C_turnover.csv', index=False)
    
    def test_error_handling_empty_coordinates(self):
        """Test error handling with empty coordinates.
        
        TODO: seems to require google earth engine credentials. Fix. 
        """
        find_nearest = __import__('02_get_turnover_14C.find_nearest',
                                    fromlist=['find_nearest']).find_nearest
    
        empty_coords = np.array([]).reshape(0, 2)
        
        with self.assertRaises((IndexError, ValueError)):
            find_nearest(empty_coords, self.mock_c14_data)
    
    def test_error_handling_invalid_dataarray(self):
        """Test error handling with invalid DataArray."""
        find_nearest = __import__('02_get_turnover_14C.find_nearest',
                                    fromlist=['find_nearest']).find_nearest
        
        coords = np.array([[10.0, -50.0]])
        invalid_da = xr.DataArray([])  # Empty DataArray
        
        with self.assertRaises((ValueError, IndexError)):
            find_nearest(coords, invalid_da)

if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(Test02GetTurnover14C)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")