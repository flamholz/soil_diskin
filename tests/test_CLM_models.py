from glob import glob
from os import path
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from scipy.io import loadmat
from soil_diskin.compartmental_models import CLM5
from soil_diskin.compartmental_models import ConfigParams, GlobalData
from soil_diskin.constants import DAYS_PER_YEAR, SECS_PER_DAY
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm

# Helper function to run the model for a single grid cell
# Defined at module level to be picklable for joblib.Parallel
def run_model_for_cell(lat, lon, config, global_data, empty_gridcell):
    """Run CLM5 model for a single grid cell."""
    ldd = global_data.make_ldd(lat, lon)
    if np.isnan(ldd.w[0,0]):
        res = empty_gridcell.copy()
        res['y'] = lat
        res['x'] = lon
        return res.expand_dims('y').expand_dims('x').stack(cell=('y', 'x'))
    CLM_model = CLM5(config, ldd)
    CLM_model.I = CLM_model.I / DAYS_PER_YEAR / SECS_PER_DAY  # Convert inputs to per second
    res = CLM_model.run(timesteps=range(11), dt= SECS_PER_DAY * 30, tres='M').expand_dims('y').expand_dims('x').stack(cell=('y', 'x'))
    return res


# Helper function to create CLM5 config and global data
# TODO: the configuration and parameterization of this class is 
# quite obtuse as evidenced by my copying this function from 
# from the script that runs CLM5. Simplify and better document
# the configuration and parameterization of this class.
def make_CLM_config():
    fn = 'data/CLM5_global_simulation/soildepth.mat'
    mat = loadmat(fn)
    zisoi = mat['zisoi'].squeeze()
    zsoi = mat['zsoi'].squeeze()
    dz = mat['dz'].squeeze()
    dz_node = mat['dz_node'].squeeze()

    # load gridded nc file with the inputs, initial values, and the environmental variables
    global_da = xr.open_dataset('data/CLM5_global_simulation/global_demo_in.nc')
    global_da = global_da.rename({'LON':'x','LAT':'y'})
    # def fix_lon(ds):
    #     ds['x'] = xr.where(ds['x']>=180,ds['x']-360,ds['x'])
    #     return ds.sortby('x')

    # global_da = fix_lon(global_da)
    global_da = global_da.rio.write_crs("EPSG:4326", inplace=True)

    # define model parameters
    CLM_params = xr.open_dataset('data/CLM5_global_simulation/clm5_params.c171117.nc')
    taus = np.array([CLM_params['tau_cwd'],
                    CLM_params['tau_l1'],
                    CLM_params['tau_l2_l3'],
                    CLM_params['tau_l2_l3'],
                    CLM_params['tau_s1'],
                    CLM_params['tau_s2'],
                    CLM_params['tau_s3']]).squeeze()
    
    taus = taus * DAYS_PER_YEAR * SECS_PER_DAY # Rates are given per year, we convert to per second
    Gamma_soil = 1e-4 / (SECS_PER_DAY * DAYS_PER_YEAR) #TODO: test this
    F_soil = 0

    # create global configuration parameters
    config = ConfigParams(decomp_depth_efolding=0.5, taus=taus, Gamma_soil=Gamma_soil, F_soil=F_soil,
                        zsoi=zsoi, zisoi=zisoi, dz=dz, dz_node=dz_node, nlevels=10, npools=7)
    global_data = GlobalData(global_da)

    return config, global_data


class TestCLM5(unittest.TestCase):
    """Test suite for CLM5 compartmental model."""

    def setUp(self):
        """Set up config and environment data for testing."""
        self.config, self.global_data = make_CLM_config()
        lat, lng = -26.58333333, 151.83333333333334
        self.env_params = self.global_data.make_ldd(lat, lng)

    def test_initialization(self):
        """Test that CLM4.5 can be instantiated."""
        model = CLM5(self.config, self.env_params)
        self.assertIsInstance(model, CLM5)
        self.assertIsNotNone(model.A)
        self.assertIsNotNone(model.V)
        self.assertIsNotNone(model.K_ts)

    def test_tri_diag_matrix(self):
        """Test CLM4.5 tri-diagonal matrix construction against gold standard examples and failing examples."""
        
        SOILDEPTH_FILE = path.join('tests/test_data/CLM45/tridiag_positive_examples/soildepth.mat')
        GOLD_EXAMPLE_FILES = glob(path.join('tests/test_data/CLM45/tridiag_positive_examples/', 'test_example*.mat'))
        FAILING_EXAMPLE_FILES = glob(path.join('tests/test_data/CLM45/tridiag_negative_examples/', 'test_example*.mat'))
        
        # test that we have some example files
        self.assertGreater(len(GOLD_EXAMPLE_FILES), 0)

        def test_file(fname):
            """Test a single file."""
            # load example data
            example_data = loadmat(fname)

            # get example parameters and create the V matrix
            Gamma_soil, F_soil, npools, nlevels = example_data['example'].squeeze()
            result = CLM5.make_V_matrix(Gamma_soil, F_soil, int(npools), int(nlevels),
                    self.config.dz, self.config.dz_node, self.config.zsoi, self.config.zisoi)

            # get the expected result from the matlab code
            expected = example_data['result'].squeeze()
            
            return result, expected
        
        # run test for gold standard examples
        for fname in GOLD_EXAMPLE_FILES:
            result, expected = test_file(fname)
            np.testing.assert_allclose(result, expected)

        # run test for failing examples
        for fname in FAILING_EXAMPLE_FILES:
            result, expected = test_file(fname)
            np.testing.assert_raises(AssertionError, np.testing.assert_allclose, result, expected)
    

    def test_A_matrix(self):
        """Test CLM4.5 A matrix construction against gold standard examples."""
        
        GOLD_EXAMPLE_FILES = glob(path.join('tests/test_data/CLM45/A_matrix_positive_examples/', 'test_*.mat'))
        
        # test that we have some example files
        self.assertGreater(len(GOLD_EXAMPLE_FILES), 0)

        for fname in GOLD_EXAMPLE_FILES:
            # load example data
            example_data = loadmat(fname)

            # get example parameters and create the A matrix
            sand_content = example_data['sand_content_vec'].squeeze()
            result = CLM5.make_A_matrix(sand_content, self.config.nlevels)

            # get the expected result from the matlab code
            expected = example_data['result'].squeeze()
            
            np.testing.assert_allclose(result, expected)

    def test_K_matrix(self):
        """Test CLM4.5 K matrix construction against gold standard examples."""
        
        SOILDEPTH_FILE = path.join('tests/test_data/CLM45/K_matrix_positive_examples/soildepth.mat')
        GOLD_EXAMPLE_FILES = glob(path.join('tests/test_data/CLM45/K_matrix_positive_examples/', 'test_*.mat'))
        
        # test that we have some example files
        self.assertGreater(len(GOLD_EXAMPLE_FILES), 0)

        for fname in GOLD_EXAMPLE_FILES:
            # load example data
            example_data = loadmat(fname)

            # get example parameters and create the K matrix
            inputs = example_data['example'].squeeze()
            decomp_depth_efolding = example_data['decomp_depth_efolding'].squeeze()
            
            w_scalar = inputs[0,:]
            t_scalar = inputs[1,:]
            o_scalar = inputs[2,:]
            n_scalar = inputs[3,:]

            # taus = self.config.taus * DAYS_PER_YEAR * SECS_PER_DAY #TODO: I'm not sure why I need to multiply by these constants here. But this works and if I don't do this the test fails.
            result = CLM5.make_K_matrix(self.config.taus, self.config.zsoi, w_scalar, t_scalar, o_scalar,
                                   n_scalar, decomp_depth_efolding, self.config.nlevels)

            # get the expected result from the matlab code
            expected = example_data['result'].squeeze()
            
            np.testing.assert_allclose(result, expected)
            
    def test_global_run(self):
        """Test a global run of CLM4.5 against gold standard netCDF output."""
     

        GOLD_EXAMPLE_FILES = glob(path.join('tests/test_data/CLM45/global_run_positive_examples/', '*nc'))

        self.assertGreater(len(GOLD_EXAMPLE_FILES), 0)

        # load the expected results
        expected = xr.open_dataarray(GOLD_EXAMPLE_FILES[0], decode_times=False)
        df = expected.sum(dim='time').to_dataframe()
        df = df[df>0].dropna()
        latlons = df.index.tolist()



        empty_gridcell = xr.concat([xr.full_like(self.global_data.make_ldd(-90, 0).X0, fill_value=np.nan)]*12, dim = 'TIME')

        # Run the model in parallel over all grid cells with a progress bar
        # Using module-level function for picklability
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_model_for_cell)(lat, lon, self.config, self.global_data, empty_gridcell) 
            for lat, lon in tqdm(latlons)
        )

        # Merge the results into a single xarray dataset
        global_result_da = xr.concat(results, dim='cell').unstack().transpose('TIME','y', 'x','pools','LEVDCMP1_10')
        
        # Calculate the depth integrated C content
        global_tot_C = (global_result_da * self.config.dz[:self.config.nlevels]).sum(dim=['pools','LEVDCMP1_10'])
        global_tot_C = global_tot_C.where(global_tot_C > 0)

        for i, (lat, lon) in enumerate(latlons):
            try:
                np.testing.assert_allclose(global_tot_C.sel(y=lat, x=lon).fillna(0), expected.sel(latitude = lat, longitude = lon).fillna(0), rtol=1e-2, atol=1e-1)
            except AssertionError as e:
                print(f"Assertion failed for cell {i} lat: {lat}, lon: {lon}")
                raise e

if __name__ == '__main__':
    unittest.main()

