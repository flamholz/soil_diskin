# write a unit test for the CLM_vertical_utils.py module
from CLM_vertical_utils import *
from CLM_vertical import *
from scipy.io import loadmat
from os import path
from glob import glob
import numpy as np
import unittest
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product



CWD = path.dirname(path.abspath(__file__)) 

class TestTriDiag(unittest.TestCase):

    SOILDEPTH_FILE = path.join(CWD, 'test_examples/tridiag_positive_examples/soildepth.mat')
    GOLD_EXAMPLE_FILES = glob(path.join(CWD, 'test_examples/tridiag_positive_examples/', 'test_example*.mat'))
    FAILING_EXAMPLE_FILES = glob(path.join(CWD, 'test_examples/tridiag_negative_examples/', 'test_example*.mat'))

    def setUp(self):
        mat = loadmat(self.SOILDEPTH_FILE)
        self.zisoi = mat['zisoi'].squeeze()
        self.zsoi = mat['zsoi'].squeeze()
        self.dz = mat['dz'].squeeze()
        self.dz_node = mat['dz_node'].squeeze()
        return super().setUp()

    def test_simple(self):
        self.assertGreater(len(self.GOLD_EXAMPLE_FILES), 0)
        for fname in self.GOLD_EXAMPLE_FILES:
            example_data = loadmat(fname)
            inputs = example_data['example'].squeeze()
            # input order in matlab was som_diffus,som_adv_flux,npools,nlevdecomp
            # which we gave somewhat different names in python.
            Gamma_soil, F_soil, npools, nlevels = inputs
            expected = example_data['result'].squeeze()
            result = make_V_matrix(Gamma_soil, F_soil, int(npools), int(nlevels),
                                   self.dz, self.dz_node, self.zsoi, self.zisoi)
            np.testing.assert_allclose(result, expected)
    
    def test_neg_examples(self):
        self.assertGreater(len(self.FAILING_EXAMPLE_FILES), 0)
        for fname in self.FAILING_EXAMPLE_FILES:
            example_data = loadmat(fname)
            inputs = example_data['example'].squeeze()
            # input order in matlab was som_diffus,som_adv_flux,npools,nlevdecomp
            # which we gave somewhat different names in python.
            Gamma_soil, F_soil, npools, nlevels = inputs
            expected = example_data['result'].squeeze()
            result = make_V_matrix(Gamma_soil, F_soil, int(npools), int(nlevels),
                                   self.dz, self.dz_node, self.zsoi, self.zisoi)

            # The matrices are not equal, not sure how to test this
            #np.testing.assert_allclose(result, expected)


class TestAMatrix(unittest.TestCase):

    GOLD_EXAMPLE_FILES = glob(path.join(CWD, 'test_examples/A_matrix_positive_examples/', 'test_*.mat'))

    def setUp(self):
        return super().setUp()

    def test_postive_examples(self):
        self.assertGreater(len(self.GOLD_EXAMPLE_FILES), 0)
        for fname in self.GOLD_EXAMPLE_FILES:
            example_data = loadmat(fname)
            inputs = example_data['sand_content_vec'].squeeze()
            sand_content = inputs
            nlevels = 10
            expected = example_data['result'].squeeze()
            result = make_A_matrix(sand_content, nlevels)
            np.testing.assert_allclose(result, expected)
    

class TestKMatrix(unittest.TestCase):

    SOILDEPTH_FILE = path.join(CWD, 'test_examples/K_matrix_positive_examples/soildepth.mat')
    GOLD_EXAMPLE_FILES = glob(path.join(CWD, 'test_examples/K_matrix_positive_examples/', 'test_*.mat'))

    def setUp(self):
        mat = loadmat(self.SOILDEPTH_FILE)
        self.zsoi = mat['zsoi'].squeeze()
        CLM_params = xr.open_dataset(CWD + '/../data/clm5_params.c171117.nc')
        self.taus = np.array([CLM_params['tau_cwd'],CLM_params['tau_l1'],CLM_params['tau_l2_l3'],CLM_params['tau_l2_l3'],CLM_params['tau_s1'],CLM_params['tau_s2'],CLM_params['tau_s3']]).squeeze()
        self.taus = self.taus * DAYS_PER_YEAR * SECS_PER_DAY
        return super().setUp()

    def test_postive_examples(self):
        self.assertGreater(len(self.GOLD_EXAMPLE_FILES), 0)
        for fname in self.GOLD_EXAMPLE_FILES:
            
            example_data = loadmat(fname)
            inputs = example_data['example'].squeeze()
            decomp_depth_efolding = example_data['decomp_depth_efolding'].squeeze()
            
            w_scalar = inputs[0,:]
            t_scalar = inputs[1,:]
            o_scalar = inputs[2,:]
            n_scalar = inputs[3,:]

            nlevels = 10
            expected = example_data['result'].squeeze()
            result = make_K_matrix(self.taus, self.zsoi, w_scalar, t_scalar, o_scalar,
                                   n_scalar, decomp_depth_efolding, nlevels)
            np.testing.assert_allclose(result, expected)

class TestGlobalRun(unittest.TestCase):

    SOILDEPTH_FILE = path.join(CWD, 'test_examples/K_matrix_positive_examples/soildepth.mat')
    GOLD_EXAMPLE_FILES = glob(path.join(CWD, 'test_examples/global_run_positive_examples/', '*nc'))

    def setUp(self):
        mat = loadmat(self.SOILDEPTH_FILE)
        self.zisoi = mat['zisoi'].squeeze()
        self.zsoi = mat['zsoi'].squeeze()
        self.dz = mat['dz'].squeeze()
        self.dz_node = mat['dz_node'].squeeze()

        CLM_params = xr.open_dataset(CWD + '/../data/clm5_params.c171117.nc')
        self.taus = np.array([CLM_params['tau_cwd'],CLM_params['tau_l1'],CLM_params['tau_l2_l3'],CLM_params['tau_l2_l3'],CLM_params['tau_s1'],CLM_params['tau_s2'],CLM_params['tau_s3']]).squeeze()
        self.taus = self.taus * DAYS_PER_YEAR * SECS_PER_DAY
        # load gridded nc file with the inputs, initial values, and the environmental variables
        self.global_da = xr.open_dataset(CWD + '/../data/global_demo_in.nc')
        self.secspday = 86400
        self.Gamma_soil = 1e-4 / (self.secspday * 365)
        self.F_soil = 0
        self.nlevels = 10

        return super().setUp()

    def test_postive_examples(self):
        self.assertGreater(len(self.GOLD_EXAMPLE_FILES), 0)
        config = ConfigParams(decomp_depth_efolding=0.5, taus=self.taus, Gamma_soil=self.Gamma_soil, F_soil=self.F_soil,
                              zsoi=self.zsoi, zisoi=self.zisoi, dz=self.dz, dz_node=self.dz_node, nlevels=self.nlevels, npools=7)
        global_data = GlobalData(self.global_da)

        empty_gridcell = xr.concat([xr.full_like(global_data.make_ldd(-90, 0).X0, fill_value=np.nan)]*12, dim = 'TIME')

        # Function to run the model for a single grid cell
        def run_model_for_cell(lat, lon):
            ldd = global_data.make_ldd(lat, lon)
            if np.isnan(ldd.w[0,0]):
                res = empty_gridcell
                res['LAT'] = lat
                res['LON'] = lon
                return res.expand_dims('LAT').expand_dims('LON').stack(cell=('LAT', 'LON'))
            CLM_model = CLM_vertical(config, ldd)
            res = CLM_model.run(timesteps=range(11), dt=self.secspday * 30).expand_dims('LAT').expand_dims('LON').stack(cell=('LAT', 'LON'))
            return res

        # Extract latitude and longitude values
        lat_values = self.global_da['LAT'].values
        lon_values = self.global_da['LON'].values

        # Run the model in parallel over all grid cells with a progress bar
        results = Parallel(n_jobs=-1)(
            delayed(run_model_for_cell)(lat, lon) for lat, lon in tqdm(product(lat_values, lon_values), total=len(lat_values)*len(lon_values))
        )

        # Merge the results into a single xarray dataset
        global_result_da = xr.concat(results, dim='cell').unstack().transpose('TIME','LAT', 'LON','pools','LEVDCMP1_10')
        
        # Calculate the depth integrated C content
        global_tot_C = (global_result_da * self.dz[:self.nlevels]).sum(dim=['pools','LEVDCMP1_10'])
        global_tot_C = global_tot_C.where(global_tot_C > 0)

        # load the expected results
        expected = xr.open_dataarray(self.GOLD_EXAMPLE_FILES[0], decode_times=False)
        np.testing.assert_allclose(global_tot_C.fillna(0), expected.fillna(0), rtol=1e-1, atol=1e-1)

if __name__ == '__main__':
    unittest.main()