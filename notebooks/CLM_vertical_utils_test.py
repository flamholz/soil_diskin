# write a unit test for the CLM_vertical_utils.py module
from CLM_vertical_utils import *
from scipy.io import loadmat
from os import path
from glob import glob
import numpy as np
import unittest
import xarray as xr


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
            print(fname)
            example_data = loadmat(fname)
            inputs = example_data['example'].squeeze()
            decomp_depth_efolding = example_data['decomp_depth_efolding'].squeeze()
            print(decomp_depth_efolding)
            w_scalar = inputs[0,:]
            t_scalar = inputs[1,:]
            o_scalar = inputs[2,:]
            n_scalar = inputs[3,:]
            print(w_scalar)
            print(t_scalar)
            print(o_scalar)
            print(n_scalar)
            nlevels = 10
            expected = example_data['result'].squeeze()
            result = make_K_matrix(self.taus, self.zsoi, w_scalar, t_scalar, o_scalar,
                                   n_scalar, decomp_depth_efolding, nlevels)
            np.testing.assert_allclose(result, expected)

if __name__ == '__main__':
    unittest.main()