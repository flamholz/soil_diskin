import unittest
import numpy as np
import pandas as pd
import xarray as xr

from scipy.io import loadmat
from soil_diskin.compartmental_models import CLM5
from soil_diskin.compartmental_models import ConfigParams, GlobalData


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
    def fix_lon(ds):
        ds['x'] = xr.where(ds['x']>=180,ds['x']-360,ds['x'])
        return ds.sortby('x')

    global_da = fix_lon(global_da)
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
    Gamma_soil = 1e-4 
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
        """Test that CLM5 can be instantiated."""
        model = CLM5(self.config, self.env_params)
        self.assertIsInstance(model, CLM5)
        self.assertIsNotNone(model.A)
        self.assertIsNotNone(model.V)
        self.assertIsNotNone(model.K_ts)


if __name__ == '__main__':
    unittest.main()
