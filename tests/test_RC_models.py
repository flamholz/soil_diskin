import unittest
import numpy as np
import pandas as pd
import xarray as xr

from collections import namedtuple
from soil_diskin.compartmental_models import ReducedComplexModel
from soil_diskin.data_wrangling import parse_he_data


# The parameterization of this class is quite obtuse, as evident
# from my need to copy this code from the script running the RC models.
# TODO: more clearly document and simplify the interface for this class.
def make_RC_params():
    # Load the site data
    site_data = pd.read_csv('results/processed_balesdent_2018.csv')
    turnover_14C = pd.read_csv('results/all_sites_14C_turnover.csv')

    #%% Create reduced complexity model predictions
    print("Generating reduced complexity model predictions...")
    file_names = {'CESM':'areacella_fx_CESM1-BGC_1pctCO2_r0i0p0',
                'IPSL':'areacella_fx_IPSL-CM5A-LR_historicalNat_r0i0p0',
                'MRI':'areacella_fx_MRI-ESM1_esmControl_r0i0p0'}

    model = 'CESM'
    mod = parse_he_data(model=model, file_names=file_names)
    extrapolated =  xr.concat([mod.sel(parameter=p).rio.write_nodata(np.nan).rio.set_spatial_dims('lon','lat').rio.write_crs(4326).rio.interpolate_na() for p in mod.parameter],dim='parameter')
    my_params = xr.concat([extrapolated.sel(
        lat=site_data.iloc[i]['Latitude'],
        lon=site_data.iloc[i]['Longitude'],
        method='ffill') for i in range(site_data.shape[0])], dim='site')
    return my_params.sel(site=0).values


class TestReducedComplexModel(unittest.TestCase):
    """Test suite for ReducedComplexModel compartmental model."""

    def setUp(self):
        """Set up minimal config and params for testing."""
        RCMConfig = namedtuple('RCMConfig', ['model', 'rs_fac', 'tau_fac', 'correct'])
        self.config = RCMConfig(
            model='CESM',
            rs_fac={'CESM': 1.0, 'IPSL': 1.0, 'MRI': 1.0},
            tau_fac={'CESM': 1.0, 'IPSL': 1.0, 'MRI': 1.0},
            correct=True
        )
        
        RCMParams = namedtuple('RCMParams', ['params'])
        self.params = RCMParams(
            params=make_RC_params()
        )

    def test_initialization(self):
        """Test that ReducedComplexModel can be instantiated."""
        model = ReducedComplexModel(self.config, self.params)
        self.assertIsInstance(model, ReducedComplexModel)
        self.assertIsNotNone(model.A)
        self.assertIsNotNone(model.u)


if __name__ == '__main__':
    unittest.main()
