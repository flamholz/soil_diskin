import unittest
import numpy as np
import pandas as pd
from collections import namedtuple
from soil_diskin.compartmental_models import JSBACH
from soil_diskin.constants import DAYS_PER_YEAR
import subprocess

class TestJSBACH(unittest.TestCase):
    """Test suite for JSBACH compartmental model."""

    def setUp(self):
        """Set up minimal config and environment data for testing."""
        JSBACHConfig = namedtuple('JSBACHConfig', ['placeholder'])
        self.config = JSBACHConfig(placeholder=True)
        
        JSBACHEnv = namedtuple('JSBACHEnv', ['I', 'T', 'P', 'd'])
        self.env_params = JSBACHEnv(
            I=np.ones(12),
            T=np.random.rand(12) * 20 + 10,  # Temperature 10-30°C
            P=np.random.rand(12) * 100 + 50,  # Precipitation 50-150mm
            d=0.1  # CWD diameter
        )

        # compile and run a test version of the Fortran JSBACH model. The code for this example run is in the testing_src directory
        subprocess.call('tests/test_data/jsbach/testing_src/run_yasso_test.sh')

    def test_initialization(self):
        """Test that JSBACH can be instantiated."""
        model = JSBACH(self.config, self.env_params)
        self.assertIsInstance(model, JSBACH)
        self.assertIsNotNone(model.A)
        self.assertIsNotNone(model.K)
        self.assertIsNotNone(model.u)

    def test_comparison_with_fortran(self):
        """Test JSBACH output against Fortran implementation."""
        # Parameters from JSBACH documentation in 
        # from https://pure.mpg.de/rest/items/item_3279802_26/component/file_3316522/content#page=107.51 and 
        # https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/src/carbon/mo_carbon_process.f90
        a_i = np.array([0.72, 5.9, 0.28, 0.031])
        a_h = 0.0016
        b1 = 9.5e-2
        b2 = -1.4e-3
        gamma = -1.21
        phi1 = -1.71
        phi2 = 0.86
        r = -0.306

        # Create JSBACH config
        JSBACH_config = namedtuple('Config', ['a_i', 'a_h', 'b1', 'b2', 'gamma', 'phi1', 'phi2', 'r'])
        config = JSBACH_config(a_i = a_i, a_h = a_h, b1 = b1, b2 = b2, gamma = gamma, phi1 = phi1, phi2 = phi2, r = r)
        
        # Create environment parameters
        one_vec = np.ones(12)
        JSBACH_env_params = namedtuple('EnvParams', ['I', 'T', 'P', 'd'])
        env_params = JSBACH_env_params(one_vec * DAYS_PER_YEAR, 25 * one_vec, one_vec, 4)

        # Instantiate model and compute output
        model = JSBACH(config=config, env_params=env_params)
        output = model._dX(t=0, X=np.ones(18))[:9]

        # Run the Fortran implementation
        
        print('Running Fortran JSBACH model for comparison...')
        # Load Fortran output
        fortran_output = pd.read_csv('tests/test_data/jsbach/yasso_output.csv')
        
        # Assert outputs are almost equal
        expected = output * (1/DAYS_PER_YEAR) + np.ones(9)
        actual = fortran_output['Value'].values[:9]
        for i in range(len(expected)):
            self.assertAlmostEqual(actual[i], expected[i], places=3)