import unittest
import numpy as np

from collections import namedtuple
from soil_diskin.compartmental_models import JSBACH


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

    def test_initialization(self):
        """Test that JSBACH can be instantiated."""
        model = JSBACH(self.config, self.env_params)
        self.assertIsInstance(model, JSBACH)
        self.assertIsNotNone(model.A)
        self.assertIsNotNone(model.K)
        self.assertIsNotNone(model.u)