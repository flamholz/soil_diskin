import itertools
import unittest
import numpy as np

from soil_diskin.models import PowerLawDisKin, GammaDisKin
from soil_diskin.models import GeneralPowerLawDisKin, LognormalDisKin
from soil_diskin.constants import GAMMA

model_classes = [PowerLawDisKin, GammaDisKin,
                 GeneralPowerLawDisKin, LognormalDisKin]

class TestImplementsAbstractMethods(unittest.TestCase):
    def test_implements_abstract_methods(self):
        for model_class in model_classes:
            param_1, param_2 = 1.1, 0.8
            instance = model_class(param_1, param_2)

            self.assertIsNotNone(instance.A)
            self.assertIsNotNone(instance.T)

            try:
                instance.params_valid()
            except NotImplementedError as e:
                self.fail(f"{model_class.__name__} does not implement params_valid: {e}")

            # All these methods have a single numeric argument
            methods2call = 's,pA,cdfA'.split(',')
            for method in methods2call:
                try:
                    getattr(instance, method)(1)
                except NotImplementedError as e:
                    self.fail(f"{model_class.__name__} does not implement {method}: {e}")

class TestPowerLawDisKin(unittest.TestCase):

    T_MINS = np.logspace(0, 2, 3)
    T_MAXS = np.logspace(2, 4, 3)

    # Values calculated in Mathematica by Yinon Bar-On
    CHECK_VALS = [
        {'tau_min': 1, 'tau_max': 10000, 't': 1000,
         'pA_expected': 0.000104694,
         'R_14C_expected': 0.987,
         'R_hat_expected': 0.997303},
        {'tau_min': 2.4, 'tau_max': 35000, 't': 1000,
         'pA_expected': 0.00010759, 
         'R_14C_expected': 0.987,
         'R_hat_expected': 0.893637},
    ]

    def test_numerical_T_integral(self):
        for t_min, t_max in itertools.product(self.T_MINS, self.T_MAXS):
            pow_diskin = PowerLawDisKin(t_min, t_max)
            T_int = pow_diskin.calc_transit_time()
            T_calc = pow_diskin.T
            print('Calcualted transit time: ', T_calc)
            print('Integrated transit time: ', T_int[0])

            A_calc = pow_diskin.A
            A_int = pow_diskin.calc_mean_age()
            print('Calculated mean age: ', A_calc)
            print('Integrated mean age: ', A_int[0])

    def test_pdf_properties(self):
        """Test that the age distribution PDF has valid properties."""
        for t_min, t_max in itertools.product(self.T_MINS, self.T_MAXS):
            pow_diskin = PowerLawDisKin(t_min, t_max)
            
            # PDF should be non-negative
            test_ages = np.logspace(0, 3, 20)
            for age in test_ages:
                pA = pow_diskin.pA(age)
                self.assertGreaterEqual(
                    pA, 0, msg=f"PDF is negative at age={age} for t_min={t_min}, t_max={t_max}")
            
            # s(t) should be decreasing
            s_vals = [pow_diskin.s(t) for t in test_ages]
            for i in range(len(s_vals) - 1):
                self.assertGreaterEqual(
                    s_vals[i], s_vals[i+1],
                    msg=f"s(t) is not decreasing for t_min={t_min}, t_max={t_max}")

    def test_cdf_properties(self):
        """Test that the CDF has valid properties."""
        for t_min, t_max in itertools.product(self.T_MINS, self.T_MAXS):
            pow_diskin = PowerLawDisKin(t_min, t_max)
            
            test_ages = np.logspace(0, 3, 20)
            cdf_vals = [pow_diskin.cdfA(t) for t in test_ages]
            
            # CDF should be non-decreasing
            for i in range(len(cdf_vals) - 1):
                self.assertLessEqual(cdf_vals[i], cdf_vals[i+1],
                                   msg=f"CDF is not increasing for t_min={t_min}, t_max={t_max}")
            
            # CDF should be between 0 and 1
            for i, cdf in enumerate(cdf_vals):
                self.assertGreaterEqual(cdf, 0,
                                      msg=f"CDF < 0 at age={test_ages[i]} for t_min={t_min}, t_max={t_max}")
                self.assertLessEqual(cdf, 1.1,  # Allow small numerical error
                                   msg=f"CDF > 1 at age={test_ages[i]} for t_min={t_min}, t_max={t_max}")

    # Make sure we run the radiocarbon age integral without error
    def test_radiocarbon_ratio(self):
        for t_min, t_max in itertools.product(self.T_MINS, self.T_MAXS):
            pow_diskin = PowerLawDisKin(t_min, t_max)
            rc_ratio = pow_diskin.calc_radiocarbon_ratio_ss()
            print('Radiocarbon ratio at 1000 yr: ', rc_ratio)
    
    # Make sure we can construct from mean age and transit time
    def test_from_age_and_transit_time(self):
        for t_min, t_max in itertools.product(self.T_MINS, self.T_MAXS):
            pow_diskin = PowerLawDisKin(t_min, t_max)
            times = np.arange(1000)
            inputs = np.ones_like(times)
            pow_diskin.run_simulation(times, inputs)

    def test_check_vals(self):
        for check_vals in self.CHECK_VALS:
            tau_min = check_vals['tau_min']
            tau_max = check_vals['tau_max']
            pow_diskin = PowerLawDisKin(tau_min, tau_max)
            t = check_vals['t']
            pA = pow_diskin.pA(t)
            R_hat = pow_diskin.calc_radiocarbon_ratio_ss()

            pA_exp = check_vals['pA_expected']
            self.assertAlmostEqual(pA, pA_exp, places=2,
                                   msg=f"pA({t}) = {pA}, expected {pA_exp}")

            R_hat_exp = check_vals['R_hat_expected']
            self.assertAlmostEqual(R_hat[0], R_hat_exp, places=2,
                                   msg=f"R_hat = {R_hat[0]}, expected {R_hat_exp}")


class TestGammaDisKin(unittest.TestCase):

    A_VALS = [2.5, 5.0, 10.0]  # shape parameter values
    B_VALS = [0.01, 0.1, 1.0]  # scale parameter values

    def test_basic_properties(self):
        """Test that basic properties of the model are calculated correctly."""
        for a, b in itertools.product(self.A_VALS, self.B_VALS):
            model = GammaDisKin(a, b)
            
            # Check that transit time is calculated
            T = model.T
            T_expected = 1 / ((a - 1) * b)
            self.assertAlmostEqual(T, T_expected, places=6,
                                   msg=f"Transit time mismatch for a={a}, b={b}")
            
            # Check that mean age is calculated correctly
            A_expected = 1 / ((a - 2) * (a - 1) * b**2) / T
            A_numeric, _ = model.calc_mean_age()
            A_analytical = model.A
            
            self.assertAlmostEqual(
                A_numeric, A_expected, places=6,
                msg=f"Numeric mean age mismatch for a={a}, b={b}")
            self.assertAlmostEqual(
                A_analytical, A_expected, places=6,
                msg=f"Analytic mean age mismatch for a={a}, b={b}")

    def test_pdf_properties(self):
        """Test that the age distribution PDF has valid properties."""
        for a, b in itertools.product(self.A_VALS, self.B_VALS):
            model = GammaDisKin(a, b)
            
            # PDF should be non-negative
            test_ages = np.logspace(0, 3, 20)
            for age in test_ages:
                pA = model.pA(age)
                self.assertGreaterEqual(
                    pA, 0, msg=f"PDF is negative at age={age} for a={a}, b={b}")
            
            # s(t) should be decreasing
            s_vals = [model.s(t) for t in test_ages]
            for i in range(len(s_vals) - 1):
                self.assertGreaterEqual(
                    s_vals[i], s_vals[i+1],
                    msg=f"s(t) is not decreasing for a={a}, b={b}")

    def test_cdf_properties(self):
        """Test that the CDF has valid properties."""
        for a, b in itertools.product(self.A_VALS, self.B_VALS):
            model = GammaDisKin(a, b)
            
            test_ages = np.logspace(0, 3, 20)
            cdf_vals = [model.cdfA(t) for t in test_ages]
            
            # CDF should be non-decreasing
            for i in range(len(cdf_vals) - 1):
                self.assertLessEqual(cdf_vals[i], cdf_vals[i+1],
                                   msg=f"CDF is not increasing for a={a}, b={b}")
            
            # CDF should be between 0 and 1
            for i, cdf in enumerate(cdf_vals):
                self.assertGreaterEqual(cdf, 0,
                                      msg=f"CDF < 0 at age={test_ages[i]} for a={a}, b={b}")
                self.assertLessEqual(cdf, 1.1,  # Allow small numerical error
                                   msg=f"CDF > 1 at age={test_ages[i]} for a={a}, b={b}")

    def test_params_valid(self):
        """Test parameter combinations.
        
        For the gamma distribution, shape (a) and scale (b) must be positive.
        """
        valid_params = [(2.5, 0.1), (5.0, 1.0), (10.0, 0.01)]
        for a, b in valid_params:
            model = GammaDisKin(a, b)
            self.assertTrue(model.params_valid(),
                           msg=f"Valid parameters a={a}, b={b} marked as invalid.")
        
        invalid_params = [(-1.0, 0.1), (5.0, -0.5), (0.0, 1.0), (-0.1, -0.2)]
        for a, b in invalid_params:
            model = GammaDisKin(a, b)
            self.assertFalse(model.params_valid(),
                            msg=f"Invalid parameters a={a}, b={b} marked as valid.")

    def test_radiocarbon_ratio(self):
        """Test that the radiocarbon ratio calculation runs without error."""
        for a, b in itertools.product(self.A_VALS, self.B_VALS):
            model = GammaDisKin(a, b)
            rc_ratio = model.calc_radiocarbon_ratio_ss()
            print(f'Radiocarbon ratio for a={a}, b={b}: {rc_ratio}')
            
            # The result should be a tuple (value, error_estimate)
            self.assertIsInstance(rc_ratio, tuple,
                                msg=f"calc_radiocarbon_ratio_ss should return a tuple for a={a}, b={b}")
            self.assertEqual(len(rc_ratio), 2,
                           msg=f"calc_radiocarbon_ratio_ss should return a 2-tuple for a={a}, b={b}")
            
            # The radiocarbon ratio should be between 0 and 1 (approximately)
            ratio_value = rc_ratio[0]
            self.assertGreaterEqual(ratio_value, 0,
                                  msg=f"Radiocarbon ratio < 0 for a={a}, b={b}")
            self.assertLessEqual(ratio_value, 1.5,  # Allow some wiggle room
                               msg=f"Radiocarbon ratio > 1.5 for a={a}, b={b}")


class TestGeneralPowerLawDisKin(unittest.TestCase):

    T_MIN_VALS = [1.0, 10.0, 100.0]  # short time scale values
    T_MAX_VALS = [1000.0, 10000.0]  # long time scale values
    BETA_VALS = [0.5, np.exp(-GAMMA), 0.9]  # beta parameter values

    def test_numerical_T_integral(self):
        """Test that the analytical and numerical transit times match."""
        for t_min, t_max, beta in itertools.product(self.T_MIN_VALS, self.T_MAX_VALS, self.BETA_VALS):
            model = GeneralPowerLawDisKin(t_min, t_max, beta)
            T_int = model.calc_transit_time()
            T_calc = model.T
            print(f'Calculated transit time: {T_calc}')
            print(f'Integrated transit time: {T_int[0]}')
            
            self.assertAlmostEqual(T_calc, T_int[0], places=3,
                                 msg=f"Transit time mismatch for t_min={t_min}, t_max={t_max}, beta={beta}")

    def test_numerical_age_integral(self):
        """Test that the analytical and numerical mean ages can be calculated."""
        for t_min, t_max, beta in itertools.product(self.T_MIN_VALS, self.T_MAX_VALS, self.BETA_VALS):
            model = GeneralPowerLawDisKin(t_min, t_max, beta)
            A_int = model.calc_mean_age()
            print(f'Integrated mean age: {A_int[0]}')
            
            # Mean age should be positive
            self.assertGreater(
                A_int[0], 0, msg=f"Mean age is not positive for t_min={t_min}, t_max={t_max}, beta={beta}")

            self.assertAlmostEqual(
                model.A, A_int[0], places=3,
                msg=f"Mean age mismatch for t_min={t_min}, t_max={t_max}, beta={beta}")
    
    def test_pdf_properties(self):
        """Test that the age distribution PDF has valid properties."""
        for t_min, t_max, beta in itertools.product(self.T_MIN_VALS, self.T_MAX_VALS, self.BETA_VALS):
            model = GeneralPowerLawDisKin(t_min, t_max, beta)
            
            # PDF should be non-negative
            test_ages = np.logspace(0, 3, 20)
            for age in test_ages:
                pA = model.pA(age)
                self.assertGreaterEqual(
                    pA, 0, msg=f"PDF is negative at age={age} for t_min={t_min}, t_max={t_max}, beta={beta}")
            
            # s(t) should be decreasing
            s_vals = [model.s(t) for t in test_ages]
            for i in range(len(s_vals) - 1):
                self.assertGreaterEqual(
                    s_vals[i], s_vals[i+1],
                    msg=f"s(t) is not decreasing for t_min={t_min}, t_max={t_max}, beta={beta}")

    def test_cdf_properties(self):
        """Test that the CDF has valid properties."""
        for t_min, t_max, beta in itertools.product(self.T_MIN_VALS, self.T_MAX_VALS, self.BETA_VALS):
            model = GeneralPowerLawDisKin(t_min, t_max, beta)
            
            test_ages = np.logspace(0, 3, 20)
            cdf_vals = [model.cdfA(t) for t in test_ages]
            
            # CDF should be non-decreasing
            for i in range(len(cdf_vals) - 1):
                self.assertLessEqual(cdf_vals[i], cdf_vals[i+1],
                                   msg=f"CDF is not increasing for t_min={t_min}, t_max={t_max}, beta={beta}")
            
            # CDF should be between 0 and 1
            for i, cdf in enumerate(cdf_vals):
                self.assertGreaterEqual(cdf, 0,
                                      msg=f"CDF < 0 at age={test_ages[i]} for t_min={t_min}, t_max={t_max}, beta={beta}")
                self.assertLessEqual(cdf, 1.1,  # Allow small numerical error
                                   msg=f"CDF > 1 at age={test_ages[i]} for t_min={t_min}, t_max={t_max}, beta={beta}")

    def test_params_valid(self):
        """Test params_valid method. 
        
        For GeneralPowerLawDisKin, t_min and t_max must be positive, and t_min < t_max.
        """
        valid_params = [(10, 1000, 0.5), (1, 10000, np.exp(-GAMMA)), (50, 5000, 0.9)]
        for t_min, t_max, beta in valid_params:
            model = GeneralPowerLawDisKin(t_min, t_max, beta)
            self.assertTrue(model.params_valid(),
                            msg=f"Valid parameters t_min={t_min}, t_max={t_max}, beta={beta} marked as invalid.")
        
        invalid_params = [(-0.001, 1000, 0.5), (-10, 5000, 0.8), (10, -100, 0.3)]
        for t_min, t_max, beta in invalid_params:
            model = GeneralPowerLawDisKin(t_min, t_max, beta)
            self.assertFalse(model.params_valid(),
                             msg=f"Invalid parameters t_min={t_min}, t_max={t_max}, beta={beta} marked as valid.")
        
        # Beta must be between 0 and 1
        with self.assertRaises(ValueError):
            GeneralPowerLawDisKin(t_min=10, t_max=1000, beta=1.5)  # beta > 1
        
        with self.assertRaises(ValueError):
            GeneralPowerLawDisKin(t_min=10, t_max=1000, beta=-0.1)  # beta < 0

    def test_radiocarbon_ratio(self):
        """Test that the radiocarbon ratio calculation runs without error."""
        for t_min, t_max, beta in itertools.product(self.T_MIN_VALS, self.T_MAX_VALS, self.BETA_VALS):
            model = GeneralPowerLawDisKin(t_min, t_max, beta)
            rc_ratio = model.calc_radiocarbon_ratio_ss()
            print(f'Radiocarbon ratio for t_min={t_min}, t_max={t_max}, beta={beta}: {rc_ratio}')
            
            # The result should be a tuple (value, error_estimate)
            self.assertIsInstance(rc_ratio, tuple,
                                msg=f"calc_radiocarbon_ratio_ss should return a tuple")
            self.assertEqual(len(rc_ratio), 2,
                           msg=f"calc_radiocarbon_ratio_ss should return a 2-tuple")
            
            # The radiocarbon ratio should be between 0 and 1 (approximately)
            ratio_value = rc_ratio[0]
            self.assertGreaterEqual(ratio_value, 0,
                                  msg=f"Radiocarbon ratio < 0 for t_min={t_min}, t_max={t_max}, beta={beta}")
            self.assertLessEqual(ratio_value, 1.5,  # Allow some wiggle room
                               msg=f"Radiocarbon ratio > 1.5 for t_min={t_min}, t_max={t_max}, beta={beta}")


class TestLognormalDisKin(unittest.TestCase):
    
    MEAN_AGES = [10, 100, 1000]
    MEAN_TRANSIT_TIMES = [10, 100, 1000]

    MUS = [0.0, 1.0, 2.0]
    SIGMAS = [0.1, 0.5, 1.0]

    INVALID_SIGMAS = [-0.1, -1.0]

    def test_basic_properties(self):
        """Test that basic properties of the model are calculated correctly."""
        for A, T in itertools.product(self.MEAN_AGES, self.MEAN_TRANSIT_TIMES):
            if A/T < 1:
                continue
            
            model = LognormalDisKin.from_age_and_transit_time(A, T)
            
            # Check that transit time matches input
            self.assertAlmostEqual(model.T, T, places=5,
                                 msg=f"Transit time mismatch for A={A}, T={T}")
            
            # Check that mean age matches input
            self.assertAlmostEqual(model.A, A, places=5,
                                 msg=f"Mean age mismatch for A={A}, T={T}")
            
        for mu, sigma in itertools.product(self.MUS, self.SIGMAS):
            model = LognormalDisKin(mu, sigma)
            expected_A = np.exp(1.5 * sigma**2 - mu)
            expected_T = np.exp(sigma**2 / 2 - mu)

            self.assertAlmostEqual(model.A, expected_A, places=5,
                                   msg=f"Mean age mismatch for mu={mu}, sigma={sigma}")
            self.assertAlmostEqual(model.T, expected_T, places=5,
                                   msg=f"Transit time mismatch for mu={mu}, sigma={sigma}")

    def test_parameter_validation(self):
        """Test that invalid parameter combinations are handled."""
        # a/T < 1 should not be allowed in from_age_and_transit_time
        # as it implies negative sigma
        with self.assertRaises(ValueError):
            # This should fail because a/T < 1 implies negative sigma_squared
            model = LognormalDisKin.from_age_and_transit_time(a=50, T=100)

        for sigma in self.INVALID_SIGMAS:
            model = LognormalDisKin(mu=0, sigma=sigma)
            self.assertFalse(model.params_valid(),
                             msg=f"Invalid sigma={sigma} marked as valid.")

    def test_input_output_relations(self):
        for a, T in itertools.product(self.MEAN_AGES, self.MEAN_TRANSIT_TIMES):
            if a/T < 1:
                with self.assertRaises(ValueError):
                    LognormalDisKin.from_age_and_transit_time(a, T)
                continue

            print('constructing model with mean_age', a, 'and T', T)
            model = LognormalDisKin.from_age_and_transit_time(a, T)
            mu, sigma = model.mu, model.sigma
            expected_ln_a = 1.5*sigma**2 - mu
            expected_ln_T = sigma**2/2 - mu
            expected_a = np.exp(expected_ln_a)
            expected_T = np.exp(expected_ln_T)
            self.assertAlmostEqual(expected_a, a, msg=f"Age mismatch for a={a}, T={T}, mu={mu}, sigma={sigma}")
            self.assertAlmostEqual(expected_T, T, msg=f"Transit time mismatch for a={a}, T={T}, mu={mu}, sigma={sigma}")    

