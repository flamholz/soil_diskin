import itertools
import unittest
import numpy as np

from scipy.integrate import quad
from scipy.stats import geninvgauss

from soil_diskin.continuum_models import PowerLawDisKin, LogUniformDisKin
from soil_diskin.continuum_models import GeneralPowerLawDisKin, LognormalDisKin
from soil_diskin.continuum_models import SoftCutoffPowerLawDisKin
from soil_diskin.constants import GAMMA, LAMBDA_14C
from soil_diskin.radiocarbon_utils import load_atm14c

model_classes = [PowerLawDisKin,
                 GeneralPowerLawDisKin, SoftCutoffPowerLawDisKin, LognormalDisKin, LogUniformDisKin]


class TestLogUniformDisKin(unittest.TestCase):
    def test_analytical_moments_and_distribution(self):
        model = LogUniformDisKin(k_min=0.01, log_width=np.log(100))

        self.assertTrue(model.params_valid())
        self.assertAlmostEqual(model.T, (100.0 - 1.0) / np.log(100.0))
        self.assertAlmostEqual(model.A, (100.0 + 1.0) / 2.0)
        self.assertAlmostEqual(model.s(0.0), 1.0)
        self.assertAlmostEqual(model.cdfA(0.0), 0.0)
        self.assertAlmostEqual(model.calc_transit_time()[0], model.T, places=5)
        self.assertAlmostEqual(model.calc_mean_age()[0], model.A, places=5)

        ages = np.logspace(-3, 5, 100)
        survival = model.s(ages)
        cdf = model.cdfA(ages)
        self.assertTrue(np.all(np.diff(survival) <= 0))
        self.assertTrue(np.all(np.diff(cdf) >= 0))
        self.assertTrue(np.all((cdf >= 0) & (cdf <= 1)))

        wide_model = LogUniformDisKin(k_min=1e-3, log_width=1000)
        self.assertTrue(np.isfinite([wide_model.T, wide_model.A]).all())
        self.assertTrue(0 < wide_model.s(1.0) < 1)
        self.assertTrue(0 < wide_model.cdfA(1.0) < 1)

    def test_parameter_validation(self):
        self.assertFalse(LogUniformDisKin(0.0, 1.0).params_valid())
        self.assertFalse(LogUniformDisKin(1.0, 0.0).params_valid())
        self.assertFalse(LogUniformDisKin(1.0, -1.0).params_valid())

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


class TestGeneralPowerLawDisKin(unittest.TestCase):

    T_MIN_VALS = [1.0, 10.0, 100.0]  # short time scale values
    T_MAX_VALS = [1000.0, 10000.0]  # long time scale values
    ALPHA_VALS = [0.5, np.exp(-GAMMA), 0.9]  # alpha parameter values

    def test_numerical_T_integral(self):
        """Test that the analytical and numerical transit times match."""
        for t_min, t_max, alpha in itertools.product(self.T_MIN_VALS, self.T_MAX_VALS, self.ALPHA_VALS):
            model = GeneralPowerLawDisKin(t_min, t_max, alpha)
            T_int = model.calc_transit_time()
            T_calc = model.T
            print(f'Calculated transit time: {T_calc}')
            print(f'Integrated transit time: {T_int[0]}')
            
            self.assertAlmostEqual(T_calc, T_int[0], places=3,
                                 msg=f"Transit time mismatch for t_min={t_min}, t_max={t_max}, alpha={alpha}")

    def test_numerical_age_integral(self):
        """Test that the analytical and numerical mean ages can be calculated."""
        for t_min, t_max, alpha in itertools.product(self.T_MIN_VALS, self.T_MAX_VALS, self.ALPHA_VALS):
            model = GeneralPowerLawDisKin(t_min, t_max, alpha)
            A_int = model.calc_mean_age()
            print(f'Integrated mean age: {A_int[0]}')
            
            # Mean age should be positive
            self.assertGreater(
                A_int[0], 0, msg=f"Mean age is not positive for t_min={t_min}, t_max={t_max}, alpha={alpha}")

            self.assertAlmostEqual(
                model.A, A_int[0], places=3,
                msg=f"Mean age mismatch for t_min={t_min}, t_max={t_max}, alpha={alpha}")
    
    def test_pdf_properties(self):
        """Test that the age distribution PDF has valid properties."""
        for t_min, t_max, alpha in itertools.product(self.T_MIN_VALS, self.T_MAX_VALS, self.ALPHA_VALS):
            model = GeneralPowerLawDisKin(t_min, t_max, alpha)
            
            # PDF should be non-negative
            test_ages = np.logspace(0, 3, 20)
            for age in test_ages:
                pA = model.pA(age)
                self.assertGreaterEqual(
                    pA, 0, msg=f"PDF is negative at age={age} for t_min={t_min}, t_max={t_max}, alpha={alpha}")
            
            # s(t) should be decreasing
            s_vals = [model.s(t) for t in test_ages]
            for i in range(len(s_vals) - 1):
                self.assertGreaterEqual(
                    s_vals[i], s_vals[i+1],
                    msg=f"s(t) is not decreasing for t_min={t_min}, t_max={t_max}, alpha={alpha}")

    def test_cdf_properties(self):
        """Test that the CDF has valid properties."""
        for t_min, t_max, alpha in itertools.product(self.T_MIN_VALS, self.T_MAX_VALS, self.ALPHA_VALS):
            model = GeneralPowerLawDisKin(t_min, t_max, alpha)
            
            test_ages = np.logspace(0, 3, 20)
            cdf_vals = [model.cdfA(t) for t in test_ages]
            
            # CDF should be non-decreasing
            for i in range(len(cdf_vals) - 1):
                self.assertLessEqual(cdf_vals[i], cdf_vals[i+1],
                                   msg=f"CDF is not increasing for t_min={t_min}, t_max={t_max}, alpha={alpha}")
            
            # CDF should be between 0 and 1
            for i, cdf in enumerate(cdf_vals):
                self.assertGreaterEqual(cdf, 0,
                                      msg=f"CDF < 0 at age={test_ages[i]} for t_min={t_min}, t_max={t_max}, alpha={alpha}")
                self.assertLessEqual(cdf, 1.1,  # Allow small numerical error
                                   msg=f"CDF > 1 at age={test_ages[i]} for t_min={t_min}, t_max={t_max}, alpha={alpha}")

    def test_params_valid(self):
        """Test params_valid method. 
        
        For GeneralPowerLawDisKin, t_min and t_max must be positive, and t_min < t_max.
        """
        valid_params = [(10, 1000, 0.5), (1, 10000, np.exp(-GAMMA)), (50, 5000, 0.9)]
        for t_min, t_max, alpha in valid_params:
            model = GeneralPowerLawDisKin(t_min, t_max, alpha)
            self.assertTrue(model.params_valid(),
                            msg=f"Valid parameters t_min={t_min}, t_max={t_max}, alpha={alpha} marked as invalid.")
        
        invalid_params = [(-0.001, 1000, 0.5), (-10, 5000, 0.8), (10, -100, 0.3)]
        for t_min, t_max, alpha in invalid_params:
            model = GeneralPowerLawDisKin(t_min, t_max, alpha)
            self.assertFalse(model.params_valid(),
                             msg=f"Invalid parameters t_min={t_min}, t_max={t_max}, alpha={alpha} marked as valid.")
        
        # The constructor enforces 0 < alpha < 1: expint gives NaN T and A for alpha >= 1.
        for alpha in [-0.1, 0.0, 1.0, 1.5]:
            with self.assertRaises(ValueError, msg=f"alpha={alpha} should raise"):
                GeneralPowerLawDisKin(t_min=10, t_max=1000, alpha=alpha)

    def test_radiocarbon_ratio(self):
        """Test that the radiocarbon ratio calculation runs without error."""
        for t_min, t_max, alpha in itertools.product(self.T_MIN_VALS, self.T_MAX_VALS, self.ALPHA_VALS):
            model = GeneralPowerLawDisKin(t_min, t_max, alpha)
            rc_ratio = model.calc_radiocarbon_ratio_ss()
            print(f'Radiocarbon ratio for t_min={t_min}, t_max={t_max}, alpha={alpha}: {rc_ratio}')
            
            # The result should be a tuple (value, error_estimate)
            self.assertIsInstance(rc_ratio, tuple,
                                msg=f"calc_radiocarbon_ratio_ss should return a tuple")
            self.assertEqual(len(rc_ratio), 2,
                           msg=f"calc_radiocarbon_ratio_ss should return a 2-tuple")
            
            # The radiocarbon ratio should be between 0 and 1 (approximately)
            ratio_value = rc_ratio[0]
            self.assertGreaterEqual(ratio_value, 0,
                                  msg=f"Radiocarbon ratio < 0 for t_min={t_min}, t_max={t_max}, alpha={alpha}")
            self.assertLessEqual(ratio_value, 1.5,  # Allow some wiggle room
                               msg=f"Radiocarbon ratio > 1.5 for t_min={t_min}, t_max={t_max}, alpha={alpha}")


class TestSoftCutoffPowerLawDisKin(unittest.TestCase):

    T_MIN_VALS = [1.0, 10.0, 100.0]  # short time scale values
    T_MAX_VALS = [1000.0, 10000.0]  # long time scale values
    ALPHA_VALS = [0.0, 0.5, np.exp(-GAMMA), 1.0, 1.5]  # alpha parameter values

    def params(self):
        return itertools.product(self.T_MIN_VALS, self.T_MAX_VALS, self.ALPHA_VALS)

    def rate_dist(self, t_min, t_max, alpha):
        """The GIG rate distribution p(k) ∝ k^(alpha-1) exp(-k t_min - 1/(k t_max))."""
        return geninvgauss(alpha, 2 * np.sqrt(t_min / t_max), scale=1 / np.sqrt(t_min * t_max))

    def expect(self, dist, func, t_min, t_max):
        """E[func(k)], integrated over log k.

        dist.expect integrates over k directly and loses accuracy for small alpha,
        where p(k) ~ 1/k spreads mass over many decades.
        """
        integrand = lambda u: func(np.exp(u)) * dist.pdf(np.exp(u)) * np.exp(u)
        lo, hi = np.log(1 / t_max) - 25, np.log(1 / t_min) + 5
        return quad(integrand, lo, hi, points=[np.log(1 / t_max), np.log(1 / t_min)], limit=500)[0]

    def test_s_is_laplace_transform_of_rate_dist(self):
        """Test that s(t) = E[exp(-k t)] under the GIG rate distribution."""
        for t_min, t_max, alpha in self.params():
            model = SoftCutoffPowerLawDisKin(t_min, t_max, alpha)
            dist = self.rate_dist(t_min, t_max, alpha)
            for t in [0., 1., 30., 1000., 30000.]:
                expected = self.expect(dist, lambda k: np.exp(-k * t), t_min, t_max)
                self.assertAlmostEqual(model.s(t), expected, places=6,
                                       msg=f"s({t}) mismatch for t_min={t_min}, t_max={t_max}, alpha={alpha}")

    def test_moments_match_rate_dist(self):
        """Test that T = E[1/k] and A = E[1/k^2] / E[1/k] under the GIG rate distribution."""
        for t_min, t_max, alpha in self.params():
            model = SoftCutoffPowerLawDisKin(t_min, t_max, alpha)
            dist = self.rate_dist(t_min, t_max, alpha)
            T = self.expect(dist, lambda k: 1 / k, t_min, t_max)
            A = self.expect(dist, lambda k: 1 / k**2, t_min, t_max) / T
            self.assertAlmostEqual(model.T / T, 1, places=6,
                                   msg=f"T mismatch for t_min={t_min}, t_max={t_max}, alpha={alpha}")
            self.assertAlmostEqual(model.A / A, 1, places=6,
                                   msg=f"A mismatch for t_min={t_min}, t_max={t_max}, alpha={alpha}")

    def test_numerical_T_integral(self):
        """Test that the analytical and numerical transit times match."""
        for t_min, t_max, alpha in self.params():
            model = SoftCutoffPowerLawDisKin(t_min, t_max, alpha)
            T_int = model.calc_transit_time()
            self.assertAlmostEqual(model.T / T_int[0], 1, places=4,
                                   msg=f"Transit time mismatch for t_min={t_min}, t_max={t_max}, alpha={alpha}")

    def test_numerical_age_integral(self):
        """Test that the analytical and numerical mean ages match."""
        for t_min, t_max, alpha in self.params():
            model = SoftCutoffPowerLawDisKin(t_min, t_max, alpha)
            A_int = model.calc_mean_age()
            self.assertAlmostEqual(model.A / A_int[0], 1, places=4,
                                   msg=f"Mean age mismatch for t_min={t_min}, t_max={t_max}, alpha={alpha}")

    def test_cdf_matches_integrated_pdf(self):
        """Test that the closed-form CDF matches the integral of pA."""
        for t_min, t_max, alpha in self.params():
            model = SoftCutoffPowerLawDisKin(t_min, t_max, alpha)
            self.assertAlmostEqual(model.cdfA(0), 0, places=12)
            self.assertAlmostEqual(model.cdfA(1e12), 1, places=12)
            test_ages = np.logspace(0, 4, 9)
            for age in test_ages:
                expected = quad(model.pA, 0, age, limit=500)[0]
                self.assertAlmostEqual(model.cdfA(age), expected, places=6,
                                       msg=f"CDF mismatch at age={age} for t_min={t_min}, t_max={t_max}, alpha={alpha}")
            cdf_vals = model.cdfA(test_ages)
            self.assertTrue(np.all(np.diff(cdf_vals) > 0),
                            msg=f"CDF is not increasing for t_min={t_min}, t_max={t_max}, alpha={alpha}")

    def test_params_valid(self):
        """Test params_valid method."""
        valid_params = [(10, 1000, 0.0), (10, 1000, 0.5), (1, 10000, np.exp(-GAMMA)), (50, 5000, 1.5)]
        for t_min, t_max, alpha in valid_params:
            model = SoftCutoffPowerLawDisKin(t_min, t_max, alpha)
            self.assertTrue(model.params_valid(),
                            msg=f"Valid parameters t_min={t_min}, t_max={t_max}, alpha={alpha} marked as invalid.")

        invalid_params = [(-0.001, 1000, 0.5), (1000, 10, 0.8), (10, -100, 0.3)]
        for t_min, t_max, alpha in invalid_params:
            model = SoftCutoffPowerLawDisKin(t_min, t_max, alpha)
            self.assertFalse(model.params_valid(),
                             msg=f"Invalid parameters t_min={t_min}, t_max={t_max}, alpha={alpha} marked as valid.")

        with self.assertRaises(ValueError):
            SoftCutoffPowerLawDisKin(t_min=10, t_max=1000, alpha=-0.1)

    def test_radiocarbon_ratio(self):
        """Test that the radiocarbon ratio calculation runs and gives a plausible value."""
        for t_min, t_max, alpha in self.params():
            model = SoftCutoffPowerLawDisKin(t_min, t_max, alpha)
            ratio_value = model.calc_radiocarbon_ratio_ss()[0]
            self.assertGreaterEqual(ratio_value, 0,
                                    msg=f"Radiocarbon ratio < 0 for t_min={t_min}, t_max={t_max}, alpha={alpha}")
            self.assertLessEqual(ratio_value, 1.5,
                                 msg=f"Radiocarbon ratio > 1.5 for t_min={t_min}, t_max={t_max}, alpha={alpha}")


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


class TestLognormalDisKinRadiocarbon(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.atm = load_atm14c()

    def test_uses_supplied_atmosphere(self):
        model = LognormalDisKin(mu=-1.0, sigma=0.8, atm=self.atm)
        self.assertIs(model.atm, self.atm)

    def test_calc_radiocarbon_ratio_ss_returns_tuple(self):
        model = LognormalDisKin(mu=-1.0, sigma=0.8, atm=self.atm)
        ratio, err = model.calc_radiocarbon_ratio_ss()
        self.assertIsInstance(ratio, float)
        self.assertEqual(err, 0.0)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.5)

    def test_unresolved_distribution_raises(self):
        """Parameters whose tails leave the log-rate grid must not return a value."""
        with self.assertRaises(ValueError):
            LognormalDisKin(mu=-15.0, sigma=8.0, atm=self.atm).calc_radiocarbon_ratio_ss()


def test_lognormal_narrow_sigma_radiocarbon_matches_single_rate_limit():
    """As sigma -> 0 under a constant atmosphere, fm -> k0 / (lambda + k0) with k0 = exp(mu)."""
    atm = load_atm14c()
    constant = type(atm)(np.array([0.0, 10.0, 25.0, 80.0]), np.ones(4), tail_fm=1.0)
    k0 = 1e-4
    # sigma = 0.01 is the narrowest the model's log-rate grid resolves.
    model = LognormalDisKin(np.log(k0), 0.01, atm=constant)
    fm, _ = model.calc_radiocarbon_ratio_ss()
    np.testing.assert_allclose(fm, k0 / (LAMBDA_14C + k0), rtol=1e-3)


def test_stock_younger_than_is_zero_at_zero_and_increasing():
    """The stock of a pool started empty starts at zero and grows monotonically."""
    ts = np.array([0.0, 0.1, 1.0, 10.0, 100.0], dtype=np.float64)
    C = LognormalDisKin(0.25, 0.5).stock_younger_than(ts)

    np.testing.assert_allclose(C[0], 0.0, atol=1e-14)
    assert np.all(C >= 0.0)
    assert np.all(np.diff(C) >= 0.0)


def test_stock_younger_than_approaches_steady_state():
    """At long times the stock (per unit input) approaches the turnover time T."""
    model = LognormalDisKin(0.1, 0.45)
    ts = np.array([0.5, 5.0, 50.0, 1000.0], dtype=np.float64)

    C = model.stock_younger_than(ts)

    assert C[-1] > C[-2]
    np.testing.assert_allclose(C[-1], model.T, rtol=2e-3, atol=0.0)


def test_stock_younger_than_accepts_scalars_and_arrays_and_gives_cdfA():
    """Scalar and array calls agree, and the age CDF is the stock over T."""
    model = LognormalDisKin.from_age_and_transit_time(30.0, 8.0)
    ts = np.logspace(-1, np.log10(200.0), 32)

    C = model.stock_younger_than(ts)

    assert C.shape == ts.shape
    np.testing.assert_array_equal(C, [model.stock_younger_than(t) for t in ts])
    np.testing.assert_allclose(model.cdfA(ts), C / model.T, rtol=1e-15)
    assert isinstance(model.cdfA(5.0), float)
