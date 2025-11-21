from soil_diskin.models import *

import unittest
import itertools

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
            T_int = pow_diskin.calc_mean_transit_time()
            T_calc = pow_diskin.T
            print('Calcualted transit time: ', T_calc)
            print('Integrated transit time: ', T_int[0])

            A_calc = pow_diskin.A
            A_int = pow_diskin.calc_mean_age()
            print('Calculated mean age: ', A_calc)
            print('Integrated mean age: ', A_int[0])

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


class TestLognormalDisKin(unittest.TestCase):
    
    MEAN_AGES = [10, 100, 1000]
    MEAN_TRANSIT_TIMES = [10, 100, 1000]

    def test_input_output(self):
        for a, T in itertools.product(self.MEAN_AGES, self.MEAN_TRANSIT_TIMES):
            if a/T < 1:
                # not allowed
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

