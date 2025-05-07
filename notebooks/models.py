import numpy as np

from scipy.integrate import quad, dblquad
from scipy.special import exp1
from scipy.stats import lognorm, norm
from constants import LAMBDA_14C, INTERP_R_14C


# TODO: make a parent class for all of the models.
# TODO: write some simple unit tests checking internal consistency of models.
# TODO: PowerLawDisKin is poorly named, the variant with t^{-alpha} is also power laws.


class PowerLawDisKin:
    """A model where rates of decay are proportional to 1/t between two bounding timescales.

    We call these bounding timescales tau_0 and tau_inf as in the notes. 
    """
    def __init__(self, tau_0, tau_inf, interp_r_14c=None):
        """
        Args:
        tau_0: float
            The short time scale
        tau_inf: float
            The long time scale
        interp_r_14c: callable
            An interpolator for the estimated historical radiocarbon concentration.
            Takes a single argument, the number of years before a reference time (e.g. 2000).
            If None uses the default interpolator from constants.
        """
        self.t0 = tau_0  # short time scale
        self.tinf = tau_inf  # long time scale
        self.tratio = tau_0/ tau_inf

        self.interp_14c = interp_r_14c
        if interp_r_14c is None:
            self.interp_14c = INTERP_R_14C

        # steady-state transit time
        e1_term = exp1(self.tratio)
        self.e1_term = e1_term
        self.T = tau_0 * np.exp(self.tratio) * e1_term

        # mean age at steady-state
        self.A = (tau_inf * np.exp(-self.tratio)/e1_term) - tau_0

    def radiocarbon_age_integrand(self, a):
        # Interpolation was done with x as years before present,
        # so a is the correct input here
        initial_r = self.interp_14c(a) 
        radiocarbon_decay = np.exp(-LAMBDA_14C*a)
        age_dist_term = np.power((self.e1_term * (self.t0 + a)), -1) * np.exp(-(self.t0 + a)/self.tinf)
        return initial_r * age_dist_term * radiocarbon_decay
    
    def mean_transit_time_integrand(self, a):
        return self.t0 * np.exp(-a/self.tinf) / (self.t0 + a)
    
    def calc_mean_transit_time(self):
        """Calculate the mean by integrating the transit time distribution."""
        return quad(self.mean_transit_time_integrand, 0, np.inf)
    
    def pA(self, a):
        t0 = self.t0
        tinf = self.tinf
        e1_term = self.e1_term
        return np.exp(-(t0 + a)/tinf) / ((t0 + a)*e1_term)

    def mean_age_integrand(self, a):
        return a*self.pA(a)
    
    def calc_mean_age(self):
        """Calculate the mean by integrating the age distribution.
        
        Returns:
            A two-tuple of the mean age and an estimate of the absolute error.
        """
        return quad(self.mean_age_integrand, 0, np.inf)


class LognormalDisKin:

    def __init__(self, mu, sigma, k_min=None, k_max=None, interp_r_14c=None):
        """
        Args:
        mu: float
            The mean of the underlying normal distribution
        sigma: float
            The standard deviation underlying normal distribution
        k_min: float
            The minimum value of k to use in the integrals. UNITS?
        k_max: float
            The maximum value of k to use in the integrals. UNITS? 
        interp_r_14c: callable
            An interpolator for the estimated historical radiocarbon concentration.
            Takes a single argument, the number of years before a reference time (e.g. 2000).
            If None uses the default interpolator from constants.
        """
        self.mu = mu
        self.k_star = np.exp(mu)
        self.sigma = sigma
        self.k_min = k_min or lognorm.ppf(1e-10, s=sigma, scale=np.exp(mu))
        self.k_max = k_max or lognorm.ppf(1.0-1e-10, s=sigma, scale=np.exp(mu))

        # rescale ks by the median
        self.kappa_min = self.k_min / self.k_star
        self.kappa_max = self.k_max / self.k_star

        # log scale
        self.q_min = np.log(self.kappa_min)
        self.q_max = np.log(self.kappa_max)        

        self.interp_14c = interp_r_14c
        if interp_r_14c is None:
            self.interp_14c = INTERP_R_14C

        # steady-state transit time
        self.T = np.exp((-self.mu + self.sigma**2)/2)
        self.a = self.T * np.exp(self.sigma**2)

    @classmethod
    def from_age_and_transit_time(cls, a, T):
        """Construct a LognormalDisKin object from the mean age and transit time.
        
        Note: a/T >= 1 is required. a/T < 1 implies a negative lognormal standard deviation.

        Args:
            a: float
                The mean age
            T: float
                The transit time

        Returns:
            LognormalDisKin
                An instance of the LognormalDisKin class.
        """
        sigma_squared = np.log(a/T)
        sigma = np.sqrt(sigma_squared)
        mu = sigma_squared/2 - np.log(T)

        # choose the min and max ks to be tails of the lognormal distribution
        lognorm_min = lognorm.ppf(1e-10, s=sigma, scale=np.exp(mu))
        lognorm_max = lognorm.ppf(1.0-1e-10, s=sigma, scale=np.exp(mu))     
        return cls(mu, sigma, k_min=lognorm_min, k_max=lognorm_max)
    
    def _pA_alpha_logscale_integrand(self, alpha, q):
        """Integrand for the age distribution in log scale."""
        p_q = norm.pdf(q, loc=0, scale=self.sigma)
        exp_factor = np.exp(-alpha*np.exp(q))
        return p_q*exp_factor
    
    def _mean_age_integrand(self, alpha, q):
        """Integrand for the mean age in log scale."""
        return alpha * self._pA_alpha_logscale_integrand(alpha, q)
    
    def calc_mean_age(self):
        """Integrate the age distribution in log scale.
        
        Returns:
            A two-tuple of the mean age and an estimate of the absolute error.
        """
        res = dblquad(self._mean_age_integrand, self.q_min, self.q_max, 0, np.inf)
        return res
    
    def _radiocarbon_age_integrand(self, a, q):
        """Integrand for the radiocarbon age distribution."""
        initial_r = self.interp_14c(a)
        radiocarbon_decay = np.exp(-LAMBDA_14C*a)
        p_q = norm.pdf(q, loc=0, scale=self.sigma)
        exp_factor = np.exp(-a*self.k_star*np.exp(q))
        return initial_r * radiocarbon_decay * p_q * exp_factor

    def calc_radiocarbon_age(self):
        """Integrate the radiocarbon age distribution."""
        res = dblquad(self._radiocarbon_age_integrand, self.q_min, self.q_max, 0, np.inf)
        return res
    

import unittest
import itertools

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

    def test_mean_age_integral(self):
        for a, T in itertools.product(self.MEAN_AGES, self.MEAN_TRANSIT_TIMES):
            if a/T <= 1:
                # not allowed
                continue
            
            print('constructing model with mean_age', a, 'and T', T)
            model = LognormalDisKin.from_age_and_transit_time(a, T)
            mu, sigma = model.mu, model.sigma
            print(f'calculated mu {mu:.2f}, sigma {sigma:.2f}')
            expected_ln_a = 1.5*sigma**2 - mu
            expected_ln_T = sigma**2/2 - mu
            self.assertAlmostEqual(np.exp(expected_ln_a), a, delta=1e-5)
            self.assertAlmostEqual(np.exp(expected_ln_T), T, delta=1e-5)

            mean_age_calc, mean_age_err = model.calc_mean_age()
            print(f'calculated mean age {mean_age_calc:.2f} +/- {mean_age_err:.2f} compared to input age {a}')

            abs_diff = np.abs(mean_age_calc - a)
            relative_diff = abs_diff / a
            self.assertLessEqual(relative_diff, 0.01,
                                 msg=f"Mean age {mean_age_calc:.2f} is not within 1% of expected {a}")
            #self.assertLessEqual(abs_diff, mean_age_err, 
            #                     msg=f"Mean age {mean_age_calc:.2f} is not within error {mean_age_err:.4f} of expected {a}")

        
if __name__ == "__main__":
    # Run the tests
    unittest.main()


class old_LognormalDisKin:
    """Not ready for use."""
    
    def __init__(self, mu, sigma, interp_r_14c=None):
        """
        Args:
        mu: float
            The mean of the underlying normal distribution
        sigma: float
            The standard deviation underlying normal distribution
        interp_r_14c: callable
            An interpolator for the estimated historical radiocarbon concentration.
            Takes a single argument, the number of years before a reference time (e.g. 2000).
            If None uses the default interpolator from constants.
        """
        self.mu = mu
        self.sigma = sigma

        self.interp_14c = interp_r_14c
        if interp_r_14c is None:
            self.interp_14c = INTERP_R_14C

        # steady-state transit time
        self.T = np.exp((-self.mu + self.sigma**2)/2)
        self.a = self.T * np.exp(self.sigma**2)

    @classmethod
    def from_age_and_transit_time(cls, a, T):
        sigma_squared = np.log(a/T)
        sigma = np.sqrt(sigma_squared)
        mu = sigma_squared - 2*(np.log(T)) 
        return cls(mu, sigma)

    def age_dist_integrand(self, k):
        sigma, mu, T = self.sigma, self.mu, self.T
        p_k = lognorm.pdf(k, s=sigma, scale=np.exp(mu))
        return p_k/T
    
    def _pA_integrand(self, a, k):
        sigma, mu, T = self.sigma, self.mu, self.T
        p_k = lognorm.pdf(k, s=sigma, scale=np.exp(mu))
        return np.exp(-k*a) * p_k / T
    
    def pA(self, a):
        return quad(self._pA_integrand, 0, np.inf, args=(a))

    def _mean_age_integrand(self, a, k):
        return a * self._pA_integrand(a, k)
    
    def mean_age(self):
        return dblquad(self._mean_age_integrand, 0, np.inf, 0, np.inf)

    def radiocarbon_age_integrand(self, a, k):
        initial_r = self.interp_14c(a)
        lognormal_term = self.age_dist_integrand(k)
        radiocarbon_decay = np.exp(-LAMBDA_14C*a)
        return initial_r * radiocarbon_decay * lognormal_term
    
    def radiocarbon_age_integrand_mc(self, x):
        a, k = x
        return self.radiocarbon_age_integrand(a, k)