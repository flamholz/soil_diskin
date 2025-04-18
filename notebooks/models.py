import numpy as np

from scipy.integrate import quad, dblquad
from scipy.special import exp1
from scipy.stats import lognorm
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
    
    def pA(self, a):
        t0 = self.t0
        tinf = self.tinf
        e1_term = self.e1_term
        return np.exp(-(t0 + a)/tinf) / ((t0 + a)*e1_term)

    def mean_age_integrand(self, a):
        return a*self.pA(a)

    

class LognormalDisKin:
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
    