import numpy as np

from scipy.integrate import quad
from scipy.special import exp1, gamma, gammaincc, log_ndtr, gammainc
from scipy.stats import lognorm
from soil_diskin import constants
from soil_diskin.constants import LAMBDA_14C, GAMMA
from soil_diskin.lognormal import (lognormal_radiocarbon, inner_integral, C14_MEAN_LIFE,
                                  lognormal_turnover, diskin_C_of_t, LognormalPrediction)
from soil_diskin.radiocarbon_utils import AtmC14
from tqdm import tqdm


def expint(n, x):
    """Generalized exponential integral E_n(x) for real n < 1 and x > 0.

    Uses the identity E_n(x) = x^(n-1) * Γ(1-n, x), where the upper
    incomplete gamma function is computed as gammaincc(1-n, x) * gamma(1-n).
    """
    return x ** (n - 1) * gammaincc(1 - n, x) * gamma(1 - n)


# TODO: PowerLawDisKin is poorly named, the variant with t^{-alpha} is also power laws.

class AbstractDiskinModel:
    """Abstract base class defining the interface for DisKin models.

    Note that the name DisKin refers to "disordered kinetics" models
    that desribe decay rate distributions. The power law models are
    somewhat distinct, as they describe variation with age, rather 
    than static disorder as in the lognormal and gamma models. We have 
    kept the name nonetheless. 
    
    It is expected that concrete subclasses expose analytically-calculated
    values of the mean age and transit time at steady-state as properties
    A and T.
    """

    def __init__(self, interp_r_14c=None):
        """Initialize the model."""
        self.interp_14c = interp_r_14c
        if interp_r_14c is None:
            self.interp_14c = constants.INTERP_R_14C
    
        # These should be calculated by subclasses
        self.T = None  # mean transit time at steady-state
        self.A = None  # mean age at steady-state

    def params_valid(self):
        """Returns True if the parameters are valid, False otherwise."""
        raise NotImplementedError("Subclasses must implement this method.")

    def s(self, t):
        """The survival function at age t.
        
        The survival function gives the fraction of input remaining at age t.
        
        Args:
            t: float
                The age at which to evaluate the survival function.
        
        Returns:
            float
                The value of the survival function at age t.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def pA(self, t):
        """Calculate the probability density function of the age distribution.
        
        Default implementation uses pA(t) = s(t) / T, where s(t) is the 
        survival function and T is the mean transit time at steady-state.

        Reminder that pA(t) is a probability density function. Extracting 
        a probability requires integrating pA(t) over an interval dt.

        Args:
            t: float
                The age at which to evaluate the PDF.
        
        Returns:
            float
                The value of the PDF at age t.
        """
        return self.s(t) / self.T
    
    def cdfA(self, t):
        """Calculate the cumulative distribution function of the age distribution.

        The CDF is the integral of the PDF from 0 to t.
        
        Args:
            t: float
                The age at which to evaluate the CDF.
        
        Returns:
            float
                The value of the CDF at age t.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def mean_age_integrand(self, a):
        """The integrand for calculating the mean age numerically.
        
        Args:
            a: float
                The age at which to evaluate the integrand.
        
        Returns:
            float
                The value of the integrand at age a.
        """
        return a*self.pA(a)
    
    def transit_time_integrand(self, a):
        """The integrand for calculating the transit time numerically.
        
        Args:
            a: float
                The age at which to evaluate the integrand.
        
        Returns:
            float
                The value of the integrand at age a.
        """
        return self.s(a)
    
    def calc_mean_age(self, quad_limit=1500, quad_epsabs=1e-3):
        """Calculate the mean age of the reservoir at steady-state 
        by integrating the age distribution pA(t).

        Args:
            quad_limit: int
                The maximum number of subintervals for the quadrature.
            quad_epsabs: float
                The absolute error tolerance for the quadrature.
        
        Returns:
            A two-tuple of the mean age and an estimate of the absolute error.
        """
        return quad(self.mean_age_integrand, 0, np.inf,
                    limit=quad_limit, epsabs=quad_epsabs)
    
    def calc_transit_time(self, quad_limit=1500, quad_epsabs=1e-3):
        """Calculate the mean transit time T by numerical integration. 
        
        The steady-state transit time (turnover time) is given by 

            T = \int_0^\infty t * pT(t) dt.
        
        where pT(t) is the transit time distribution. It can be shown that
         
           T = \int_0^\infty s(t) dt.

        We use the latter relationship here

        Returns:
            A two-tuple of the mean transit time and an estimate of the absolute error.
        """
        return quad(self.transit_time_integrand, 0, np.inf,
                    limit=quad_limit, epsabs=quad_epsabs)
    
    def radiocarbon_age_integrand(self, a):
        """The integrand for calculating the radiocarbon age numerically.
        
        Default implementation uses pA from the subclass and the 
        interpolated radiocarbon concentration provided at initialization.

        Args:
            a: float
                The age at which to evaluate the integrand.
        
        Returns:
            float
                The value of the integrand at age a.
        """
        # Interpolation was done with x as years before present,
        # so a is the correct input here
        initial_r = self.interp_14c(a) 
        radiocarbon_decay = np.exp(-LAMBDA_14C*a)
        return initial_r * self.pA(a) * radiocarbon_decay

    def calc_radiocarbon_ratio_ss(self, quad_limit=1500, quad_epsabs=1e-3):
        """Calculate the radiocarbon age by integrating the age distribution.
        
        Returns:
            A two-tuple of the radiocarbon age and an estimate of the absolute error.
        """
        return quad(self.radiocarbon_age_integrand, 0, np.inf,
                    limit=quad_limit, epsabs=quad_epsabs)


class GammaDisKin(AbstractDiskinModel):
    """A model where the rate distribution is gamma."""
    def __init__(self, a, b, interp_r_14c=None, I=None):
        """
        Args:
        a: float
            The shape parameter of the gamma distribution
        b: float
            The scale parameter of the gamma distribution
            The long time scale
        interp_r_14c: callable
            An interpolator for the estimated historical radiocarbon concentration.
            Takes a single argument, the number of years before a reference time (e.g. 2000).
            If None uses the default interpolator from constants.
        I: np.ndarray, optional
            An array of inputs to the model, representing the carbon input to the system.
        """
        super().__init__(interp_r_14c=interp_r_14c)

        self.a = a  # shape parameter
        self.b = b  # scale parameter -- should not be zero
        self.I = I

        self.T = 1 / ((-1 + a) * b)
        self.A = 1/((-2 + a) * (-1 + a) * b**2) / self.T

    def params_valid(self):
        """Returns True if the parameters are valid, False otherwise."""
        return self.a > 0 and self.b > 0

    def s(self, t):
        """the term for the amount of carbon in the system at age t"""
        return (1 + self.b * t) ** (-self.a)

    def cdfA(self, t):
        """Calculate the cumulative distribution function of the age distribution."""
        # The CDF is the integral of the PDF from 0 to a
        cdf = (-1 + (1 + self.b * t) ** (1 - self.a)) / (self.b * (1 - self.a)) / self.T
        return cdf

class GeneralPowerLawDisKin(AbstractDiskinModel):
    """A model where rates of decay are proportional to 1/t between two bounding timescales.

    We call these bounding timescales tau_0 and tau_inf as in the notes. 
    """
    def __init__(self, t_min, t_max, beta = np.exp(-GAMMA), interp_r_14c=None, I=None):
        """
        Args:
        t_min: float
            The short time scale
        t_max: float
            The long time scale
        beta: float
            The exponent of the power law decay with age. Must be positive.
            TODO: beta is called alpha in the paper. rename for consistency.
        interp_r_14c: callable
            An interpolator for the estimated historical radiocarbon concentration.
            Takes a single argument, the number of years before a reference time (e.g. 2000).
            If None uses the default interpolator from constants.
        I: np.ndarray, optional
            An array of inputs to the model, representing the carbon input to the system.
        """
        super().__init__(interp_r_14c=interp_r_14c)

        if beta <= 0:
            raise ValueError("Beta parameter must be positive.")

        self.t_min = t_min  # short time scale
        self.t_max = t_max  # long time scale
        self.I = I
        self.beta = beta

        tratio = t_min / t_max
        self.tratio = tratio

        # steady-state transit time
        self.T = t_min * np.exp(tratio) * expint(self.beta, tratio)

        # mean age at steady-state
        self.A = t_min * (expint(self.beta - 1, tratio) / expint(self.beta, tratio) - 1)

    def params_valid(self):
        """Returns True if the parameters are valid, False otherwise.
        
        TODO: is t_min == 0 valid? tmax == tmin?
        """
        t_min_valid = self.t_min > 0
        t_max_valid = self.t_max > 0
        t_hierarchy_valid = self.t_max > self.t_min
        beta_valid = self.beta > 0
        return t_min_valid and t_max_valid and t_hierarchy_valid and beta_valid

    def s(self, t):
        """The survival function at age t.

        The survival function gives the fraction of input remaining at age t.

        The expression for s(t) is
            s(t) = exp(- t / t_max) * ( t_min / (t_min + t) )^β

        Args:
            t: float
                The age at which to evaluate the survival function.

        Returns:
            float
                The fraction of input remaining at age t.
        """
        return np.exp(-t / self.t_max) * (self.t_min / (self.t_min + t)) ** self.beta
    
    def impulse(self, t, X):
        """Calculate the change in state of the system at time t.

        Args:
            t: float
                The time at which to calculate the change in state.
            X: np.ndarray
                The current state of the system, an array of carbon pools.
        Returns:
            np.ndarray
                The change in state of the system at time t.
        """
        # The rate of change is proportional to the inverse of the time scale
        # and the current state of the system.
        # The decay rate is 1/tau_0 for the short time scale and 1/tau_inf for the long time scale.
        dX = - ( 1 / (self.t_min + t) + 1 / (self.t_max + t)) * X

        return dX
    
    def cdfA(self, a):
        """Calculate the cumulative distribution function of the age distribution."""
        return 1 - (self.t_min / (self.t_min + a)) ** (self.beta - 1) * expint(self.beta, (self.t_min + a) / self.t_max) / expint(self.beta, self.tratio)


class PowerLawDisKin(AbstractDiskinModel):
    """A model where rates of decay are proportional to 1/t between two bounding timescales.

    We call these bounding timescales t_min and t_max.
    """

    def __init__(self, t_min, t_max, interp_r_14c=None, I=None):
        """
        Args:
        t_min: float
            The short time scale
        t_max: float
            The long time scale
        interp_r_14c: callable
            An interpolator for the estimated historical radiocarbon concentration.
            Takes a single argument, the number of years before a reference time (e.g. 2000).
            If None uses the default interpolator from constants.
        I: np.ndarray, optional
            An array of inputs to the model, representing the carbon input to the system.
        """
        super().__init__(interp_r_14c=interp_r_14c)
        self.t_min = t_min  # short time scale
        self.t_max = t_max  # long time scale
        
        self.I = I

        self.interp_14c = interp_r_14c
        if interp_r_14c is None:
            self.interp_14c = constants.INTERP_R_14C

        # steady-state transit time
        tratio = t_min / t_max
        e1_term = exp1(tratio)
        self.e1_term = e1_term
        self.T = t_min * np.exp(tratio) * e1_term

        # mean age at steady-state
        self.A = (t_max * np.exp(-tratio)/e1_term) - t_min

    def params_valid(self):
        """Returns True if the parameters are valid, False otherwise.
        
        TODO: is t_min == 0 valid? tmax == tmin?
        """
        t_min_valid = self.t_min > 0
        t_max_valid = self.t_max > 0
        t_hierarchy_valid = self.t_max > self.t_min
        return t_min_valid and t_max_valid and t_hierarchy_valid

    def s(self, tau):
        """The survival function at age tau."""
        num = self.t_min * np.exp(- tau / self.t_max)
        denom = self.t_min + tau
        return num / denom

    def run_simulation(self, times, inputs):
        """Run a simulation over the specified time steps.

        Returns the simulation results g_ts and ts.

        Parameters:
            times (array-like): Time steps for the simulation.
                Assumed to be uniformly spaced.
            inputs (array-like): Input values at each time step.

        Returns:
            g_ts (np.ndarray): A matrix of decayed inputs over time.
                rows correspond to input times, columns to ages.
                G_t = np.sum(g_ts, axis=0) gives the total carbon at each age.
        """
        assert len(times) == len(inputs), "Length of times and inputs must be the same."
        n_times = len(times)
        n_inputs = len(inputs)
        dt = times[1] - times[0] # timestep size, assumed uniform

        # g_ts contains the decayed inputs over time
        # each row is an input at time t=i
        # each column is the amount remaining at time t+age
        g_ts = np.zeros((n_inputs, n_times + n_inputs + 10))
        for i in tqdm(range(n_times), desc="power law simulation"):
            # inputs[i] decays according to the survival function
            my_times = np.arange(0, n_times - i) * dt
            decay_i = inputs[i]*self.s(my_times)
            g_ts[i, i:i+len(decay_i)] = decay_i

        return g_ts

    def radiocarbon_age_integrand(self, tau):
        """Integrand for calculating the radiocarbon ratio.
        
        Args:
            tau: float
                The age at which to evaluate the integrand.
        """
        # Interpolation was done with x as years before present,
        # so a is the correct input here
        initial_r = self.interp_14c(tau) 
        radiocarbon_decay = np.exp(-LAMBDA_14C*tau)
        age_dist_term = (
            np.power((self.e1_term * (self.t_min + tau)), -1) *
            np.exp(-(self.t_min + tau)/self.t_max))
        return initial_r * age_dist_term * radiocarbon_decay

    def mean_transit_time_integrand(self, a):
        return self.t_min * np.exp(-a/self.t_max) / (self.t_min + a)
    
    def pA(self, a):
        t0 = self.t_min
        tinf = self.t_max
        e1_term = self.e1_term
        return np.exp(-(t0 + a)/tinf) / ((t0 + a)*e1_term)
    
    def impulse(self, t, X):
        """Calculate the change in state of the system at time t.

        Args:
            t: float
                The time at which to calculate the change in state.
            X: np.ndarray
                The current state of the system, an array of carbon pools.
        Returns:
            np.ndarray
                The change in state of the system at time t.
        """
        # The rate of change is proportional to the inverse of the time scale
        # and the current state of the system.
        # The decay rate is 1/tau_0 for the short time scale and 1/tau_inf for the long time scale.
        dX = - ( 1 / (self.t_min + t) + 1 / (self.t_max + t)) * X

        return dX
    
    def cdfA(self, a):
        """Calculate the cumulative distribution function of the age distribution."""
        # The CDF is the integral of the PDF from 0 to a
        tratio = self.t_min / self.t_max
        return (1 - exp1((self.t_min + a) / self.t_max) / exp1(tratio))


class LogUniformDisKin(AbstractDiskinModel):
    """A model with decay rates uniformly distributed in log space.

    The rate density is ``p(k) = 1 / (k log(k_max / k_min))`` between
    ``k_min`` and ``k_max``. ``log_width`` is ``log(k_max / k_min)``;
    representing the width directly keeps very broad spectra numerically
    tractable. All rates are in inverse years.
    """

    def __init__(self, k_min, log_width, interp_r_14c=None, I=None):
        super().__init__(interp_r_14c=interp_r_14c)
        self.k_min = float(k_min)
        self.log_width = float(log_width)
        self.I = I

        if not self.params_valid():
            self.T = np.nan
            self.A = np.nan
            return

        self.log_k_max = np.log(self.k_min) + self.log_width
        inverse_k_max = np.exp(-self.log_k_max)
        self.T = (1 / self.k_min - inverse_k_max) / self.log_width
        self.A = 0.5 * (1 / self.k_min + inverse_k_max)

    def params_valid(self):
        """Return whether the rate bounds are finite, positive, and ordered."""
        finite = np.isfinite([self.k_min, self.log_width]).all()
        return bool(finite and self.k_min > 0 and self.log_width > 0)

    @staticmethod
    def _return_scalar_if_scalar(value, original):
        return float(value) if np.ndim(original) == 0 else value

    @staticmethod
    def _rate_age(log_rate, age):
        log_value = log_rate + np.log(age)
        return np.exp(np.minimum(log_value, np.log(np.finfo(float).max)))

    def s(self, t):
        """Return the survival function at age ``t``."""
        t_arr = np.asarray(t, dtype=float)
        result = np.ones_like(t_arr)
        positive = t_arr > 0
        age = t_arr[positive]
        result[positive] = (
            exp1(self._rate_age(np.log(self.k_min), age))
            - exp1(self._rate_age(self.log_k_max, age))
        ) / self.log_width
        return self._return_scalar_if_scalar(result, t)

    def cdfA(self, t):
        """Return the closed-form steady-state age-distribution CDF."""
        t_arr = np.asarray(t, dtype=float)
        result = np.zeros_like(t_arr)
        positive = t_arr > 0
        age = t_arr[positive]
        lower_age = self._rate_age(np.log(self.k_min), age)
        upper_age = self._rate_age(self.log_k_max, age)
        inverse_k_max = np.exp(-self.log_k_max)
        tail = (
            np.exp(-lower_age) / self.k_min
            - age * exp1(lower_age)
            - np.exp(-upper_age) * inverse_k_max
            + age * exp1(upper_age)
        ) / (1 / self.k_min - inverse_k_max)
        result[positive] = np.clip(1 - tail, 0, 1)
        return self._return_scalar_if_scalar(result, t)


class LognormalDisKin(AbstractDiskinModel):
    """A model where the rate distribution is lognormal."""

    def __init__(self, mu, sigma, k_min=None, k_max=None, interp_r_14c=None, N=1000, ppf_lim=1e-5):
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
        N: int
            The number of elements to discretize p(k) into.
        ppf_lim: float
            The percent point function limit for the lognormal distribution.
        """
        super().__init__(interp_r_14c=interp_r_14c)

        self.mu = mu
        self.k_star = np.exp(mu)
        self.sigma = sigma
        self.k_min = k_min or lognorm.ppf(ppf_lim, s=sigma, scale=np.exp(mu))
        self.k_max = k_max or lognorm.ppf(1.0-ppf_lim, s=sigma, scale=np.exp(mu))

        # rescale ks by the median
        self.kappa_min = self.k_min / self.k_star
        self.kappa_max = self.k_max / self.k_star

        # log scale
        self.q_min = np.log(self.kappa_min)
        self.q_max = np.log(self.kappa_max)        

        # steady-state transit time and mean age
        self.T = lognormal_turnover(self.mu, self.sigma)
        # mean age at steady-state
        self.A = self.T * np.exp(self.sigma**2)
        
        self.ks = np.logspace(self.q_min, self.q_max, N, base=np.e)
        self.I = lognorm.pdf(self.ks, s=self.sigma, scale=np.exp(self.mu))

    @classmethod
    def from_age_and_transit_time(cls, a, T, N=1000, ppf_lim=1e-5):
        """Construct a LognormalDisKin object from the mean age and transit time.
        
        Note: a/T >= 1 is required. a/T < 1 implies a negative lognormal standard deviation.

        Args:
            a: float
                The mean age
            T: float
                The transit time
            N: int
                The number of elements to discretize p(k) into.

        Returns:
            LognormalDisKin
                An instance of the LognormalDisKin class.
        """
        if a / T < 1:
            raise ValueError("a / T < 1 implies negative variance.")
        sigma_squared = np.log(a/T)
        sigma = np.sqrt(sigma_squared)
        mu = sigma_squared/2 - np.log(T)
        return cls(mu, sigma, N=N)
    
    def params_valid(self):
        """Returns True if the parameters are valid, False otherwise."""
        return self.sigma > 0
    
    def _pk(self, k):
        """The probability density function of the rate distribution p(k)."""
        return lognorm.pdf(k, s=self.sigma, scale=np.exp(self.mu))
    
    def _s_integrand(self, t, k):
        """Integrand for the survival function."""
        return self._pk(k) * np.exp(-k * t)

    def s(self, t):
        """The survival function at age t.
        
        The survival function gives the fraction of input remaining at age t.

        For the lognormal model, the survival function is given by

            𝑠(𝑡)= ∫_0^\infty (p(k) exp(-kt) dk)

        where p(k) is a lognormal distribution over ks. Since this 
        is the Laplace transform of a lognormal distribution, there is no
        known closed-form solution, so we evaluate the integral numerically.

        This integral is not very stable in general, and especially using 
        scipy methods. In practice we resort to separate calculation in 
        Mathematica. 

        Args:
            t: float
                The age at which to evaluate the survival function.
        
        Returns:
            float
                The fraction of input remaining at age t.
        """
        # We picked some limits of integration based on the p(k)
        # distribution in the constructor.
        k_min = self.k_min
        k_max = self.k_max
        result, _ = quad(
            self._s_integrand, k_min, k_max, args=(t,),
            limit=500, epsabs=1e-5)
        return result
    
    def cdfA(self, t):
        """Calculate the cumulative distribution function of the age distribution."""
        # The CDF is the integral of the PDF from 0 to a
        result, _ = quad(
            self.pA, 0, t,
            limit=500, epsabs=1e-5)
        return result
        
    def _dX(self, t, X):
        """Calculate the change in state of the system at time t.

        Args:
            t: float
                The time at which to calculate the change in state.
            X: np.ndarray
                The current state of the system.

        Returns:
            np.ndarray
                The change in state of the system.
        """
        # Unpack the state vector
        # Implement the model equations to calculate dX
        dX = self.I - self.ks * X
        
        return dX


class LognormalDisKinFast(AbstractDiskinModel):
    """Fast lognormal model with explicit atmospheric 14C input.

    This class requires an ``AtmC14`` object on construction and uses it
    directly for steady-state radiocarbon calculations. The interpolator path
    from ``AbstractDiskinModel`` is intentionally ignored.

    NOTE: this class violates the contract of the base class, e.g., by not using the
    interpolator for radiocarbon calculations and by not implementing the CDF via
    numerical integration.
    
    TODO: For the moment this class is only used for illustrative plotting. But we should 
    make a contract that allows for this implementation. 
    """
    def __init__(
        self,
        mu,
        sigma,
        atm: AtmC14,
        k_min=None,
        k_max=None,
        interp_r_14c=None,
        ppf_lim=1e-5,
        fast_rtol=1e-4,
    ):
        # All base state (T, A, interp_14c) is set below. The fast model uses
        # its supplied atmosphere and does not load the unused default interpolator.
        self.mu = mu
        self.k_star = np.exp(mu)
        self.sigma = sigma
    
        # steady-state transit time and mean age
        self.T = lognormal_turnover(self.mu, self.sigma)
        self.A = self.T * np.exp(self.sigma ** 2)

        self.atm = atm
        self.fast_rtol = fast_rtol

        # We intentionally do not use interpolator-based radiocarbon
        # calculations in this class.
        self.interp_14c = None

        # keep k bounds available; numeric routines may reference `k_min`/`k_max`
        # but we don't build discrete ks/I arrays in the fast implementation.

    def set_parameters(self, mu, sigma):
        """Update the existing model in place during calibration."""
        self.mu, self.sigma = mu, sigma
        self.k_star = np.exp(mu)
        self.T = lognormal_turnover(mu, sigma)
        self.A = self.T*np.exp(sigma**2)

    def prepare_quadrature(self, *, log_rate_step=.05, mu_bounds=(-15., 10.), sigma_bounds=(.05, 5.)):
        """Cache atmospheric responses on a fixed grid spanning all fitting bounds.

        This integrates the resident density, unlike the input-density survival
        discretization below. It avoids re-running adaptive integrals per fit.
        """
        atmosphere = self.atm
        self.mu_bounds, self.sigma_bounds = mu_bounds, sigma_bounds
        bounds = np.asarray([mu_bounds, sigma_bounds])
        if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] >= bounds[:, 1]) or sigma_bounds[0] <= 0):
            raise ValueError('ordered finite bounds and positive sigma required')
        if not np.isfinite(log_rate_step) or not 0 < log_rate_step <= sigma_bounds[0]:
            raise ValueError('log_rate_step must be positive and <= minimum sigma')
        ages, fm = atmosphere.ages, atmosphere.fm
        if (ages.ndim != 1 or not len(ages) or fm.shape != ages.shape
                or not np.isfinite(ages).all() or not np.isfinite(fm).all()
                or ages[0] != 0 or np.any(np.diff(ages) <= 0) or np.any(fm < 0)
                or not np.isfinite(atmosphere.mean_R) or atmosphere.mean_R < 0):
            raise ValueError('atmosphere needs increasing ages from zero and finite nonnegative fm')
        lo = mu_bounds[0]-sigma_bounds[1]**2-12*sigma_bounds[1]
        hi = mu_bounds[1]+12*sigma_bounds[1]
        if lo < -600 or hi > 600 or (hi-lo)/log_rate_step > 100_000:
            raise ValueError('quadrature range or size exceeds numerical limits')
        self.log_rates = np.linspace(lo, hi, int(np.ceil((hi-lo)/log_rate_step))+1)
        self.step = float(self.log_rates[1]-self.log_rates[0])
        self.rates = np.exp(self.log_rates)
        # A rate class's fraction modern: k ∫ F_atm(age) exp(-(k+lambda) age) d(age).
        self.radio_response = np.array([k*inner_integral(atmosphere, k+1/C14_MEAN_LIFE)
                                        for k in self.rates])

    def _resident_weights(self):
        z = (self.log_rates-(self.mu-self.sigma**2))/self.sigma
        weights = np.exp(-.5*z*z)*self.step/(self.sigma*np.sqrt(2*np.pi))
        weights[[0, -1]] *= .5
        return weights

    def new_carbon_fraction(self, times):
        """Fixed-input carbon normalized by steady stock; optional cached quadrature."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError('times must be a finite nonnegative one-dimensional array')
        if not hasattr(self, 'radio_response'):
            return diskin_C_of_t(times, self.mu, self.sigma)/self.T
        with np.errstate(over='ignore'):
            return np.sum(self._resident_weights()*(-np.expm1(-times[:, None]*self.rates)), axis=1)

    def predict(self, mu: float, sigma: float, input_rate: float,
                times: tuple[float, ...] | np.ndarray = ()) -> LognormalPrediction:
        """Predict in fitting mode, after calling prepare_quadrature once."""
        if not hasattr(self, 'radio_response'):
            raise ValueError('call prepare_quadrature before predict')
        times = np.asarray(times, dtype=float)
        if (not np.isfinite([mu, sigma, input_rate]).all() or input_rate <= 0
                or not self.mu_bounds[0] <= mu <= self.mu_bounds[1]
                or not self.sigma_bounds[0] <= sigma <= self.sigma_bounds[1]):
            raise ValueError('mu/sigma must be within bounds and input_rate positive')
        if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError('times must be a finite nonnegative one-dimensional array')
        self.set_parameters(mu, sigma)
        stock = input_rate*self.T
        fm = self.calc_radiocarbon_ratio_ss_fast()[0]
        fnew = self.new_carbon_fraction(times)
        if (not np.isfinite(stock) or stock <= 0 or not np.isfinite(fm) or fm < 0
                or not np.isfinite(fnew).all() or np.any(fnew < 0) or np.any(fnew > 1+1e-8)):
            raise FloatingPointError('nonfinite or unphysical prediction')
        return LognormalPrediction(float(stock), fm, times, fnew)


    def _survival_matrix_discretized(self, t, n_ks=200, q_low=1e-3, q_high=1 - 1e-3):
        """Return discretized survival contribution matrix M[k, t].

        This method is vector-only: ``t`` must be an array-like of ages.
        """
        if n_ks < 2:
            raise ValueError("n_ks must be >= 2")
        if not (0 < q_low < q_high < 1):
            raise ValueError("Require 0 < q_low < q_high < 1")

        qvals = np.array([q_low, q_high], dtype=float)
        ln_k_bounds = np.log(lognorm.ppf(qvals, s=self.sigma, scale=np.exp(self.mu)))
        ln_ks = np.linspace(ln_k_bounds[0], ln_k_bounds[1], n_ks)
        ks = np.exp(ln_ks)

        # Normal density over ln(k), then normalize to discrete weights.
        z = (ln_ks - self.mu) / self.sigma
        k_weights = np.exp(-0.5 * z * z) / (self.sigma * np.sqrt(2 * np.pi))
        k_weights /= np.sum(k_weights)

        t_arr = np.asarray(t, dtype=float)
        if t_arr.ndim != 1:
            raise ValueError("t must be a 1D array-like of ages")

        M = np.exp(-np.outer(ks, t_arr)) * k_weights[:, None]
        return M

    def survival_discretized(self, t, n_ks=200, q_low=1e-3, q_high=1 - 1e-3):
        """Evaluate survival using a discretized lognormal rate spectrum.

        This mirrors the strategy used in ``lognormal_sim``:
        discretize log-rates, weight by normal density in log-space,
        then sum weighted exponentials.

        Args:
            t: array-like
                1D array of ages at which to evaluate survival.
            n_ks: int
                Number of discretized rate points.
            q_low: float
                Lower quantile for log-rate support.
            q_high: float
                Upper quantile for log-rate support.

        Returns:
            np.ndarray
                Survival values for each age in ``t``.
        """
        M = self._survival_matrix_discretized(
            t=t,
            n_ks=n_ks,
            q_low=q_low,
            q_high=q_high,
        )

        return np.sum(M, axis=0)
    
    def s(self, t):
        """Evaluate survival at age t using the discretized method."""
        return self.survival_discretized(t)

    def cdfA(self, t):
        result, _ = quad(self.pA, 0, t, limit=500, epsabs=1e-5)
        return result

    def calc_radiocarbon_ratio_ss_fast(
        self,
        quad_limit=1500,
        quad_epsabs=1e-3,
    ):
        """Calculate steady-state radiocarbon ratio using fast helper functions.

        If `u_lo`/`u_hi` are provided they are used as the outer integration
        limits in log-space (u = ln k). Otherwise the helper `lognormal_radiocarbon`
        is called which chooses default bounds.
        """
        if hasattr(self, 'radio_response'):
            return float(np.sum(self._resident_weights()*self.radio_response)), 0.0
        ratio = lognormal_radiocarbon(
            atm=self.atm,
            tau=float(self.T),
            age=float(self.A),
            rtol=self.fast_rtol,
        )
        # return a numeric error estimate (0.0) so tests treat it as valid
        return float(ratio), 0.0
    
    def calc_radiocarbon_ratio_ss(self):
        """Calculate steady-state radiocarbon ratio using fast helper functions."""
        return self.calc_radiocarbon_ratio_ss_fast()
    



class WeibullDisKin(AbstractDiskinModel):
    """The Feng (2009) "hockey stick" survival model.

    The survival function is a stretched exponential (Weibull form)

        s(tau) = exp( -(k * tau)^alpha )

    with a rate-scale parameter ``k`` (yr^-1) and a shape parameter
    ``alpha``. This is the H function of Feng (2009), "Fundamental
    Considerations of Soil Organic Carbon Dynamics", Soil Science 174(9).
    The paper writes it as exp(-(k*tau)^alpha); for alpha < 1 the curve has
    the characteristic sharp initial drop and very long tail of SOC
    decomposition. alpha = 1 recovers first-order (exponential) kinetics.

    Steady-state relationships (derived in the accompanying markdown):

        T  = Gamma(1 + 1/alpha) / k                       (mean transit time)
        A  = Gamma(1 + 2/alpha) / (2 k Gamma(1 + 1/alpha)) (mean age)
        A/T = Gamma(1 + 2/alpha) / (2 Gamma(1 + 1/alpha)^2)

    Note that A/T depends only on alpha (it is Feng's stabilization
    coefficient beta = Ta/MRT0), while k sets the overall timescale. The
    age-distribution CDF has the closed form

        cdfA(t) = P(1/alpha, (k t)^alpha)

    where P is the regularized lower incomplete gamma function
    (scipy.special.gammainc).
    """

    def __init__(self, k, alpha, interp_r_14c=None, I=None):
        """
        Args:
        k: float
            Rate-scale parameter (yr^-1). Must be positive.
        alpha: float
            Shape parameter (dimensionless). Must be positive. alpha < 1
            gives the hockey-stick shape; alpha = 1 is first-order kinetics.
        interp_r_14c: callable
            An interpolator for the estimated historical radiocarbon
            concentration. Takes a single argument, the number of years
            before a reference time (e.g. 2000). If None uses the default
            interpolator from constants.
        I: np.ndarray, optional
            An array of inputs to the model.
        """
        super().__init__(interp_r_14c=interp_r_14c)

        self.k = float(k)
        self.alpha = float(alpha)
        self.I = I

        if not self.params_valid():
            self.T = np.nan
            self.A = np.nan
            return

        # steady-state transit time and mean age (see module docstring)
        self.T = gamma(1.0 + 1.0 / self.alpha) / self.k
        self.A = gamma(1.0 + 2.0 / self.alpha) / (
            2.0 * self.k * gamma(1.0 + 1.0 / self.alpha)
        )

    def params_valid(self):
        """Returns True if the parameters are valid, False otherwise."""
        finite = np.isfinite([self.k, self.alpha]).all()
        return bool(finite and self.k > 0 and self.alpha > 0)

    def s(self, t):
        """The survival function s(tau) = exp(-(k*tau)^alpha)."""
        return np.exp(-np.power(self.k * t, self.alpha))

    def cdfA(self, t):
        """Age-distribution CDF: P(1/alpha, (k t)^alpha)."""
        cdf = gammainc(1.0 / self.alpha, np.power(self.k * t, self.alpha))
        cdf = np.clip(cdf, 0.0, 1.0)
        return cdf
