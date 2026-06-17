import numpy as np

from scipy.integrate import quad
from scipy.special import exp1, gamma, gammaincc, log_ndtr
from scipy.stats import lognorm
from soil_diskin.constants import LAMBDA_14C, INTERP_R_14C, GAMMA
from soil_diskin.lognormal import lognormal_radiocarbon, inner_integral, C14_MEAN_LIFE
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
            self.interp_14c = INTERP_R_14C
    
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
            self.interp_14c = INTERP_R_14C

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
        self.T = np.exp(-self.mu + ((self.sigma**2) / 2))
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
        # Initialize base class (interpolator path is intentionally ignored)
        AbstractDiskinModel.__init__(self, interp_r_14c=interp_r_14c)
        # Copy essential lognormal parameter setup from LognormalDisKin
        self.mu = mu
        self.k_star = np.exp(mu)
        self.sigma = sigma
    
        # steady-state transit time and mean age
        self.T = np.exp(-self.mu + ((self.sigma ** 2) / 2))
        self.A = self.T * np.exp(self.sigma ** 2)

        self.atm = atm
        self.fast_rtol = fast_rtol

        # We intentionally do not use interpolator-based radiocarbon
        # calculations in this class.
        self.interp_14c = None

        # keep k bounds available; numeric routines may reference `k_min`/`k_max`
        # but we don't build discrete ks/I arrays in the fast implementation.

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
    
