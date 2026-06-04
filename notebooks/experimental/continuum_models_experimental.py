import numpy as np

from scipy.integrate import quad
from scipy.special import exp1, gamma, gammaincc, log_ndtr
from scipy.stats import lognorm
from soil_diskin.constants import LAMBDA_14C, INTERP_R_14C, GAMMA
from tqdm import tqdm
from soil_diskin.continuum_models import AbstractDiskinModel

def expint(n, x):
    """Generalized exponential integral E_n(x) for real n < 1 and x > 0.

    Uses the identity E_n(x) = x^(n-1) * Γ(1-n, x), where the upper
    incomplete gamma function is computed as gammaincc(1-n, x) * gamma(1-n).
    """
    return x ** (n - 1) * gammaincc(1 - n, x) * gamma(1 - n)


def _return_scalar_if_scalar(result, original_input):
    """Return a Python float when the corresponding input was scalar."""
    if np.asarray(original_input).ndim == 0:
        return float(np.asarray(result))
    return result


def _logsubexp(log_a, log_b):
    """Return log(exp(log_a) - exp(log_b)) for log_a >= log_b."""
    ratio = np.exp(np.minimum(log_b - log_a, 0.0))
    return log_a + np.log1p(-ratio)


def _log_ndtr_diff(z_high, z_low):
    """Return log(Phi(z_high) - Phi(z_low))."""
    z_high = np.asarray(z_high, dtype=float)
    z_low = np.asarray(z_low, dtype=float)
    use_right_tail = z_low > 0

    log_a = np.where(use_right_tail, log_ndtr(-z_low), log_ndtr(z_high))
    log_b = np.where(use_right_tail, log_ndtr(-z_high), log_ndtr(z_low))
    result = _logsubexp(log_a, log_b)
    result = np.where(z_high > z_low, result, -np.inf)

    if result.ndim == 0:
        return float(result)
    return result


class GaussianDisKin(AbstractDiskinModel):
    """A bounded Gaussian rate-distribution continuum model.

    The rate density is a normal distribution with mean mu and standard
    deviation sigma, truncated and renormalized on [k_min, k_max].
    The default bounds are in yr^-1.
    """

    def __init__(
        self,
        mu,
        sigma,
        k_min=1e-4,
        k_max=1.0,
        interp_r_14c=None,
        I=None,
        quad_points=512,
    ):
        super().__init__(interp_r_14c=interp_r_14c)

        self.mu = float(mu)
        self.sigma = float(sigma)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.I = I
        self.quad_points = int(quad_points)

        if not self.params_valid(include_normalization=False):
            self.T = np.nan
            self.A = np.nan
            self.log_normalization = np.nan
            self._quad_ks = np.array([])
            self._quad_mass_weights = np.array([])
            return

        self.z_min = (self.k_min - self.mu) / self.sigma
        self.z_max = (self.k_max - self.mu) / self.sigma
        self.log_normalization = _log_ndtr_diff(self.z_max, self.z_min)

        if not self.params_valid(include_normalization=True):
            self.T = np.nan
            self.A = np.nan
            self._quad_ks = np.array([])
            self._quad_mass_weights = np.array([])
            return

        self._setup_quadrature()
        self.T = float(np.sum(self._quad_mass_weights / self._quad_ks))
        self.A = float(
            np.sum(self._quad_mass_weights / self._quad_ks**2) / self.T
        )

    def params_valid(self, include_normalization=True):
        """Returns True if the parameters are valid, False otherwise."""
        finite_params = np.isfinite(
            [self.mu, self.sigma, self.k_min, self.k_max]
        ).all()
        basic_valid = (
            finite_params
            and self.sigma > 0
            and self.k_min > 0
            and self.k_max > self.k_min
            and self.quad_points > 0
        )
        if not include_normalization:
            return basic_valid

        return (
            basic_valid
            and np.isfinite(self.log_normalization)
            and self.log_normalization > -np.inf
        )

    def _setup_quadrature(self):
        """Build deterministic quadrature nodes for truncated-normal averages."""
        if self.z_min < 12 and self.z_max > -12:
            z_min = max(self.z_min, -12.0)
            z_max = min(self.z_max, 12.0)
        else:
            z_min = self.z_min
            z_max = self.z_max

        z_nodes, z_weights = np.polynomial.legendre.leggauss(self.quad_points)
        z_midpoint = 0.5 * (z_max + z_min)
        z_half_width = 0.5 * (z_max - z_min)
        z_nodes = z_midpoint + z_half_width * z_nodes
        z_weights = z_half_width * z_weights

        ks = self.mu + self.sigma * z_nodes
        log_phi = -0.5 * z_nodes**2 - 0.5 * np.log(2 * np.pi)
        mass_weights = z_weights * np.exp(log_phi - self.log_normalization)

        in_bounds = (ks >= self.k_min) & (ks <= self.k_max)
        self._quad_ks = ks[in_bounds]
        self._quad_mass_weights = mass_weights[in_bounds]

    def _pk(self, k):
        """The probability density function of the rate distribution p(k)."""
        k = np.asarray(k, dtype=float)
        z = (k - self.mu) / self.sigma
        log_pdf = (
            -0.5 * z**2
            - np.log(self.sigma)
            - 0.5 * np.log(2 * np.pi)
            - self.log_normalization
        )
        pdf = np.where(
            (k >= self.k_min) & (k <= self.k_max),
            np.exp(log_pdf),
            0.0,
        )
        return _return_scalar_if_scalar(pdf, k)

    def s(self, t):
        """The survival function at age t."""
        t = np.asarray(t, dtype=float)
        z_min_shifted = (self.k_min - self.mu + self.sigma**2 * t) / self.sigma
        z_max_shifted = (self.k_max - self.mu + self.sigma**2 * t) / self.sigma
        log_survival = (
            -self.mu * t
            + 0.5 * self.sigma**2 * t**2
            + _log_ndtr_diff(z_max_shifted, z_min_shifted)
            - self.log_normalization
        )
        survival = np.exp(log_survival)
        survival = np.clip(survival, 0.0, 1.0)
        return _return_scalar_if_scalar(survival, t)

    def cdfA(self, a):
        """Calculate the cumulative distribution function of the age distribution."""
        a = np.asarray(a, dtype=float)
        a_flat = a.reshape(-1)
        cdf_terms = (
            self._quad_mass_weights[:, None]
            * (-np.expm1(-self._quad_ks[:, None] * a_flat[None, :]))
            / self._quad_ks[:, None]
        )
        cdf = (np.sum(cdf_terms, axis=0) / self.T).reshape(a.shape)
        cdf = np.clip(cdf, 0.0, 1.0)
        return _return_scalar_if_scalar(cdf, a)


class LogUniformRateDisKin(AbstractDiskinModel):
    """A continuum model where p(k) is proportional to 1/k on [k_min, k_max]."""

    def __init__(self, k_min, k_max, interp_r_14c=None, I=None):
        super().__init__(interp_r_14c=interp_r_14c)

        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.I = I

        if not self.params_valid():
            self.log_k_ratio = np.nan
            self.T = np.nan
            self.A = np.nan
            return

        self.log_k_ratio = np.log(self.k_max / self.k_min)
        self.T = (
            (1 / self.k_min - 1 / self.k_max) / self.log_k_ratio
        )
        self.A = 0.5 * (1 / self.k_min + 1 / self.k_max)

    def params_valid(self):
        """Returns True if the parameters are valid, False otherwise."""
        finite_params = np.isfinite([self.k_min, self.k_max]).all()
        return finite_params and self.k_min > 0 and self.k_max > self.k_min

    def _pk(self, k):
        """The probability density function of the rate distribution p(k)."""
        k = np.asarray(k, dtype=float)
        pdf = np.where(
            (k >= self.k_min) & (k <= self.k_max),
            1 / (k * self.log_k_ratio),
            0.0,
        )
        return _return_scalar_if_scalar(pdf, k)

    def s(self, t):
        """The survival function at age t."""
        t = np.asarray(t, dtype=float)
        survival = np.ones_like(t, dtype=float)
        positive = t > 0
        survival[positive] = (
            exp1(self.k_min * t[positive])
            - exp1(self.k_max * t[positive])
        ) / self.log_k_ratio
        survival = np.clip(survival, 0.0, 1.0)
        return _return_scalar_if_scalar(survival, t)

    @staticmethod
    def _cdf_antiderivative(k, a):
        """Antiderivative of (1 - exp(-k*a)) / k^2 with respect to k."""
        x = k * a
        result = np.zeros_like(a, dtype=float)
        positive = a > 0
        result[positive] = (
            (np.exp(-x[positive]) - 1) / k
            - a[positive] * exp1(x[positive])
        )
        return result

    def cdfA(self, a):
        """Calculate the cumulative distribution function of the age distribution."""
        a = np.asarray(a, dtype=float)
        numerator = self._cdf_antiderivative(
            self.k_max, a
        ) - self._cdf_antiderivative(self.k_min, a)
        cdf = numerator / (self.T * self.log_k_ratio)
        cdf = np.clip(cdf, 0.0, 1.0)
        return _return_scalar_if_scalar(cdf, a)

