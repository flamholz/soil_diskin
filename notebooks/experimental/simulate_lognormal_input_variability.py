"""Simulate sites whose lognormal SOC pool receives noisy annual inputs.

Each simulated site draws mu and sigma of its lognormal rate distribution
independently and uniformly. Each range has the midpoint of the calibrated
values in the 03b table and range_factor (default 2) times their width. The
site's mean input I is the NPP of a row drawn with replacement from the same
table. Every year the site receives I*(1 + eps), with eps ~ N(0, cv^2) drawn
independently each year and site.

Each pool starts at the analytic steady state for constant input I at the start
of the atmospheric 14C record (55,050 years before 2000), with the record's
pre-history value tail_fm (its oldest value), and is stepped one year at a time to 2000. Starting
at steady state matters: the stock sits in the slow tail of the rate
distribution, so spinning up from empty would take ~10^5 years per site.

Since eps has mean zero, the expected stock is I*T. With --cv 0 the simulated
Fm reproduces the steady-state Fm of LognormalDisKin.calc_radiocarbon_ratio_ss.

Outputs one row per simulated site: stock, Fm and delta-14C in 2000, the mean
simulated NPP, and the expected (noise-free) stock and Fm for comparison.

Usage:
    python notebooks/experimental/simulate_lognormal_input_variability.py \
        [--n-sites 100] [--cv 0.05] [--range-factor 2] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # Support direct execution.

from soil_diskin.continuum_models import LognormalDisKin
from soil_diskin.lognormal import C14_MEAN_LIFE
from soil_diskin.radiocarbon_utils import AtmC14, load_atm14c

# Parameter bounds of the 03b fit (calibration.lognormal_spec); within them
# LognormalDisKin resolves the steady-state radiocarbon (fm_expected).
BOUNDS = {'mu': (-15., 10.), 'sigma': (.01, 5.)}
CALIBRATION = ROOT/'results/03_calibrate_models/lognormal_model_optimization_results.csv'
OUTPUT = ROOT/'results/experimental/lognormal_input_variability.csv'


class LognormalInputSimulator:
    """Lognormal continuum pools for a batch of sites, stepped one year at a time.

    Each site's rate spectrum is discretized with Gauss-Legendre nodes in
    u = ln k. The range covers the mass-weighted distribution of ln k,
    N(mu - sigma^2, sigma), which holds the stock, as well as the input-weighted
    one, N(mu, sigma): [mu - sigma^2 - width*sigma, mu + width*sigma].

    Within each year the input and atmospheric Fm are constant, so each bin is
    advanced exactly:
        C   <- exp(-k) C   + w u (1 - exp(-k)) / k
        C14 <- exp(-k') C14 + w u F (1 - exp(-k')) / k',   k' = k + lambda
    where w is the bin's share of the input, u the year's input and F the
    year's atmospheric Fm. C14 is in units of stock times Fm.
    """

    def __init__(self, mu, sigma, atm: AtmC14, n_nodes=400, width=8.):
        """
        Args:
            mu, sigma: array-like, one value per site; parameters of ln k (k in 1/yr).
            atm: atmospheric 14C record with unit spacing in years before 2000.
            n_nodes: number of Gauss-Legendre nodes (rate bins) per site.
            width: half-width of the ln k range in units of sigma.
        """
        if np.any(np.diff(atm.ages) != 1):
            raise ValueError('atmospheric record must have annual spacing')
        mu = np.atleast_1d(np.asarray(mu, float))[:, None]
        sigma = np.atleast_1d(np.asarray(sigma, float))[:, None]
        x, w = np.polynomial.legendre.leggauss(n_nodes)
        lo, hi = mu - sigma**2 - width*sigma, mu + width*sigma
        half_width = (hi - lo)/2
        u = lo + half_width*(x + 1)
        self.atm = atm
        self.k = np.exp(u)  # rows index sites, columns index rate bins
        self.weights = half_width*w*norm.pdf(u, mu, sigma)  # share of input per bin
        self.T = np.exp(sigma[:, 0]**2/2 - mu[:, 0])  # stock per unit input at steady state
        # Exact one-year update factors (see the class docstring). decay is the share of a
        # bin's carbon left after a year. gain is the carbon one unit of input adds over the
        # year, net of decay within the year: w * int_0^1 exp(-k (1 - s)) ds = w (1 - exp(-k))/k.
        # It tends to w for slow bins and to w/k for fast ones, so the steady state is exactly
        # w u/k, with no time-step error. -expm1(-k) is 1 - exp(-k) without the cancellation
        # that 1 - np.exp(-k) suffers for very slow bins (k ~ 1e-12/yr).
        # The 14C factors are the same with k + lambda: turnover plus radioactive decay.
        k14 = self.k + 1/C14_MEAN_LIFE
        self.decay = np.exp(-self.k)
        self.gain = self.weights*-np.expm1(-self.k)/self.k
        self.decay_14C = np.exp(-k14)
        self.gain_14C = self.weights*-np.expm1(-k14)/k14
        self._k14 = k14

    @property
    def n_years(self):
        """Number of annual steps from the start of the record to 2000."""
        return len(self.atm.ages) - 1

    def steady_state(self, I):
        """Per-bin stock and C14 at steady state for constant input I (one per site)."""
        I = np.asarray(I, float)[:, None]
        return self.weights*I/self.k, self.weights*I*self.atm.tail_fm/self._k14

    def run(self, inputs, I_initial=None):
        """Step every site from the start of the record to 2000.

        Args:
            inputs: (n_sites, n_years) annual inputs, oldest year first.
            I_initial: input rate defining the initial steady state; defaults
                to the mean of each site's inputs.

        Returns:
            (stock, fm) in 2000, one value per site.
        """
        inputs = np.asarray(inputs, float)
        if inputs.shape != (len(self.k), self.n_years):
            raise ValueError(f'inputs must have shape {(len(self.k), self.n_years)}')
        I_initial = inputs.mean(axis=1) if I_initial is None else I_initial
        C, C14 = self.steady_state(I_initial)
        # Year n (oldest first) spans ages [a_i, a_i + 1) with i = len - 2 - n,
        # and AtmC14 holds fm[i] over that interval.
        fm_by_year = self.atm.fm[-2::-1]
        for n in range(self.n_years):
            u = inputs[:, n, None]
            C = self.decay*C + self.gain*u
            C14 = self.decay_14C*C14 + self.gain_14C*(u*fm_by_year[n])
        stock = C.sum(axis=1)
        return stock, C14.sum(axis=1)/stock


def sample_parameters(calibrated, n_sites, range_factor, rng):
    """Draw mu and sigma independently and uniformly from widened calibrated ranges.

    Prints each sampling range next to BOUNDS and flags ranges that exceed them.
    """
    parameters = {}
    for name in ['mu', 'sigma']:
        lo, hi = calibrated[name].min(), calibrated[name].max()
        half_width = range_factor*(hi - lo)/2
        low, high = (lo + hi)/2 - half_width, (lo + hi)/2 + half_width
        bound_lo, bound_hi = BOUNDS[name]
        flag = '' if bound_lo <= low and high <= bound_hi else '  <-- OUTSIDE BOUNDS'
        print(f'{name}: calibrated [{lo:.3f}, {hi:.3f}], sampled [{low:.3f}, {high:.3f}] '
              f'(x{range_factor:g}), bounds [{bound_lo:g}, {bound_hi:g}]{flag}')
        parameters[name] = rng.uniform(low, high, n_sites)
    return pd.DataFrame(parameters)


def main(n_sites=100, cv=.05, range_factor=2., seed=0, n_nodes=400, calibration=CALIBRATION,
         output=OUTPUT):
    calibrated = pd.read_csv(calibration)
    calibrated = calibrated[calibrated.calibration_status.eq('calibrated')]
    rng = np.random.default_rng(seed)
    rows = rng.integers(len(calibrated), size=n_sites)
    sites = sample_parameters(calibrated, n_sites, range_factor, rng).assign(
        npp_source_row=calibrated.index[rows], NPP=calibrated.NPP.to_numpy()[rows])
    atm = load_atm14c(ROOT/'data/14C_atm_annot.csv')

    simulator = LognormalInputSimulator(sites.mu, sites.sigma, atm, n_nodes=n_nodes)
    I = sites.NPP.to_numpy()/1000  # g C/m²/yr -> kg C/m²/yr, matching the stocks
    inputs = I[:, None]*(1 + cv*rng.standard_normal((n_sites, simulator.n_years)))
    stock, fm = simulator.run(inputs, I_initial=I)

    result = sites.assign(
        NPP_mean_simulated=inputs.mean(axis=1)*1000, cv=cv, range_factor=range_factor,
        stock=stock, stock_expected=I*simulator.T, fm=fm, delta14C=(fm - 1)*1000,
        fm_expected=[LognormalDisKin(m, s, atm=atm).calc_radiocarbon_ratio_ss()[0]
                     for m, s in zip(sites.mu, sites.sigma)])
    result['stock_relative_error'] = result.stock/result.stock_expected - 1
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print(f'Saved {n_sites} simulated sites to {output}')
    print('stock / expected - 1: mean {:.2e}, max |.| {:.2e}'.format(
        result.stock_relative_error.mean(), result.stock_relative_error.abs().max()))
    print('fm - expected:        mean {:.2e}, max |.| {:.2e}'.format(
        (result.fm - result.fm_expected).mean(), (result.fm - result.fm_expected).abs().max()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-sites', type=int, default=300, help='number of sites to simulate')
    parser.add_argument('--cv', type=float, default=.05, help='SD of annual input noise relative to I')
    parser.add_argument('--range-factor', type=float, default=2.5,
                        help='width of the mu and sigma sampling ranges relative to the calibrated ranges')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-nodes', type=int, default=400, help='rate bins per site')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    main(**vars(parser.parse_args()))
