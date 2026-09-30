import itertools as it
import numpy as np
import matplotlib.pyplot as plt

from soil_diskin.continuum_models import LognormalDisKin, PowerLawDisKin
from soil_diskin.lognormal import run_diskin_fast
from tqdm import tqdm
from scipy.stats import lognorm, norm

import viz

"""
Makes figure 1: Schematic of continuum model structure and age dist.

Intended to be run from the project root directory. 
"""

# use the style file
plt.style.use('notebooks/style.mpl')


np.random.seed(1234)
colors = viz.color_palette()


def plot_survival_fn(ax, my_model, ages2plot, color):
    """Plot the survival function."""
    # an impulse at tau = 0 
    markerline, stemlines, _ = ax.stem(0, 1, color)
    markerline.set_markersize(0)
    stemlines.set_linewidth(1)

    ax.plot(ages2plot, my_model.s_vec(ages2plot),
            color=color, lw=1, label=f'$\\mu={my_model.mu}$, $\\sigma={my_model.sigma}$')


def plot_ss_age_distribution(ax, my_model, max_t, color):
    """Plot the CDF age distribution at steady state.
    
    Args:
        ax: matplotlib axis to plot on
        my_model: the continuum model to use
        max_t: the maximum time to plot
        color: the color to use for the plot
    """
    # C(t) for a pool started empty under constant unit input equals the
    # steady-state carbon younger than t, so C(t) / C_ss is the age CDF.
    # C_ss = input * T for unit input.
    ages, C = run_diskin_fast(my_model.T, my_model.A, tmax=max_t)
    ax.semilogx(ages, C / my_model.T, color=color, lw=1, label='analytic')

# %%
# Plot figure 1
if __name__ == "__main__":
    print("Plotting figure 1...")
    mosaic = 'ABC'
    fig, axs = plt.subplot_mosaic(mosaic, layout='constrained',
                                  figsize=(4.76, 1.5), dpi=300)
    fig.get_layout_engine().set(hspace=0.1, wspace=0.1)

    # typical values based on fig. S2
    mu = 1
    sigma = 2.5
    model = LognormalDisKin(mu=mu, sigma=sigma, ppf_lim=1e-7)

    # Panel A -- lognormal continuum over time
    ax = axs['A']
    # Each input pulse is associated with a lognormal distribution of decay rates such
    # that the amount of material with rate constant k = lognormal(k; mu, sigma) dk.
    # Since we plot against ln k, we plot the density of ln k, which is the
    # underlying normal(ln k; mu, sigma) and integrates to 1 over ln k.
    # Each k bin decays exponentially with time t as exp(-k*t).
    ln_k_bins = np.linspace(np.log(model.k_min), np.log(model.k_max), 1000)
    k_bins = np.exp(ln_k_bins)
    initial_distribution = norm.pdf(ln_k_bins, loc=model.mu, scale=model.sigma)
    ts = [0, 1, 10] # years
    distribution_over_time = np.array([initial_distribution * np.exp(-k_bins * t) for t in ts])

    # check that initial distribution integrates equals distribution_over_time[0]
    assert np.isclose(np.sum(initial_distribution - distribution_over_time[0]), 0.0, atol=1e-6)

    color_order = [colors[x] for x in ['dark_grey', 'dark_blue', 'blue']]
    labels = r'input,$\tau=1$ y,$\tau=10$ y'.split(',')
    for i, t in enumerate(ts):
        print(f"i={i}: Plotting residual a distribution at t={t} yr...")
        ax.plot(ln_k_bins, distribution_over_time[i],
                label=labels[i], lw=1, color=color_order[i])

    ax.set_xlabel(r'$\ln k$')
    ax.set_ylabel(r'density, $p(\ln k)$')
    ax.set_title(r'an aging continuum')
    ax.legend(loc='upper right', fontsize=5, frameon=False, handlelength=0.8, handletextpad=0.3)

    # Survival function for different mu / sigma
    ax = axs['B']
    ages2plot = np.arange(0, 100, 0.1) # finer age steps
    models = [
        LognormalDisKin(mu=0, sigma=2.5, ppf_lim=1e-7),
        LognormalDisKin(mu=1, sigma=2, ppf_lim=1e-7),
        model, # mu=1, sigma=2.5
    ]
    model_color_order = [colors[x] for x in ['purple', 'dark_brown', 'dark_grey']]
    for i, my_model in enumerate(models):
        color = color_order[i]
        plot_survival_fn(ax, my_model, ages2plot, color=model_color_order[i])
    ax.set_xlim(-1, 20)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(r'fraction remaining, $s(\tau)$')
    ax.set_xlabel(r'age $\tau$ (yr)')
    ax.set_title(r'survival function, s($\tau$)')
    ax.legend(loc='upper right', fontsize=5, frameon=False, handlelength=0.8, handletextpad=0.3)

    # Panel C -- CDF of age distribution at steady state for the models above
    ax = axs['C']
    for i, my_model in enumerate(models):
        plot_ss_age_distribution(ax, my_model, max_t=1e5, color=model_color_order[i])
    ax.set_title(r'steady-state SOC age dist.')
    ax.set_ylabel('cumulative fraction')
    ax.set_xlabel(r'age $\tau$ (yr)')

    # Label the subplots A,B,C
    for i, label in enumerate('ABC'):
        axs[label].text(
            -0.3, 1.1, label, transform=axs[label].transAxes,
            fontsize=7, va='top', ha='left')

    plt.savefig('figures/fig1_new.png', dpi=300)