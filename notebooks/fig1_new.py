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
    mosaic = 'ABCD'
    fig, axs = plt.subplot_mosaic(mosaic, layout='constrained',
                                  figsize=(6.3, 1.5), dpi=300,
                                  width_ratios=[0.8, 1, 1, 1])
    fig.get_layout_engine().set(hspace=0.1, wspace=0.1)

    # typical values based on fig. S2
    mu = 1
    sigma = 2.5
    model = LognormalDisKin(mu=mu, sigma=sigma, ppf_lim=1e-8)

    # Panel A -- schematic of the parallel continuum model structure
    ax = axs['A']
    ax.set_axis_off()
    ax.set_title(r'parallel continuum model')
    # draw the diagram in an inset that extends past the axes so it can
    # use the space below the title and next to the panel label
    diagram_ax = ax.inset_axes([-0.25, -0.15, 1.25, 1.3])
    diagram = plt.imread('graphics/parallel_continuum_model.png')
    diagram_ax.imshow(diagram, interpolation='antialiased')
    diagram_ax.set_axis_off()

    # Panel B -- lognormal continuum over time
    ax = axs['B']
    # Each input pulse is associated with a lognormal distribution of decay rates such
    # that the amount of material with rate constant k = lognormal(k; mu, sigma) dk.
    # Since we plot against ln k, we plot the density of ln k, which is the
    # underlying normal(ln k; mu, sigma) and integrates to 1 over ln k.
    # Each k bin decays exponentially with time t as exp(-k*t).
    ln_k_bins = np.linspace(np.log(model.k_min), np.log(model.k_max), 1000)
    k_bins = np.exp(ln_k_bins)
    initial_distribution = norm.pdf(ln_k_bins, loc=model.mu, scale=model.sigma)
    initial_amount = 500 # g C, a typical input to 1 m2 in 1 yr
    ts = [0, 1, 10] # years
    distribution_over_time = np.array([initial_amount * initial_distribution * np.exp(-k_bins * t) for t in ts])

    color_order = [colors[x] for x in ['dark_grey', 'dark_blue', 'blue']]
    labels = r'input,$\tau=1$ yr,$\tau=10$ yr'.split(',')
    for i, t in enumerate(ts):
        print(f"i={i}: Plotting residual a distribution at t={t} yr...")
        ax.plot(ln_k_bins, distribution_over_time[i],
                label=labels[i], lw=1, color=color_order[i])

    ax.set_xlabel(r'$\ln ( k \cdot 1 \text{ yr})$')
    ax.set_ylabel(r'carbon density (g C)')
    ax.set_title(r'an aging continuum')
    ax.legend(loc='upper right', fontsize=5, frameon=False, handlelength=0.8, handletextpad=0.3)
    # write mu and sigma on the plot
    ax.text(0.85, 0.07, f'$\mu={mu}$,\n$\sigma={sigma}$', transform=ax.transAxes,
            fontsize=5, va='bottom', ha='center')

    # Panel C -- survival function for different mu / sigma
    ax = axs['C']
    ages2plot = np.arange(0, 100, 0.1) # finer age steps
    models = [
        LognormalDisKin(mu=0, sigma=2.5, ppf_lim=1e-8),
        LognormalDisKin(mu=1, sigma=2, ppf_lim=1e-8),
        model, # mu=1, sigma=2.5
    ]
    model_color_order = [colors[x] for x in ['purple', 'dark_brown', 'dark_grey']]
    for i, my_model in enumerate(models):
        color = color_order[i]
        plot_survival_fn(ax, my_model, ages2plot, color=model_color_order[i])
    ax.set_xlim(-1, 20)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(r'fraction remaining, $s(\tau)$')
    ax.set_yticks([0, 0.5, 1.0], ['0', '0.5', '1.0'])
    ax.set_xlabel(r'age $\tau$ (yr)')
    ax.set_title(r'survival function, s($\tau$)')
    ax.legend(loc='upper right', fontsize=5, frameon=False, handlelength=0.8, handletextpad=0.3)

    # Panel D -- CDF of age distribution at steady state for the models above
    ax = axs['D']
    for i, my_model in enumerate(models):
        plot_ss_age_distribution(ax, my_model, max_t=1e5, color=model_color_order[i])
    ax.set_title(r'steady-state SOC age dist.')
    ax.set_ylabel('cumulative fraction')
    ax.set_yticks([0, 0.5, 1.0], ['0', '0.5', '1.0'])
    ax.set_xlabel(r'age $\tau$ (yr)')

    # Label the subplots A-D
    for label in 'ABCD':
        axs[label].text(
            -0.3, 1.1, label, transform=axs[label].transAxes,
            fontsize=7, va='top', ha='left')

    plt.savefig('figures/fig1_new.png', dpi=300)