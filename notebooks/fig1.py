import numpy as np
import matplotlib.pyplot as plt

from soil_diskin.continuum_models import LognormalDisKin
from scipy.stats import norm

import viz

"""
Makes figure 1: Schematic of continuum model structure and age dist.

Intended to be run from the project root directory. 
"""

# use the style file
plt.style.use('notebooks/style.mpl')


def plot_ss_age_distribution(ax, my_model, max_t, color):
    """Plot the CDF age distribution at steady state.
    
    Args:
        ax: matplotlib axis to plot on
        my_model: the continuum model to use
        max_t: the maximum time to plot
        color: the color to use for the plot
    """
    ages = np.logspace(-1, np.log10(max_t), 1000)
    ax.semilogx(ages, my_model.cdfA(ages), color=color, lw=1.5)

# %%
# Plot figure 1
if __name__ == "__main__":
    print("Plotting figure 1...")
    colors = viz.color_palette()
    mosaic = 'ABCD'
    fig, axs = plt.subplot_mosaic(mosaic, layout='constrained',
                                  figsize=(7.24, 1.7), dpi=300)
    fig.get_layout_engine().set(hspace=0.1, wspace=0.1)

    # typical values based on fig. S2
    mu = 1
    sigma = 2.5
    focal_model = LognormalDisKin(mu=mu, sigma=sigma)
    models = [
        LognormalDisKin(mu=-1, sigma=2.5),
        LognormalDisKin(mu=1, sigma=1.5),
        focal_model, # mu=1, sigma=2.5
    ]
    model_color_order = [colors[x] for x in ['purple', 'dark_brown', 'dark_grey']]

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

    # Panel B -- lognormal continuum of inputs
    ax = axs['B']
    # Each input pulse is associated with a lognormal distribution of decay rates such
    # that the amount of material with rate constant k = lognormal(k; mu, sigma) dk.
    # Since we plot against ln k, we plot the density of ln k, which is the
    # underlying normal(ln k; mu, sigma) and integrates to 1 over ln k.
    # Each k bin decays exponentially with time t as exp(-k*t).
    # Span the focal model's 1e-9 and 1 - 1e-9 quantiles of ln k.
    z = norm.isf(1e-9)
    ln_k_bins = np.linspace(mu - z * sigma, mu + z * sigma, 10000)
    k_bins = np.exp(ln_k_bins)
    initial_distributions = [norm.pdf(ln_k_bins, loc=my_model.mu, scale=my_model.sigma) for my_model in models]
    
    for i, my_model in enumerate(models):
        ax.semilogx(k_bins, initial_distributions[i], color=model_color_order[i],
                    lw=1.5, label=f'$\\mu={my_model.mu}$, $\\sigma={my_model.sigma}$')

    ax.set_xlabel(r'$k$ (yr$^{-1}$)')
    ax.set_ylabel(r'density')
    ax.set_title(r'a continuum of inputs')
    #ax.legend(loc='upper right', fontsize=5, frameon=False, handlelength=0.8, handletextpad=0.3)

    # Panel C -- survival function for different mu / sigma
    ax = axs['C']
    ages2plot = np.arange(0, 100, 0.1) # finer age steps

    for i, my_model in enumerate(models):
        color = model_color_order[i]
        ax.plot(ages2plot, my_model.s(ages2plot),
                color=color, lw=1.5,
                label=f'$\\mu={my_model.mu}$, $\\sigma={my_model.sigma}$')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(r'fraction remaining, $s(\tau)$')
    ax.set_yticks([0, 0.5, 1.0], ['0', '0.5', '1.0'])
    ax.set_xlabel(r'age $\tau$ (yr)')
    ax.set_title(r'survival function, s($\tau$)')
    ax.legend(loc='upper right', fontsize=5, framealpha=0.8, handlelength=0.8, handletextpad=0.3)

    # Panel D -- CDF of age distribution at steady state for the models above
    ax = axs['D']
    for i, my_model in enumerate(models):
        plot_ss_age_distribution(ax, my_model, max_t=1e5, color=model_color_order[i])
    ax.set_title(r'steady-state SOC age dist.')
    ax.set_ylabel('cumulative fraction')
    ax.set_yticks([0, 0.5, 1.0], ['0', '0.5', '1.0'])
    ax.set_xlabel(r'age $\tau$ (yr)')
    ax.set_xlim(1e-1, 1e5)

    # Label the subplots A-D
    for label in 'ABCD':
        axs[label].text(
            -0.3, 1.1, label, transform=axs[label].transAxes,
            fontsize=7, va='top', ha='left')

    plt.savefig('figures/fig1.png', dpi=300)