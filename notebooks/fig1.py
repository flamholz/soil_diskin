# %%
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from constants import LAMBDA_14C, C14_DATA, INTERP_R_14C
from matplotlib.patches import FancyArrowPatch
from scipy.special import exp1, expi
from scipy import stats
from scipy import integrate
from scipy.interpolate import interp1d
from tqdm import tqdm

import viz

# use the style file
plt.style.use('notebooks/style.mpl')

# TODO: should probably move all this funcitonality into models.py
class PowerLawSimulation:
    def __init__(self, t_min, t_max):
        self.t_min = t_min
        self.t_max = t_max

        self.g_ts = None
        self.ts = None
        self.G_t = None
        self.interp_14c = interp1d(
            C14_DATA.years_before_2000, C14_DATA.R_14C,
            kind='zero', fill_value='extrapolate')

    def turnover_time(self):
        """Calculate the mean turnover time of the system."""
        # calculate the exponential integral E1(t_min/t_max) 
        E1_term = exp1(self.t_min / self.t_max)
        return self.t_min * np.exp(self.t_min / self.t_max) * E1_term
    
    def survival_fn(self, ages):
        """Define a power-law survival function.
        
        Parameters:
            ages (array-like): Array of ages.
        """
        float_ages = ages.astype(np.float64)
        num = self.t_min*np.exp(-float_ages/self.t_max)
        denom = float_ages + self.t_min
        return num / denom
    
    def age_distribution_pdf_analytic_ss(self, ages):
        """Calculate the age distribution function.
        
        Parameters:
            ages (array-like): Array of ages.
        """
        s = self.survival_fn(ages)
        return s / self.turnover_time()

    def age_distribution_cdf_analytic_ss(self, ages):
        """Calculate the cumulative distribution of
        the steady-state age distribution.

        Integral of PDF was calculated in Mathematica.

        CDF(ages) = (e^(tmin/tmax) tmin (E1(tmin/tmax) + E1((tmin + ages)/tmax)))/T

        Parameters:
            ages (array-like): Array of ages.
        """
        tmin, tmax = self.t_min, self.t_max
        T = self.turnover_time()
        Ei_term = expi(- tmin / tmax) - expi( -(tmin + ages)/tmax)
        return - np.exp(tmin/tmax) * tmin * Ei_term / T

    def radiocarbon_age_integrand(self, a):
        # Interpolation was done with x as years before present,
        # so a is the correct input here
        initial_r = self.interp_14c(a) 
        radiocarbon_decay = np.exp(-LAMBDA_14C*a)
        E1_term = exp1(self.t_min / self.t_max)
        age_dist_term = np.power((E1_term * (self.t_min + a)), -1) * np.exp(-(self.t_min + a)/self.t_max)
        return initial_r * age_dist_term * radiocarbon_decay
    
    def radiocarbon_ratio_ss(self):
        """Calculate the steady-state radiocarbon ratio."""
        integral, _ = integrate.quad(
            self.radiocarbon_age_integrand, 0, np.inf,
            limit=1500, epsabs=1e-3)
        return integral
        
    def run(self, inputs):
        """Run the simulation over the specified time steps.

        Note: assumes inputs occur at each time step, so that 
        the timesteps = np.arange(len(inputs)).

        Stores state in this object.

        Parameters:
            inputs (array-like): Input values at each time step.
        """
        n_time_steps = len(inputs)
        ts = np.arange(n_time_steps)

        # g_ts contains the decayed inputs over time
        # each row is an input at time t=i
        # each column is the amount remaining at time t+age        
        g_ts = np.zeros((n_time_steps, n_time_steps))

        g_ts = np.zeros((len(inputs), len(ts)+len(inputs) + 10))
        for i in range(n_time_steps):
            decay_i = inputs[i]*self.survival_fn(np.arange(n_time_steps))
            g_ts[i, i:i+len(decay_i)] = decay_i
            
        self.g_ts = g_ts.copy()
        self.ts = ts
        self.G_t = np.sum(self.g_ts, axis=0)

        return g_ts, ts

np.random.seed(1234)
colors = viz.color_palette()
color_name_order = np.array([k for k, _ in colors.items()])
np.random.shuffle(color_name_order)
#color_name_order = 'dark_blue,light_green,purple,yellow,red,light_blue,pale_green,dark_purple,yellow,light_red'.split(',')
color_order = [colors[n] for n in color_name_order]

# %%
# inputs are a train of impulses J_t 
# that are normally distributed with mean 10 and sd 2
J_t = np.random.normal(10, 2, 5000)
ages = np.arange(2000.0, 0.1)

my_sim = PowerLawSimulation(t_min=1, t_max=1000)
g_ts, ts = my_sim.run(J_t)

def plot_inputs(ax):
    """Plot the inputs over time as a stem plot."""
    plt.sca(ax)

    for i, (J, color) in enumerate(zip(J_t[:5], color_order[:5])):
        markerline, stemlines, _ = plt.stem(
            i, J, color)
        #markerline.set_markeredgecolor('k')
        #markerline.set_markeredgewidth(0.5)
        markerline.set_markersize(4)
        stemlines.set_linewidth(1)
        
    plt.xlabel(r'time $t$')
    plt.xticks(np.arange(0, 8, 5))
    plt.xlim(-1, 8)
    plt.ylim(0, 16.5)
    plt.yticks(np.arange(0, 17, 5))
    plt.text(5.5, 5, '...', fontsize=12, fontweight='bold',
             ha='center', va='center')
    plt.ylabel(r'carbon inputs $J(t)$')
    plt.title('inputs over time')

def plot_survival_fn(ax, ages2plot):
    """Plot the survival function."""
    plt.sca(ax)

    # an impulse at tau = 0 
    markerline, stemlines, _ = plt.stem(0, 1, color_order[0])
    markerline.set_markersize(4)
    stemlines.set_linewidth(1)

    plt.plot(ages2plot, my_sim.survival_fn(ages2plot),
             color=color_order[0], lw=1)
    plt.xlabel(r'age $\tau$')
    plt.xlim(-3, 50)
    plt.ylim(0, 1.1)
    plt.ylabel(r'remaining carbon $s(\tau)$')
    plt.title('decay with age')

def plot_independent_decays(ax, J_t, my_sim):
    """Plot the independent decays of inputs over time."""
    plt.sca(ax)

    # Only have 10 colors -- plot the first 10
    for i, (J, color) in enumerate(zip(J_t[:10], color_order[:10])):
        markerline, stemlines, _ = plt.stem(
            i, J, color)
        markerline.set_markersize(4)
        stemlines.set_linewidth(1)

    for i in range(10):
        decay_i = J_t[i] * my_sim.survival_fn(ages2plot)
        plt.plot(ages2plot + i, decay_i, color=color_order[i], lw=1,
                zorder=-1)

    plt.xlabel('time')
    plt.ylabel(r'$J(t-\tau)\cdot s(\tau)$')
    plt.text(15.5, 5, '...', fontsize=12, fontweight='bold',
             ha='center', va='center')
    plt.xlim(-1, 20)
    plt.ylim(0, 16.5)
    plt.yticks(np.arange(0, 17, 5))
    plt.title('inputs decay independently')

def plot_total_stocks(ax, my_sim, my_t):
    """Plot the total stocks as the sum of residual inputs over time."""
    plt.sca(ax)
    
    # stocks from each input over time
    g_ts = my_sim.g_ts.copy()

    nts = g_ts.shape[1]
    njs = g_ts.shape[0]
    bottom = np.zeros(nts)
    ts = np.arange(nts)
    for i in range(njs):
        color = color_order[i % len(color_order)]
        top = bottom + g_ts[i, :]
        plt.fill_between(ts, bottom, top, color=color, alpha=0.7, 
                         edgecolor='k', lw=0.2)
        bottom = top

    # overplot the total stocks line
    G_t = my_sim.G_t.copy()
    plt.plot(ts, G_t, color='black', lw=2)

    # annotate the line with a curved arrow
    ymax = G_t[my_t]
    arrowprops=dict(arrowstyle='->', connectionstyle='angle', lw=1)
    plt.annotate('SOC age dist.', xy=(my_t, ymax-10), xytext=(my_t - 5, ymax + 10),
                 arrowprops=arrowprops, fontsize=5.5, ha='center',
                 bbox=dict(boxstyle='square,pad=0.0', edgecolor='none', facecolor='None'))
    plt.plot([my_t, my_t], [0, ymax], color='k', linestyle='--', lw=1)

    plt.xlabel('time')
    plt.xlim(0, 50)
    plt.ylim(0, 60)
    plt.ylabel(r'total carbon stocks $G(t)$')
    plt.title(r'stocks = sum of residual inputs')

def plot_age_distribution(ax, my_sim, my_t):
    # simulated age distribution at time my_t
    plt.sca(ax)

    g_ts = my_sim.g_ts.copy()
    simulated_ages = my_t - my_sim.ts
    simulated_stocks = g_ts[:, my_t] / np.sum(g_ts[:, my_t])
    # remove negative ages and normalize
    SOC_ages = simulated_ages[:my_t]
    stocks = simulated_stocks[:my_t]
    normalized_stocks = stocks / np.sum(stocks)

    repeating_colors = it.cycle(color_order)
    repeating_colors = [next(repeating_colors) for _ in range(2*my_t)]
    plt.bar(SOC_ages + 0.5, normalized_stocks,
            color=repeating_colors, alpha=0.9, 
            edgecolor='k', lw=0.50)

    plt.xlim(0, my_t)
    plt.yticks(np.arange(0, 0.11, 0.05))
    plt.ylabel('proportion of stocks')
    plt.xlabel(r'age $\tau$')
    plt.title(f'age distribution at t = {my_t}')

def plot_ss_age_distribution_inset(ax, my_sim, my_t):
    """Plot the CDF age distribution at steady state."""
    max_age2plot = 50
    max_idx = max_age2plot * 10
    ages_fine = np.arange(0, my_t, 0.1)
    age_dist_cdf = my_sim.age_distribution_cdf_analytic_ss(ages_fine)
    ax.plot(ages_fine[:max_idx], age_dist_cdf[:max_idx],
            color='black', lw=1)

    # simulated age distribution at time my_t
    simulated_ages = my_t - my_sim.ts
    simulated_stocks = my_sim.g_ts[:, my_t] / np.sum(my_sim.g_ts[:, my_t])

    # remove negative ages (keep only ages >= 1) and reorder from youngest->oldest
    SOC_ages = simulated_ages[:my_t]
    stocks_truncated = simulated_stocks[:my_t]

    # Reverse so arrays are ordered from youngest (age small) to oldest (age large)
    SOC_ages = SOC_ages[::-1]
    stocks_truncated = stocks_truncated[::-1]

    # cumulative from youngest to oldest and normalize over the truncated range
    cumulative_stocks = np.cumsum(stocks_truncated)
    normalized_cumulative_stocks = cumulative_stocks / np.sum(stocks_truncated)

    ages2plot = SOC_ages[::5]
    stock2plot = normalized_cumulative_stocks[::5]
    # Plot a subset of points to avoid too much overlap
    ax.scatter(ages2plot, stock2plot, color='grey',
               s=10, edgecolors='w', lw=0.3, alpha=0.5,
               zorder=10)
    ax.set_xlim(0, max_age2plot)

# %%
mosaic = 'ABC\nDEF'
fig, axs = plt.subplot_mosaic(mosaic, layout='constrained',
                              figsize=(4.76, 3), dpi=300)

# Panel A -- schematic of three-pool model
ax = axs['A']

ax.set_xlim(0, 10)
ax.set_ylim(3, 10)
ax.axis('off') # Hide the axes

# add label (a)
#ax.text(1, 10, 'A', ha='center', va='center', fontsize=14)

# 1. Define and draw the boxes for the pools
fast_pool_rect = patches.Rectangle((3.1, 7.5), 3.8, 1, edgecolor='black', facecolor='white', lw=1.1)
slow_pool_rect = patches.Rectangle((3.1, 5.5), 3.8, 1, edgecolor='black', facecolor='white', lw=1.1)
passive_rect = patches.Rectangle((3.1, 3.5), 3.8, 1, edgecolor='black', facecolor='white', lw=1.1)

ax.add_patch(fast_pool_rect)
ax.add_patch(slow_pool_rect)
ax.add_patch(passive_rect)

# 2. Add text labels inside the boxes
ax.text(5, 8, 'fast pool', ha='center', va='center', fontsize=6)
ax.text(5, 6, 'slow pool', ha='center', va='center', fontsize=6)
ax.text(5, 4, 'passive', ha='center', va='center', fontsize=6)

# 3. Add the "input carbon" label
ax.text(0.1, 7, 'input\ncarbon', ha='center', va='center', fontsize=6)

# 4. Add all the arrows
# Dotted arrows for input carbon with solid arrowheads
# First arrow: draw dotted line, then solid arrowhead
#ax.plot([2.15, 2.85], [7.3, 7.86], 'k:', linewidth=1)  # dotted line
ax.annotate('', xy=(0.15, 2.85), xytext=(2.85, 7.85), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1, mutation_scale=10))

# Second arrow: draw dotted line, then solid arrowhead
#ax.plot([2.15, 2.85], [6.85, 6.15], 'k:', linewidth=1)  # dotted line
ax.annotate('', xy=(0.15, 2.85), xytext=(2.85, 6.15), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1, mutation_scale=10))

ax.plot([6.5, 6.5], [6.65, 7.3], 'k-', linewidth=1)  # dotted line
ax.annotate('', xy=(6.5, 7.5), xytext=(6.5, 6.5), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=10))

ax.plot([3.5, 3.5], [6.75, 7.4], 'k-', linewidth=1)  # dotted line
ax.annotate('', xy=(3.5, 6.5), xytext=(3.5, 7.5), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=10))

ax.plot([5, 5], [5.35, 4.65], 'k-', linewidth=1)  # dotted line
ax.annotate('', xy=(5, 4.5), xytext=(5, 5.5), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=10))

ax.text(5, 7, 'CO$_2$', ha='center', va='center', rotation=0, fontsize=6)
ax.text(6.5, 5, 'CO$_2$', ha='center', va='center', rotation=0, fontsize=6)

ax.annotate('', xy=(5.55, 7), xytext=(6.5, 6.65),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=.75, 
                          connectionstyle='angle3,angleA=75,angleB=0', mutation_scale=5))

# Add curved arrow from middle of rectangular arrow to CO2 label
ax.annotate('', xy=(4.55, 7), xytext=(3.5, 7.25),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=1, 
                          connectionstyle='angle3,angleA=75,angleB=0', mutation_scale=5))


ax.annotate('', xy=(7, 5), xytext=(8, 5.5),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=.75, 
                          connectionstyle='angle3,angleA=90,angleB=0', mutation_scale=5))


ax.annotate('', xy=(6, 5), xytext=(5, 5.25),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=.75, 
                          connectionstyle='angle3,angleA=90,angleB=0', mutation_scale=5))



# Rectangular CO2 arrow from passive to fast pool with filled arrowheads on both ends
# Draw the rectangular path as a line
ax.plot([7.3, 8, 8, 7.3], [4, 4, 8, 8], 'k-', linewidth=1)

# Add filled arrowheads at both ends
ax.annotate('', xy=(7, 4), xytext=(7.15, 4), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=20))
ax.annotate('', xy=(7, 8), xytext=(7.15, 8), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=20))

# Add curved arrow from middle of rectangular arrow to CO2 label
ax.annotate('', xy=(8.5, 6), xytext=(8, 6),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=.75, 
                          connectionstyle='angle3,angleA=90,angleB=25', mutation_scale=10))

# Add the CO2 label for the rectangular arrow
ax.text(8.5, 6, 'CO$_2$', ha='left', va='center', rotation=0, fontsize=6)

# Panel B -- inputs over time
ax = axs['B']
plot_inputs(ax)

# Panel C -- survival function for a single input
ages2plot = np.arange(0, 50, 0.1) # finer age steps
plot_survival_fn(axs['C'], ages2plot)

# Panel D -- independent decays of inputs over time
plot_independent_decays(axs['D'], J_t, my_sim)

# Panel E -- total stocks as the sum of residual inputs over time
my_t = 40
plot_total_stocks(axs['E'], my_sim, my_t=my_t)

# Panel F -- age distribution at time my_t
plot_age_distribution(axs['F'], my_sim, my_t=my_t)

# inset axes with analytic age distribution at steady state
inset_ax = axs['F'].inset_axes([0.5, 0.45, 0.4, 0.4])
plot_ss_age_distribution_inset(inset_ax, my_sim, my_t=2000)
inset_ax.tick_params(axis='both', which='major', labelsize=5,
                     size=2, pad=0.3)
inset_ax.set_title('steady-state', fontsize=5)
inset_ax.set_xlabel(r'age $\tau$', fontsize=5)
inset_ax.set_ylabel(r'CDF of $p_A(\tau)$', fontsize=5)

# Add panel labels
panel_labels = 'ABCDEF'
for key in panel_labels:
    axs[key].text(
        -0.3, 1.1, key, transform=axs[key].transAxes,
        fontsize=7, va='top', ha='left')

plt.savefig('figures/fig1.png', dpi=600)

# Make a presentation version of the above figure
# five panels in a row showing only B, C, D, E, F from above
mosaic = 'ABC\nDEF'
fig, axs = plt.subplot_mosaic(mosaic, layout='constrained',
                              figsize=(4.25, 3), dpi=300)

plot_inputs(axs['A'])

plot_survival_fn(axs['B'], ages2plot)

plot_independent_decays(axs['C'], J_t, my_sim)

plot_total_stocks(axs['D'], my_sim, my_t=my_t)

plot_age_distribution(axs['E'], my_sim, my_t=my_t)

plot_ss_age_distribution_inset(axs['F'], my_sim, my_t=2000)
plt.sca(axs['F'])
plt.title('steady-state age dist.')
plt.xlabel(r'age $\tau$')
plt.ylabel(r'CDF of $p_A(\tau)$')

plt.savefig('figures/fig1_presentation.png', dpi=600)


# %%
