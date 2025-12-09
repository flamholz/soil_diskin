import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from soil_diskin.continuum_models import PowerLawDisKin
from tqdm import tqdm

import viz

"""
Currently makes figures 1 and 2.

Figure 1: Schematic of model structure and age distribution calculation.

Figure 2: Model calibration procedure.

Intended to be run from the project root directory. 
"""

# use the style file
plt.style.use('notebooks/style.mpl')

np.random.seed(1234)
colors = viz.color_palette()
color_name_order = np.array([k for k, _ in colors.items()])
np.random.shuffle(color_name_order)
#color_name_order = 'dark_blue,light_green,purple,yellow,red,light_blue,pale_green,dark_purple,yellow,light_red'.split(',')
color_order = [colors[n] for n in color_name_order]

# %%
# inputs are a train of impulses J_t 
# that are normally distributed with mean 10 and sd 2
n_inputs = 50000
J_t = np.random.normal(10, 2, n_inputs)
ages = np.arange(2000.0, 0.1)

# Use the power law model to simulate decays for figures.
my_sim = PowerLawDisKin(t_min=1, t_max=1000)
ts = np.arange(n_inputs)
g_ts = my_sim.run_simulation(ts, J_t)

def plot_inputs(ax, J_t, title='inputs over time'):
    """Plot the inputs over time as a stem plot."""
    plt.sca(ax)

    for i, (J, color) in enumerate(zip(J_t[:5], color_order[:5])):
        markerline, stemlines, _ = plt.stem(
            i, J, color)
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
    if title:
        plt.title(title)

def plot_survival_fn(ax, ages2plot, title='decay with age'):
    """Plot the survival function."""
    plt.sca(ax)

    # an impulse at tau = 0 
    markerline, stemlines, _ = plt.stem(0, 1, color_order[0])
    markerline.set_markersize(4)
    stemlines.set_linewidth(1)

    plt.plot(ages2plot, my_sim.s(ages2plot),
             color=color_order[0], lw=1)
    plt.xlabel(r'age $\tau$')
    plt.xlim(-3, 50)
    plt.ylim(0, 1.1)
    plt.ylabel(r'remaining carbon $s(\tau)$')
    if title:
        plt.title(title)

def plot_independent_decays(ax, J_t, ages2plot, my_sim, title='inputs decay independently'):
    """Plot the independent decays of inputs over time."""
    plt.sca(ax)

    # Only have 10 colors -- plot the first 10
    for i, (J, color) in enumerate(zip(J_t[:10], color_order[:10])):
        markerline, stemlines, _ = plt.stem(
            i, J, color)
        markerline.set_markersize(4)
        stemlines.set_linewidth(1)

    for i in range(10):
        decay_i = J_t[i] * my_sim.s(ages2plot)
        plt.plot(ages2plot + i, decay_i, color=color_order[i], lw=1,
                zorder=-1)

    plt.xlabel('time $t$')
    plt.ylabel(r'$J(t-\tau)\cdot s(\tau)$')
    plt.text(15.5, 5, '...', fontsize=12, fontweight='bold',
             ha='center', va='center')
    plt.xlim(-1, 20)
    plt.ylim(0, 16.5)
    plt.yticks(np.arange(0, 17, 5))
    if title:
        plt.title(title)

def plot_total_stocks(ax, my_t, g_ts,
                      annotate_age_dist=True,
                      title='stocks = sum of residual inputs'):
    """Plot the total stocks as the sum of residual inputs over time.
    
    Args:
        ax: matplotlib axis to plot on
        my_t: time step to annotate
        g_ts: matrix of decayed inputs over time
    """
    plt.sca(ax)
    
    nts = g_ts.shape[1]
    njs = g_ts.shape[0]
    njs2plot = min(njs, 200)
    bottom = np.zeros(nts)
    ts = np.arange(nts)
    for i in tqdm(range(njs2plot), desc="plot total stocks"):
        color = color_order[i % len(color_order)]
        top = bottom + g_ts[i, :]
        plt.fill_between(ts, bottom, top, color=color, alpha=0.7, 
                         edgecolor='k', lw=0.2)
        bottom = top

    # overplot the total stocks line
    G_t = np.sum(g_ts, axis=0)
    plt.plot(ts[:njs2plot], G_t[:njs2plot], color='black', lw=2)

    # annotate the line with a curved arrow
    if annotate_age_dist:
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
    if title:
        plt.title(title)

def plot_age_distribution(ax, my_t, ts, g_ts, title=True):
    # simulated age distribution at time my_t
    plt.sca(ax)

    simulated_ages = my_t - ts
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
    plt.yticks([0.0, 0.1])
    plt.ylabel('proportion of stocks')
    plt.xlabel(r'age $\tau$')
    if title:
        plt.title(f'age distribution at t = {my_t}')

def plot_ss_age_distribution_inset(ax, my_sim, my_t, ts, g_ts):
    """Plot the CDF age distribution at steady state."""
    max_age2plot = 1800
    max_idx = max_age2plot * 10
    ages_fine = np.arange(0, my_t, 0.1)
    age_dist_cdf = my_sim.cdfA(ages_fine)
    ax.set_xscale('log')
    ax.plot(ages_fine[:max_idx], age_dist_cdf[:max_idx],
            color='black', lw=1, label='analytic')

    # simulated age distribution at time my_t
    simulated_ages = my_t - ts
    simulated_stocks = g_ts[:, my_t] / np.sum(g_ts[:, my_t])

    # remove negative ages (keep only ages >= 1) and reorder from youngest->oldest
    SOC_ages = simulated_ages[:my_t]
    stocks_truncated = simulated_stocks[:my_t]

    # Reverse so arrays are ordered from youngest (age small) to oldest (age large)
    SOC_ages = SOC_ages[::-1]
    stocks_truncated = stocks_truncated[::-1]

    # cumulative from youngest to oldest and normalize over the truncated range
    cumulative_stocks = np.cumsum(stocks_truncated)
    normalized_cumulative_stocks = cumulative_stocks / np.sum(stocks_truncated)

    # pick log spaced ages to plot from SOC_ages
    idxs2plot = np.power(2, np.arange(0, np.log2(len(SOC_ages)), 0.5)).astype(int)
    idxs2plot = idxs2plot[idxs2plot < len(SOC_ages)]

    ages2plot = SOC_ages[idxs2plot]
    stock2plot = normalized_cumulative_stocks[idxs2plot]
    # Plot a subset of points to avoid too much overlap
    ax.scatter(ages2plot, stock2plot, color='grey',
               s=10, edgecolors='w', lw=0.3, alpha=0.5,
               zorder=10, label='simulation')
    ax.set_xlim(1, max_age2plot)
    ax.set_xticks([10, 1000])

# %%
# Plot figure 1
print("Plotting figure 1...")
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
# Arrows from input carbon to fast and slow pools
# Arrow to fast pool
ax.annotate('', xy=(3.1, 8), xytext=(0.8, 7.25), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0.8, mutation_scale=5))

# Arrow to slow pool
ax.annotate('', xy=(3.1, 6), xytext=(0.8, 6.75), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0.8, mutation_scale=5))

# Respiration from fast pool
ax.plot([6.5, 6.5], [6.5, 7.5], 'k-', linewidth=0.8)
ax.annotate('', xy=(6.5, 7.5), xytext=(6.5, 6.5), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=7))

# Transfer from fast to slow pool
ax.plot([3.5, 3.5], [6.5, 7.5], 'k-', linewidth=0.8)
ax.annotate('', xy=(3.5, 6.5), xytext=(3.5, 7.5), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=7))

# Transfer from slow to passive pool
ax.plot([5, 5], [4.5, 5.5], 'k-', linewidth=0.8)
ax.annotate('', xy=(5, 4.5), xytext=(5, 5.5), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=7))

ax.text(5, 7, 'CO$_2$', ha='center', va='center', rotation=0, fontsize=6)
ax.text(6.5, 5, 'CO$_2$', ha='center', va='center', rotation=0, fontsize=6)

# Curved arrow from fast pool respiration to CO2 label
ax.annotate('', xy=(5.55, 7), xytext=(6.5, 6.65),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0.6, 
                          connectionstyle='angle3,angleA=75,angleB=0', mutation_scale=4))

# Curved arrow from fast->slow transfer to CO2 label
ax.annotate('', xy=(4.55, 7), xytext=(3.5, 7.25),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0.6, 
                          connectionstyle='angle3,angleA=75,angleB=0', mutation_scale=4))

# Respiration from slow pool
ax.annotate('', xy=(7, 5), xytext=(8, 5.5),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0.6, 
                          connectionstyle='angle3,angleA=90,angleB=0', mutation_scale=4))

# Curved arrow from slow->passive transfer to CO2 label
ax.annotate('', xy=(6, 5), xytext=(5, 5.25),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0.6, 
                          connectionstyle='angle3,angleA=90,angleB=0', mutation_scale=4))


# Rectangular CO2 arrow from passive to fast pool with filled arrowheads on both ends
# Draw the rectangular path as a line
ax.plot([7.3, 8, 8, 7.3], [4, 4, 8, 8], 'k-', linewidth=0.8)

# Add filled arrowheads at both ends
ax.annotate('', xy=(7, 4), xytext=(7.2, 4), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=7))
ax.annotate('', xy=(7, 8), xytext=(7.2, 8), 
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0, mutation_scale=7))

# Add curved arrow from middle of rectangular arrow to CO2 label
ax.annotate('', xy=(8.5, 6), xytext=(8, 6),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0.6, 
                          connectionstyle='angle3,angleA=90,angleB=25', mutation_scale=4))

# Add the CO2 label for the rectangular arrow
ax.text(8.5, 6, 'CO$_2$', ha='left', va='center', rotation=0, fontsize=6)
ax.text(8.5, 6, 'CO$_2$', ha='left', va='center', rotation=0, fontsize=6)

# Panel B -- inputs over time
ax = axs['B']
print('Plotting panel B: inputs over time...')
plot_inputs(ax, J_t)

# Panel C -- survival function for a single input
ages2plot = np.arange(0, 50, 0.1) # finer age steps
print('Plotting panel C: survival function...')
plot_survival_fn(axs['C'], ages2plot)

# Panel D -- independent decays of inputs over time
print('Plotting panel D: independent decays...')
plot_independent_decays(axs['D'], J_t, ages2plot, my_sim)

# Panel E -- total stocks as the sum of residual inputs over time
my_t = 40
print('Plotting panel E: total stocks...')
plot_total_stocks(axs['E'], my_t, g_ts)

# Panel F -- age distribution at time my_t
print('Plotting panel F: age distribution...')
plot_age_distribution(axs['F'], my_t, ts, g_ts)
# inset axes with analytic age distribution at steady state
inset_ax = axs['F'].inset_axes([0.5, 0.45, 0.4, 0.4])
plot_ss_age_distribution_inset(inset_ax, my_sim,
                               my_t=n_inputs-1, ts=ts, g_ts=g_ts)
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

# %%
# Make a presentation version of the above figure
# five panels in a row showing only B, C, D, E, F from above
print("Plotting presentation version of figure 1...")
mosaic = 'ABC\nDEF'
fig, axs = plt.subplot_mosaic(mosaic, layout='constrained',
                              figsize=(4.25, 2.75), dpi=300)

plot_inputs(axs['A'], J_t, title=None)

plot_survival_fn(axs['B'], ages2plot, title=None)

plot_independent_decays(axs['C'], J_t, ages2plot, my_sim, title=None)

plot_total_stocks(axs['D'], my_t=my_t, g_ts=g_ts,
                  annotate_age_dist=False,
                  title=None)

plot_age_distribution(axs['E'], my_t=my_t, ts=ts, g_ts=g_ts,
                      title=None)

plt.sca(axs['F'])
plot_ss_age_distribution_inset(
    axs['F'], my_sim, my_t=n_inputs-1,
    ts=ts, g_ts=g_ts)
plt.legend(loc=2, fontsize=5, frameon=False)
plt.sca(axs['F'])
#plt.title('steady-state age dist.')
plt.xlabel(r'age $\tau$')
plt.ylabel(r'CDF of $p_A(\tau)$')

plt.savefig('figures/fig1_presentation.png', dpi=600)

# %%
# Make a presentation version of the above figure
# five panels in a row showing only B, C, D, E, F from above
print("Plotting presentation version of figure 1...")
mosaic = 'ABCD'
fig, axs = plt.subplot_mosaic(mosaic, layout='constrained',
                              figsize=(7.25, 1.75), dpi=300)

plot_inputs(axs['A'], J_t, title=None)

plot_survival_fn(axs['B'], ages2plot, title=None)

plot_independent_decays(axs['C'], J_t, ages2plot, my_sim, title=None)

plot_total_stocks(axs['D'], my_t=my_t, g_ts=g_ts,
                  annotate_age_dist=False,
                  title=None)

# set font sizes for all axes and labels
for ax in axs.values():
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)

plt.savefig('figures/fig1_presentation2.png', dpi=600)