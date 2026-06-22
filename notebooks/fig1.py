import itertools as it
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.image import BboxImage
from soil_diskin.continuum_models import PowerLawDisKin
from tqdm import tqdm

import viz

"""
Makes figure 1: Schematic of model structure and age distribution calculation.

Intended to be run from the project root directory. 
"""

# use the style file
plt.style.use('notebooks/style.mpl')

np.random.seed(1234)
colors = viz.color_palette()
color_name_order = np.array([k for k, _ in colors.items()])
np.random.shuffle(color_name_order)
color_order = [colors[n] for n in color_name_order]

# %%
# Load precomputed simulation results from fig1_calcs.py
# (Computation moved to notebook/fig1_calcs.py for faster iteration)
data = np.load('results/fig1_calcs.npz')
n_inputs = int(data['n_inputs'])
J_t = data['J_t']
ts = data['ts']
g_ts = data['g_ts']

# NOTE: other figures now focus on the lognormal model,
# we can switch to the lognormal here if desired, but it
# emphasizes exactly the same concepts. 
my_sim = PowerLawDisKin(t_min=1, t_max=1000)

def plot_inputs(ax, J_t, title='inputs over time'):
    """Plot the inputs over time as a stem plot."""
    for i, (J, color) in enumerate(zip(J_t[:5], color_order[:5])):
        markerline, stemlines, _ = ax.stem(
            i, J, color)
        markerline.set_markersize(4)
        stemlines.set_linewidth(1)
        
    ax.set_xlabel(r'time $t$')
    ax.set_xticks(np.arange(0, 8, 5))
    ax.set_xlim(-1, 8)
    ax.set_ylim(0, 16.5)
    ax.set_yticks(np.arange(0, 17, 5))
    ax.text(5.5, 5, '...', fontsize=12, fontweight='bold',
             ha='center', va='center')
    ax.set_ylabel(r'carbon inputs $J(t)$')
    if title:
        ax.set_title(title)

def plot_survival_fn(ax, ages2plot, title='decay with age'):
    """Plot the survival function."""
    # an impulse at tau = 0 
    markerline, stemlines, _ = ax.stem(0, 1, color_order[0])
    markerline.set_markersize(4)
    stemlines.set_linewidth(1)

    ax.plot(ages2plot, my_sim.s(ages2plot),
             color=color_order[0], lw=1)
    ax.set_xlabel(r'age $\tau$')
    ax.set_xlim(-3, 50)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(r'remaining carbon $s(\tau)$')
    if title:
        ax.set_title(title)

def plot_independent_decays(ax, J_t, ages2plot, my_sim, title='inputs decay independently'):
    """Plot the independent decays of inputs over time."""
    # Only have 10 colors -- plot the first 10
    for i, (J, color) in enumerate(zip(J_t[:10], color_order[:10])):
        markerline, stemlines, _ = ax.stem(
            i, J, color)
        markerline.set_markersize(4)
        stemlines.set_linewidth(1)

    for i in range(10):
        decay_i = J_t[i] * my_sim.s(ages2plot)
        ax.plot(ages2plot + i, decay_i, color=color_order[i], lw=1,
                zorder=-1)

    ax.set_xlabel('time $t$')
    ax.set_ylabel(r'$J(t-\tau)\cdot s(\tau)$')
    ax.text(15.5, 5, '...', fontsize=12, fontweight='bold',
             ha='center', va='center')
    ax.set_xlim(-1, 20)
    ax.set_ylim(0, 16.5)
    ax.set_yticks(np.arange(0, 17, 5))
    if title:
        ax.set_title(title)

def plot_total_stocks(ax, my_t, g_ts,
                      annotate_age_dist=True,
                      title='stocks = sum of residual inputs'):
    """Plot the total stocks as the sum of residual inputs over time.
    
    Args:
        ax: matplotlib axis to plot on
        my_t: time step to annotate
        g_ts: matrix of decayed inputs over time
    """
    nts = g_ts.shape[1]
    njs = g_ts.shape[0]
    njs2plot = min(njs, 200)
    bottom = np.zeros(nts)
    ts = np.arange(nts)
    for i in tqdm(range(njs2plot), desc="plot total stocks"):
        color = color_order[i % len(color_order)]
        top = bottom + g_ts[i, :]
        ax.fill_between(ts, bottom, top, color=color, alpha=0.7, 
                         edgecolor='k', lw=0.2)
        bottom = top

    # overplot the total stocks line
    G_t = np.sum(g_ts, axis=0)
    ax.plot(ts[:njs2plot], G_t[:njs2plot], color='black', lw=2)

    # annotate the line with a curved arrow
    if annotate_age_dist:
        ymax = G_t[my_t]
        arrowprops=dict(arrowstyle='->', connectionstyle='angle', lw=1)
        ax.annotate('SOC age dist.', xy=(my_t, ymax-10), xytext=(my_t - 5, ymax + 8),
                    arrowprops=arrowprops, fontsize=5.5, ha='center',
                    bbox=dict(boxstyle='square,pad=0.0', edgecolor='none', facecolor='None'))
        ax.plot([my_t, my_t], [0, ymax], color='k', linestyle='--', lw=1)

    ax.set_xlabel('time')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 60)
    ax.set_ylabel(r'total carbon stocks $G(t)$')
    if title:
        ax.set_title(title)

def plot_age_distribution(ax, my_t, ts, g_ts, title=True):
    # simulated age distribution at time my_t
    simulated_ages = my_t - ts
    simulated_stocks = g_ts[:, my_t] / np.sum(g_ts[:, my_t])
    # remove negative ages and normalize
    SOC_ages = simulated_ages[:my_t]
    stocks = simulated_stocks[:my_t]
    normalized_stocks = stocks / np.sum(stocks)

    repeating_colors = it.cycle(color_order)
    repeating_colors = [next(repeating_colors) for _ in range(2*my_t)]
    ax.bar(SOC_ages + 0.5, normalized_stocks,
            color=repeating_colors, alpha=0.9, 
            edgecolor='k', lw=0.50)

    ax.set_xlim(0, my_t)
    ax.set_yticks([0.0, 0.1])
    ax.set_ylabel('proportion of stocks')
    ax.set_xlabel(r'age $\tau$')
    if title:
        ax.set_title(f'age distribution at t = {my_t}')

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
if __name__ == "__main__":
    print("Plotting figure 1...")
    mosaic = 'ABC\nDEF'
    fig, axs = plt.subplot_mosaic(mosaic, layout='constrained',
                                  figsize=(4.76, 3), dpi=300)

    # Panel A -- schematic of three-pool model loaded from a png
    ax = axs['A'] 
    # Clear the axes and turn off ticks/spines to reserve space for the diagram
    # we will paste the diagram in later. 
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

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
    key = 'A'
    axs[key].text(
            -0.3, 1.1, key, transform=axs[key].transAxes,
            fontsize=7, va='top', ha='left')
    
    panel_labels = 'BCDEF'
    for key in panel_labels:
        axs[key].text(
            -0.3, 1.1, key, transform=axs[key].transAxes,
            fontsize=7, va='top', ha='left')

    # Before saving, compute the pixel coordinates of panel A
    fig.canvas.draw()
    bbox_A = axs['A'].get_window_extent(renderer=fig.canvas.get_renderer())
    # Convert to pixels (already in display coordinates)
    x_min = int(bbox_A.x0)
    y_min = int(fig.get_window_extent(renderer=fig.canvas.get_renderer()).height - bbox_A.y1)  # flip y
    x_max = int(bbox_A.x1)
    y_max = int(fig.get_window_extent(renderer=fig.canvas.get_renderer()).height - bbox_A.y0)

    panel_a_width = x_max - x_min
    panel_a_height = y_max - y_min

    plt.savefig('figures/fig1_tmp.png', dpi=600)

    # Create final figure by hand
    from PIL import Image

    # Now paste the diagram into the exact panel A location
    base_fig = Image.open('figures/fig1_tmp.png')
    panel_a_img = Image.open('graphics/century_model_diagram.png')

    # Resize to fit the panel A bounding box exactly
    my_width = int(panel_a_width * 1.3)
    my_height = int(panel_a_height * 1.3)
    panel_a_resized = panel_a_img.resize(
        (my_width, my_height),Image.Resampling.LANCZOS)

    # Paste at the calculated coordinates
    my_x_min = int(x_min - 0.2*panel_a_width)
    my_y_min = int(y_min - 0.15*panel_a_height)
    base_fig.paste(panel_a_resized, (my_x_min, my_y_min))

    # Save final figure
    base_fig.save('figures/fig1.png')
    print("Saved to figures/fig1.png")