# %%
# If currently in notebooks/ directory, go up one level.
import os
import sys

if os.path.basename(os.getcwd()) == 'notebooks':
    sys.path.append(os.getcwd())
    os.chdir('..')

# %% imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.layout_engine import ConstrainedLayoutEngine
from scipy.special import expit, logit
from scipy.stats import norm

from soil_diskin.continuum_models import LognormalDisKin

import viz

colors = viz.color_palette()

# Use the style file.
plt.style.use('notebooks/style.mpl')

# %%
# Plot figure 2 -- goal is to build something that I can edit in inkscape
# Hence the empty axes and imperfect spacing. 
if __name__ == "__main__":
    print("Plotting figure 2...")
    # A spans both rows on the left, B and C are stacked in the middle,
    # and D spans both rows on the right. D is square and A is 2/3 its
    # width; B and C are left free to absorb the remaining width.
    mosaic = 'ABD;ACD'
    layout = ConstrainedLayoutEngine(hspace=0, h_pad=0.01, wspace=0.05)
    fig, axs = plt.subplot_mosaic(mosaic, layout=layout,
                                  figsize=(4.76, 1.7), dpi=300,
                                  width_ratios=[5, 2.5, 6])

    # Panel A -- schematic of soil measurements
    ax = axs['A']
    ax.set_axis_off()

    # typical values based on fig. S2, matching the focal model in fig. 1
    mu = 1
    sigma = 2.5
    model_color = colors['dark_grey']

    # Panel B -- lognormal distribution of decay rates in the inputs.
    # Plotted against ln k, so the density is the underlying normal.
    ax = axs['B']
    ln_ks = np.linspace(mu - 3.5 * sigma, mu + 3.5 * sigma, 1000)
    density = norm.pdf(ln_ks, loc=mu, scale=sigma)
    ax.semilogx(np.exp(ln_ks), density, color=model_color, lw=1)
    # label mu and sigma as in the talk version of this figure: sigma is a
    # double-headed arrow spanning mu +/- sigma, mu is an arrow down to the
    # axis at the mean; each label sits in a small grey box.
    peak = float(norm.pdf(mu, loc=mu, scale=sigma))
    one_sd = float(norm.pdf(mu + sigma, loc=mu, scale=sigma))
    label_box = dict(boxstyle='round,pad=0.15', fc=colors['light_grey'], ec='none')
    ax.annotate('', xy=(np.exp(mu + sigma), one_sd),
                xytext=(np.exp(mu - sigma), one_sd),
                arrowprops=dict(arrowstyle='<->', lw=0.5, color='black',
                                shrinkA=0, shrinkB=0))
    ax.text(np.exp(mu), one_sd * 1.12, r'$\sigma$', ha='center', va='bottom',
            fontsize=5, bbox=label_box)
    ax.annotate(r'$\mu$', xy=(np.exp(mu), 0), xytext=(np.exp(mu), one_sd * 0.55),
                ha='center', va='center', fontsize=5, bbox=label_box,
                arrowprops=dict(arrowstyle='->', lw=0.5, color='black',
                                shrinkA=0, shrinkB=0))
    ax.set_ylim(0, peak * 1.15)
    ax.set_yticks([])
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.set_xlabel(r'$\ln (k \, / \, \mathrm{yr}^{-1})$', labelpad=1)
    ax.set_ylabel(r'$p(k)$', labelpad=2)
    ax.set_title('calibrated model')

    # Panel C -- CDF of the steady-state age distribution for the same model.
    ax = axs['C']
    ages = np.logspace(-1, 5, 1000)
    ax.semilogx(ages, LognormalDisKin(mu, sigma).cdfA(ages), color=model_color, lw=1)
    ax.set_xlim(1e-1, 1e5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.set_xlabel(r'age $\tau$ (yr)', labelpad=1)
    ax.set_ylabel(r'CDF($\tau$) = F$_{new}$', labelpad=2)
    # an independent 13C-based point measurement of the CDF at a single age,
    # a bit off the model curve, with dashed guides to both axes
    tau_obs, cdf_obs = 30, 0.2
    ax.plot([tau_obs, tau_obs, ax.get_xlim()[0]], [0, cdf_obs, cdf_obs],
            ls='--', color='grey', lw=0.5, zorder=1)
    ax.scatter([tau_obs], [cdf_obs], color=colors['dark_blue'], edgecolor='black',
               linewidth=0.3, s=8, alpha=0.8, zorder=3)
    # ax.annotate('independent\ndata', xy=(tau_obs, cdf_obs),
    #             xytext=(8e4, 0.33), ha='right', va='center', ma='center', fontsize=5,
    #             arrowprops=dict(arrowstyle='->', lw=0.5, color='black',
    #                             shrinkA=1, shrinkB=2,
    #                             connectionstyle='arc3,rad=0.3'))

    # Panel D -- illustrative predicted vs. observed F_new scatter.
    # Fictional data that anticipates the real comparison in fig. 4.
    ax = axs['D']
    rng = np.random.default_rng(123)
    # five sites roughly evenly spread over observed F_new = 0.1 - 0.8
    observed = np.linspace(0.1, 0.8, 5) + rng.normal(0, 0.02, size=5)
    # add noise on the logit scale so predictions stay within (0, 1)
    predicted = expit(logit(observed) + rng.normal(0, 0.8, size=observed.size))
    ax.scatter(observed, predicted, color=colors['dark_blue'], edgecolor='black',
               linewidth=0.3, s=8, alpha=0.8, zorder=2)
    # label two points as distinct sites. The label sits above and between
    # them; arrows leave its left and right sides horizontally, then bend
    # down onto each point ("angle, angle 90" style from the matplotlib
    # annotation demo).
    order = np.argsort(observed)
    left_i, right_i = order[0], order[-2]
    label = ax.text(float(observed[left_i] + observed[right_i]) / 2, 0.85,
                    'different\nsites', ha='center', va='center', fontsize=6)
    # (0, 0.5) and (1, 0.5) are the label's left- and right-middle, in
    # fractions of its bounding box
    for i, side in ((left_i, (0, 0.5)), (right_i, (1, 0.5))):
        ax.annotate('', xy=(float(observed[i]), float(predicted[i])),
                    xytext=side, textcoords=label,
                    arrowprops=dict(arrowstyle='->', lw=0.5, color='black',
                                    shrinkA=1, shrinkB=3,
                                    connectionstyle='angle,angleA=0,angleB=90,rad=5'))
    ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=0.75, zorder=0)
    ax.set_title(r'predict independent data')
    ax.set(xlabel=r'observed F$_{new}$', ylabel=r'predicted F$_{new}$',
           xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), aspect='equal',
           xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])

    plt.savefig('figures/fig2.svg', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig2.png', dpi=300, bbox_inches='tight')