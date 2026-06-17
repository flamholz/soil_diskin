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
from scipy import integrate
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from soil_diskin.continuum_models import LognormalDisKinFast
from soil_diskin.radiocarbon_utils import load_atm14c

import viz

colors = viz.color_palette()

# Use the style file.
plt.style.use('notebooks/style.mpl')

# %% Load the results of the mu, sigma scan.
data = np.load('results/fig2_calcs.npz')
mu_values = data['mu_values']
sigma_values = data['sigma_values']
T_matrix = data['T_matrix']
R_14C_matrix = data['R_14C_matrix']

SIGMA, MU = np.meshgrid(sigma_values, mu_values, indexing='xy')

# %% For a few target T and R_14C values, find the model
# parameters that best reproduce those values.
target_Ts = [32, 34, 36]
target_R_14Cs = [1.09, 1.04, 0.92]
calibrated_Ts = []
calibrated_R_14Cs = []
calibrated_models = []
atm = load_atm14c()

for target_T, target_R_14C in zip(target_Ts, target_R_14Cs):
    pct_diffs_T = 100 * (T_matrix - target_T) / target_T
    pct_diffs_R_14C = 100 * (R_14C_matrix - target_R_14C) / target_R_14C
    loss_mat = pct_diffs_T ** 2 + pct_diffs_R_14C ** 2
    min_idx = np.unravel_index(np.argmin(loss_mat), loss_mat.shape)

    mu_cal = mu_values[min_idx[0]]
    sigma_cal = sigma_values[min_idx[1]]

    print(f'For target T={target_T}, R_14C={target_R_14C}:')
    print(f'Calibrated parameters: mu = {mu_cal}, sigma = {sigma_cal}')
    print(f'With Turnover Time = {T_matrix[min_idx]}, R_14C = {R_14C_matrix[min_idx]}')
    print(f'Percent differences: dT/T = {pct_diffs_T[min_idx]}, dR_14C/R_14C = {pct_diffs_R_14C[min_idx]}')
    print('')

    calibrated_Ts.append(T_matrix[min_idx])
    calibrated_R_14Cs.append(R_14C_matrix[min_idx])
    calibrated_models.append(LognormalDisKinFast(mu=mu_cal, sigma=sigma_cal, atm=atm))

# %% Plot figure 2, which diagrams the calibration procedure.
mosaic = 'ABC\nDEF'
fig, axs = plt.subplot_mosaic(mosaic, layout='constrained', figsize=(7.25, 4))

# --- band constants and plotting helpers ---
BAND_LOW = 30
BAND_HIGH = 40
BAND_ALPHA = 0.35
BAND_COLOR_B = "#ac9304"
BAND_COLOR_C = "#fff4b2"

# single mask reused across panels
mask_band = (T_matrix >= BAND_LOW) & (T_matrix <= BAND_HIGH)

def plot_calibrated_markers(ax, models, colors, s=15, lw=1.5):
    for my_model, color in zip(models, colors):
        ax.scatter(my_model.sigma, my_model.mu,
                   marker='o', facecolors='none',
                   edgecolors=color, s=s, linewidths=lw, zorder=20)

# Panel B -- contour plot of turnover time in mu/sigma space.
ax = axs['B']
# Ensure the contour uses the full data range for coloring
tmin, tmax = np.nanmin(T_matrix), np.nanmax(T_matrix)
levels_T = np.linspace(tmin, tmax, 10)
cf = ax.contourf(SIGMA, MU, T_matrix, levels=levels_T, cmap='Blues', vmin=tmin, vmax=tmax)
cbar = fig.colorbar(cf, ax=ax, label='Turnover Time (years)')
# simplify colorbar ticks: integers, limited number
cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

# Highlight band with translucent fill and dashed contours
ax.contourf(SIGMA, MU, mask_band.astype(int), levels=[0.5, 1.5],
            colors=[BAND_COLOR_B], alpha=BAND_ALPHA, zorder=2)
ax.contour(SIGMA, MU, T_matrix, levels=[BAND_LOW, BAND_HIGH], colors='k',
           linestyles='--', linewidths=0.5, zorder=3)

calibrated_colors = [colors[c] for c in ['dark_green', 'purple', 'dark_blue']]
plot_calibrated_markers(ax, calibrated_models, calibrated_colors)

ax.set_xlabel(r'lognormal $\sigma$')
ax.set_ylabel(r'lognormal $\mu$')
ax.set_title('turnover time (T)')

# Panel C -- contour plot of steady-state radiocarbon ratio.
ax = axs['C']
# Use full data range for radiocarbon ratio contour as well
rmin, rmax = np.nanmin(R_14C_matrix), np.nanmax(R_14C_matrix)
levels_R = np.linspace(rmin, rmax, 10)
cf2 = ax.contourf(SIGMA, MU, R_14C_matrix, levels=levels_R, cmap='Greys', vmin=rmin, vmax=rmax)
# Overlay the same T-band highlight on Panel C (use shared mask_band)
ax.contourf(SIGMA, MU, mask_band.astype(int), levels=[0.5, 1.5],
            colors=[BAND_COLOR_C], alpha=BAND_ALPHA, zorder=2)
ax.contour(SIGMA, MU, T_matrix, levels=[BAND_LOW, BAND_HIGH], colors='k', linestyles='--', linewidths=0.5,
           zorder=3)

cbar2 = fig.colorbar(cf2, ax=ax, label='$^{14}C / ^{12}C$')
# fewer, nicely formatted ticks for radiocarbon ratio
cbar2.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plot_calibrated_markers(ax, calibrated_models, calibrated_colors)

ax.set_xlabel(r'lognormal $\sigma$')
ax.set_ylabel(r'lognormal $\mu$')
ax.set_title(r'steady-state $^{14}C / ^{12}C$ ratio')

# Panel E -- survival functions for calibrated points.
plt.sca(axs['E'])
plt.xscale('log')
ages = np.logspace(-3, np.log10(1800), 200)
for i, (my_model, color) in enumerate(zip(calibrated_models, calibrated_colors)):
    survival_fn = my_model.survival_discretized(ages)
    T_val = calibrated_Ts[i]
    R_14C_val = calibrated_R_14Cs[i]
    label = f'$T={int(T_val)}$; $^{{14}}C/^{{12}}C={R_14C_val:.2f}$'
    plt.plot(ages, survival_fn, color=color, lw=2, label=label)

plt.xlim(1e-2, 1800)
plt.xlabel(r'age $\tau$ (years)')
plt.ylabel(r'$s(\tau)$')
plt.title('calibrated survival functions')
plt.legend(loc='upper right', fontsize=5, frameon=False).set_visible(False)

# Panel F -- CDF of age distributions for calibrated points.
plt.sca(axs['F'])
plt.xscale('log')
# precompute ages grid once
ages_cdf = np.logspace(-3, np.log10(100000), 200)
for my_model, color in zip(calibrated_models, calibrated_colors):
    # Vectorized survival -> PDF -> normalized CDF using trapezoidal quadrature
    s_tau = my_model.survival_discretized(ages_cdf)
    pA = s_tau / my_model.T
    pA_integral = integrate.trapezoid(pA, ages_cdf)
    pA_normalized = pA / pA_integral
    cdf_A = integrate.cumulative_trapezoid(pA_normalized, ages_cdf, initial=0)
    plt.plot(ages_cdf, cdf_A, color=color, lw=2)

obs_age, obs_CDF = 100, 0.67
plt.scatter([obs_age], [obs_CDF], color='none',
            edgecolor='k', marker='o', s=15, lw=1,
            zorder=10)
plt.axvline(x=obs_age, ymin=0, ymax=obs_CDF,
            color='grey', linestyle='--', lw=1, zorder=-1)
ax = axs['F']
ax.hlines(y=obs_CDF, xmin=ax.get_xlim()[0], xmax=obs_age,
          colors='grey', linestyles='--', linewidth=1, zorder=-1)

plt.xlim(1e-1, 100000)
plt.xlabel(r'age $\tau$ (years)')
plt.ylabel(r'CDF of $p_A(\tau)$')
plt.title('calibrated age distributions')

# Turn off axes for panels A and D to reserve illustration space.
axs['A'].axis('off')
axs['D'].axis('off')

plt.savefig('figures/fig2.png', dpi=300, bbox_inches='tight')
