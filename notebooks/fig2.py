# %%
import numpy as np
import matplotlib.pyplot as plt

from soil_diskin.models import PowerLawDisKin

import viz
colors = viz.color_palette()

# use the style file
plt.style.use('notebooks/style.mpl')

# %% Load the results of the t_min, t_max scan
data = np.load('results/fig2_calcs.npz')
t_min_values = data['t_min_values']
t_max_values = data['t_max_values']
T_matrix = data['T_matrix']
R_14C_matrix = data['R_14C_matrix']

t_max_grid, t_min_grid = np.meshgrid(
    t_max_values, t_min_values, indexing='ij')

# %% For a few target T and R_14C values, find the model
# parameters that best reproduce those values.

target_Ts = [1, 3, 10, 50]
target_R_14Cs = [1.1, 1., 0.88, 0.78]
calibrated_Ts = []
calibrated_R_14Cs = []
calibrated_models = []

for target_T, target_R_14C in zip(target_Ts, target_R_14Cs):
    # find positions where T = target_T and R_14C = target_R_14C
    T_positions = np.argwhere(np.isclose(T_matrix, target_T, rtol=0.1))
    R_14C_positions = np.argwhere(np.isclose(R_14C_matrix, target_R_14C, rtol=0.1))

    pct_diffs_T = 100 * (T_matrix - target_T) / target_T
    pct_diffs_R_14C = 100 * (R_14C_matrix - target_R_14C) / target_R_14C
    loss_mat = (pct_diffs_T**2 + pct_diffs_R_14C**2)
    min_idx = np.unravel_index(np.argmin(loss_mat), loss_mat.shape)

    print(f'For target T={target_T}, R_14C={target_R_14C}:')
    print(f'Calibrated parameters: t_min = {t_min_values[min_idx[0]]}, t_max = {t_max_values[min_idx[1]]}')
    print(f'With Turnover Time = {T_matrix[min_idx]}, R_14C = {R_14C_matrix[min_idx]}')
    print(f'Percent differences: dT/T = {pct_diffs_T[min_idx]}, dR_14C/R_14C = {pct_diffs_R_14C[min_idx]}')
    print('')

    calibrated_Ts.append(T_matrix[min_idx])
    calibrated_R_14Cs.append(R_14C_matrix[min_idx])
    calibrated_model = PowerLawDisKin(
        t_min=t_min_values[min_idx[0]],
        t_max=t_max_values[min_idx[1]])
    calibrated_models.append(calibrated_model)

# %% Plot figure 2, which diagrams the calibration procedure
mosaic = 'ABC\nDEF'
fig, axs = plt.subplot_mosaic(
    mosaic, layout='constrained', figsize=(7.25, 4))

# Panel A -- contour plot of turnover time with log-scaled axes
plt.sca(axs['B'])
plt.contourf(t_max_grid, t_min_grid, T_matrix,
             levels=10, cmap='Blues')
plt.colorbar(label='Turnover Time (years)')

# Mark the calibrated points
calibrated_colors = [colors[c] for c in ['dark_green', 'purple', 'red', 'dark_blue']]
for my_model, color in zip(calibrated_models, calibrated_colors):
    plt.scatter(my_model.t_max, my_model.t_min,
                marker='o', color='none',
                edgecolor=color, s=15, lw=1.5)

plt.xlabel('$t_{max}$ (years)')
plt.ylabel('$t_{min}$ (years)')
plt.title('turnover time (T)')

# Panel B -- contour plot of steady-state radiocarbon ratio
plt.sca(axs['C'])
plt.contourf(t_max_grid, t_min_grid, R_14C_matrix,
             levels=10, cmap='Greys')
plt.colorbar(label='$^{14}C / ^{12}C$')

# Mark the calibrated points
for my_model, color in zip(calibrated_models, calibrated_colors):
    plt.scatter(my_model.t_max, my_model.t_min,
                marker='o', color='none',
                edgecolor=color, s=15, lw=1.5)
plt.xlabel('$t_{max}$ (years)')
plt.ylabel('$t_{min}$ (years)')
plt.title(r'steady-state $^{14}C / ^{12}C$ ratio')

# Plot the survival functions for the calibrated points
plt.sca(axs['E'])
plt.xscale('log')
ages = np.logspace(-3, np.log10(1800), 200)
for i, (my_model, color) in enumerate(
    zip(calibrated_models, calibrated_colors)):
    survival_fn = my_model.s(ages)
    T_val = calibrated_Ts[i]
    R_14C_val = calibrated_R_14Cs[i]
    label=f'$T={int(T_val)}$; $^{{14}}C/^{{12}}C={R_14C_val:.2f}$'
    plt.plot(ages, survival_fn, color=color, lw=2, label=label)
plt.xlim(1e-2, 1800)
plt.xlabel(r'age $\tau$ (years)')
plt.ylabel(r'$s(\tau)$')
plt.title('calibrated survival functions')
# hiding the legend, will make in inkscape
plt.legend(loc='upper right', fontsize=5,
           frameon=False).set_visible(False)

plt.sca(axs['F'])
plt.xscale('log')
# plot the CDF of the age distributions for the calibrated points
for idx, (my_model, color) in enumerate(zip(calibrated_models, calibrated_colors)):
    # ages = np.arange(2000)
    ages = np.logspace(-3, np.log10(100000), 200)
    age_dist_cdf = my_model.cdfA(ages)
    plt.plot(ages, age_dist_cdf, color=color, lw=2)

# Diagram the process of model validation from a single point estimate
# of the CDF at age = 100 for example. 
obs_age, obs_CDF = 10, 0.21
plt.scatter([obs_age], [obs_CDF], color='none', 
            edgecolor='k', marker='o', s=15, lw=1,
            zorder=10)
# draw a dashed line to the x-axis from this point
plt.axvline(x=obs_age, ymin=0, ymax=obs_CDF,
            color='grey', linestyle='--', lw=1, zorder=-1)
ax = axs['F']
ax.hlines(y=obs_CDF, xmin=ax.get_xlim()[0], xmax=obs_age,
          colors='grey', linestyles='--', linewidth=1, zorder=-1)

plt.xlim(1e-1, 100000)
plt.xlabel(r'age $\tau$ (years)')
plt.ylabel(r'CDF of $p_A(\tau)$')
plt.title('calibrated age distributions')

# Turn off axes for panels A and D
axs['A'].axis('off')
axs['D'].axis('off')

plt.savefig('figures/fig2.png', dpi=300, bbox_inches='tight')
