# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import viz

# %%
plt.style.use('notebooks/style.mpl')
pal = viz.color_palette()

powerlaw_params = pd.read_csv('results/03_calibrate_models/powerlaw_model_optimization_results.csv')
gen_powerlaw_params = pd.read_csv('results/03_calibrate_models/general_powerlaw_model_optimization_results.csv')
gen_powerlaw_params_beta_half = pd.read_csv('results/03_calibrate_models/general_powerlaw_model_optimization_results_beta_half.csv')
lognormal_params = pd.read_csv('results/03_calibrate_models/03b_lognormal_predictions_calcurve_python.csv')

# %%
# Each model gets a color matching fig4
model_colors = {
    'lognormal':                   pal['dark_blue'],
    'power law ($\\alpha=1$)':     pal['blue'],
    'power law ($\\alpha=e^{-\\gamma}$)':    pal['light_blue'],
    'power law ($\\alpha=e^{-\\gamma}/2$)':  pal['dark_grey'],
}

fig, axs = plt.subplots(1, 4, figsize=(7.24, 2.5), dpi=300, constrained_layout=True)

kde_kws = dict(fill=True, alpha=0.5, linewidth=1.5, log_scale=True)

# Panel A: lognormal fitted median age
ax = axs[0]
sns.kdeplot(lognormal_params['pred'], ax=ax, color=pal['dark_blue'],
            label='lognormal', **kde_kws)
ax.set_xlabel('fitted median age (yr)')
ax.set_title('lognormal $\\tilde{a}$')

# Panel B: t_min across all three power law variants
ax = axs[1]
for label, df, color in [
    ('power law ($\\alpha=1$)',              powerlaw_params,            pal['blue']),
    ('power law ($\\alpha=e^{-\\gamma}$)',   gen_powerlaw_params,        pal['light_blue']),
    ('power law ($\\alpha=e^{-\\gamma}/2$)', gen_powerlaw_params_beta_half, pal['dark_grey']),
]:
    sns.kdeplot(df['t_min'], ax=ax, color=color, label=label, **kde_kws)
ax.set_xlabel('$t_{min}$ (yr)')
ax.set_title('$t_{min}$')

# Panel C: t_max across all three power law variants
ax = axs[2]
for label, df, color in [
    ('power law ($\\alpha=1$)',              powerlaw_params,            pal['blue']),
    ('power law ($\\alpha=e^{-\\gamma}$)',   gen_powerlaw_params,        pal['light_blue']),
    ('power law ($\\alpha=e^{-\\gamma}/2$)', gen_powerlaw_params_beta_half, pal['dark_grey']),
]:
    sns.kdeplot(df['t_max'], ax=ax, color=color, label=label, **kde_kws)
ax.set_xlabel('$t_{max}$ (yr)')
ax.set_title('$t_{max}$')

# Panel D: fitted turnover time across all four models
ax = axs[3]
sns.kdeplot(lognormal_params['turnover'], ax=ax, color=pal['dark_blue'],
            label='lognormal', **kde_kws)
for label, df, color, tcol in [
    ('power law ($\\alpha=1$)',              powerlaw_params,               pal['blue'],       'modeled_tau'),
    ('power law ($\\alpha=e^{-\\gamma}$)',   gen_powerlaw_params,           pal['light_blue'], 'modeled_T'),
    ('power law ($\\alpha=e^{-\\gamma}/2$)', gen_powerlaw_params_beta_half, pal['dark_grey'],  'modeled_T'),
]:
    sns.kdeplot(df[tcol], ax=ax, color=color, label=label, **kde_kws)
ax.set_xlabel('turnover time (yr)')
ax.set_title('fitted turnover time')

# Shared formatting
for ax in axs:
    ax.set_ylabel('density')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Single legend on the last panel
axs[3].legend(fontsize=5, loc='upper right')

# Subplot labels
for ax, label in zip(axs, 'ABCD'):
    ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontsize=7, va='top', ha='left')

fig.savefig('figures/figS2.png', dpi=600, bbox_inches='tight')
