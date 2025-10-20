# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

# %%
# Load libraries
from notebooks.models import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from permetrics.regression import RegressionMetric

# %%
# Read the raw data and calculate the ratio in C stocks between the reference and the experiment sites.
raw_site_data = pd.read_excel('data/balesdent_2018/balesdent_2018_raw.xlsx', skiprows=7).query('MAT_C>17.0 & PANN_mm>1000.0 & `P to PET ratio` >0.8')
site_data = pd.read_csv('results/processed_balesdent_2018.csv')
J_ratio = raw_site_data['Cref_0-100estim'] / raw_site_data['Ctotal_0-100estim']


# %% [markdown]
# ### Power-law

# %% [markdown]
# For the unlabeled system, we have the amount of carbon that is of a specific age $J * p_A(t) * T$ - so that the integral of all of the carbon in the system is $J * T$:
# 
# $$ J*p_A(t)*T = J\frac{1}{E_1(\frac{a}{\tau})} \frac{e^{-(a+t)/\tau}}{a+t} ae^{a/\tau}E_1(a/\tau) = J \frac{ae^{-t/\tau}}{a+t} = Js(t)$$
# 
# where T is:
# $$ T = ae^{a/\tau}E_1(a/\tau) $$
# 
# At time $t'$=0, we stop the inputs to the system, the amount of carbon in the unlabeled system is all of the carbon that is older than $t'$:
# 
# $$ C_{unlabeled}(t') = \int_{t'}^{\infty} Js(t'') dt'' = 
# J\int_{t'}^{\infty} \frac{ae^{-t''/\tau}}{a+t''} dt'' = 
# J \left.-ae^{a/\tau} E_1(\frac{a + t'}{\tau})\right|^{\infty}_{t'}  = J ae^{a/\tau} E_1(\frac{a + t'}{\tau})$$
# 
# and the amount of carbon in the labeled system at time $t'$ is the same, but only for tages 0 to $t'$:
# 
# $$ C_{labeled}(t') = \int_{0}^{t'} Js(t'') dt'' = 
# J\int_{0}^{t'} \frac{ae^{-t''/\tau}}{a+t''} dt'' = 
# J \left.-ae^{a/\tau} E_1(\frac{a + t'}{\tau})\right|^{0}_{t'}  = J ae^{a/\tau}[E_1(\frac{a}{\tau}) - E_1(\frac{a + t'}{\tau})]$$
# 
# Assuming we had inputs $J_1$ and parameters $a_1$ and $\tau_1$ for the unlabeled system and $J_2$ and parameters $a_2$ and $\tau_2$ for the labeled system, then fraction of labeled carbon in the system at time $t'$ is:
# 
# $$ f(t') = \frac{J_2 \times C_{labeled}(t')}{J_1 \times C_{unlabeled}(t') + J_2 \times C_{labeled}(t')} = \frac{J_2 a_2e^{a_2/\tau_2} E_1(\frac{a_2 + t'}{\tau_2})}{J_1 a_1e^{a_1/\tau_1}[E_1(\frac{a_1}{\tau_1}) - E_1(\frac{a_1 + t'}{\tau_1})] + J_2 a_2e^{a_2/\tau_2} E_1(\frac{a_2 + t'}{\tau_2})}$$

# %% [markdown]
# ### Gamma model
# The definitions are the same, the integral for s(t) is:
# 
# $$ C_{unlabeled}(t') = \int_{t'}^{\infty} Js(t'') dt'' = 
# J\int_{t'}^{\infty} (1 + \theta t'') ^{-\alpha} dt'' = \left. J \frac{(1+\theta t'')^{1-\alpha}}{(\alpha-1)\theta} \right|^{\infty}_{t'}  = J \frac{(1+\theta t')^{1-\alpha}}{(\alpha-1)\theta} $$
# 
# and 
# $$ C_{labeled}(t') = \int_{0}^{t'} Js(t'') dt'' = 
# J\int_{0}^{t'} (1 + \theta t'') ^{-\alpha} dt'' = \left. J \frac{(1+\theta t'')^{1-\alpha}}{(\alpha-1)\theta} \right|^{t'}_{0}  = J \frac{1 - (1+\theta t')^{1-\alpha}}{(\alpha-1)\theta} $$
# 
# so the fraction of labeled carbon in the system at time $t'$ is:
# 
# $$ f(t') = \frac{J_2 \times C_{labeled}(t')}{J_1 \times C_{unlabeled}(t') + J_2 \times C_{labeled}(t')} = \frac{J_2 \frac{1 - (1+\theta_2 t')^{1-\alpha_2}}{(\alpha_2-1)\theta_2}}{J_1 \frac{(1+\theta_1 t')^{1-\alpha_1}}{(\alpha_1-1)\theta_1} + J_2 \frac{1 - (1+\theta_2 t')^{1-\alpha_2}}{(\alpha_2-1)\theta_2}}$$

# %%
# load lognormal data
ln_input_data = pd.read_csv('results/06_sensitivity_analysis/lognormal_input_data.csv')
ln_tau_data = pd.read_csv('results/06_sensitivity_analysis/lognormal_tau_data.csv')
ln_age_data = pd.read_csv('results/06_sensitivity_analysis/lognormal_age_data.csv')

ts_ln = ln_input_data['time'].values
ln_input_data = [ln_input_data.values[:,i] for i in range(ln_input_data.shape[1]-1)]
ln_tau_data = [ln_tau_data.values[:,i] for i in range(ln_tau_data.shape[1]-1)]
ln_age_data = [ln_age_data.values[:,i] for i in range(ln_age_data.shape[1]-1)]

# %%
# Run sensitivity analysis for power law and gamma models
powerlaw_params = pd.read_csv('results/03_calibrate_models/powerlaw_model_optimization_results.csv')
gamma_params = pd.read_csv('results/03_calibrate_models/gamma_model_optimization_results.csv')

a1 = powerlaw_params['tau_0'].mean()
tau1 = powerlaw_params['tau_inf'].mean()

gamma_alpha = gamma_params['a'].mean()
gamma_theta = gamma_params['b'].mean()

ratios = np.percentile(J_ratio.dropna().values,[2.5, 25, 50, 75, 97.5])
ts = np.logspace(-1, 6, 1000)  # time in years

green_colors = plt.cm.Greens(np.linspace(0.2, 0.8, 5))
red_colors = plt.cm.Reds(np.linspace(0.2, 0.8, 5))
blue_colors = plt.cm.Blues(np.linspace(0.2, 0.8, 5))

# for the power law model
int_s_labeled  = lambda t, a, tau:  a * np.exp(a / tau) * (exp1(a / tau) - exp1((a + t) / tau)  )
int_s_unlabeled = lambda t, a, tau: a * np.exp(a / tau) *  exp1((a + t) / tau) 
f_new = lambda t, J1, J2, a1, a2, tau1, tau2: J2 * int_s_labeled(t, a2, tau2) / (J2 * int_s_labeled(t, a2, tau2) + J1 * int_s_unlabeled(t, a1, tau1))

pl_input_data = [f_new(ts, 1, 1 * j, a1, a1, tau1, tau1) for j in ratios]
pl_a_data = [f_new(ts, 1, 1 , a1, a1 * j, tau1, tau1) for j in ratios]
pl_tau_data = [f_new(ts, 1, 1 , a1, a1, tau1, tau1 * j) for j in ratios]

# for the gamma model
g_int_s_labeled = lambda t, theta, alpha: (1-(1 + theta * t) ** (1 - alpha)) / ((alpha - 1) * theta)
g_int_s_unlabeled = lambda t, theta, alpha: (1 + theta * t) ** (1 - alpha) / ((alpha - 1) * theta)
g_f_new = lambda t, J1, J2, alpha1, alpha2, theta1, theta2: J2 * g_int_s_labeled(t, theta2, alpha2) / (J2 * g_int_s_labeled(t, theta2, alpha2) + J1 * g_int_s_unlabeled(t, theta1, alpha1))

gamma_input_data = [g_f_new(ts, 1, 1 * j, gamma_alpha, gamma_alpha, gamma_theta, gamma_theta) for j in ratios]
gamma_a_data = [g_f_new(ts, 1, 1 , gamma_alpha, gamma_alpha * j, gamma_theta, gamma_theta) for j in ratios]
gamma_tau_data = [g_f_new(ts, 1, 1 , gamma_alpha, gamma_alpha, gamma_theta, gamma_theta * j) for j in ratios]


# %%
# Plot results
def plot_data(ax, ts, data, colors, title_label, legend_label):
    for j, d in enumerate(data):
        ax.semilogx(ts, d, color=colors[j], label=f'{legend_label} = {ratios[j]:.2f}')
        
        ax.legend(loc='lower right')
        sns.regplot(data=site_data, x="Duration_labeling", y="total_fnew",ax=ax,scatter_kws={'color':'k'},line_kws={'color':'k','lw':0},x_bins=[3,10,30,50,100,300,1000,3000],fit_reg=False,ci=95)
        ax.set_title(title_label)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Fraction of labeled carbon')

fig, axs = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True, constrained_layout=True, dpi=300)    
plot_data(axs[0, 0], ts, pl_input_data, green_colors, 'Change in input', '$J_{post}$/$J_{pre}$')
plot_data(axs[0, 1], ts, pl_a_data, red_colors, 'Change in $t_{min}$', '$t_{min,post}$/$t_{min,pre}$')
plot_data(axs[0, 2], ts, pl_tau_data, blue_colors, 'Change in $t_{max}$', '$t_{max,post}$/$t_{max,pre}$')

plot_data(axs[1, 0], ts_ln, ln_input_data, green_colors, 'Change in input', '$J_{post}$/$J_{pre}$')
plot_data(axs[1, 1], ts_ln, ln_input_data, red_colors, 'Change in turnover $T$', '$T_{post}$/$T_{pre}$')
plot_data(axs[1, 2], ts_ln, ln_input_data, blue_colors, 'Change in mean age $A$', '$A_{post}$/$A_{pre}$')

plot_data(axs[2, 0], ts, gamma_input_data, green_colors, 'Change in input', '$J_{post}$/$J_{pre}$')
plot_data(axs[2, 1], ts, gamma_a_data, red_colors, 'Change in $\\alpha$', '$\\alpha_{post}$/$\\alpha_{pre}$')
plot_data(axs[2, 2], ts, gamma_tau_data, blue_colors, 'Change in $\\theta$', '$\\theta_{post}$/$\\theta_{pre}$')
axs[0,0].text(-0.3, 0.5, 'Power-law model', transform=axs[0,0].transAxes, ha='center', va='center', fontsize=16, alpha=1, rotation=90)
axs[1,0].text(-0.3, 0.5, 'Lognormal model', transform=axs[1,0].transAxes, ha='center', va='center', fontsize=16, alpha=1, rotation=90)
axs[2,0].text(-0.3, 0.5, 'Gamma model', transform=axs[2,0].transAxes, ha='center', va='center', fontsize=16, alpha=1, rotation=90)
plt.tight_layout()

# Save figure
plt.savefig('figures/figS5.png', dpi=600)


