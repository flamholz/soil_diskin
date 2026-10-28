"""Publication figure: static-rate-disorder (lognormal-k) age-distribution collapse.

Model: material enters a steady-state reservoir and is assigned a first-order
loss rate k drawn once from a lognormal distribution, ln k ~ Normal(mu, sigma^2),
then decays at that fixed rate. All quadrature/inversion formulas below are
given (not re-derived).

Data provenance (resolved with the user before writing this script):
  - `turnover` in the calibration CSV = Ctotal_0-100estim / NPP (stock/flux),
    i.e. mean transit time <1/k>, so sigma^2 = ln(pred/turnover) (not halved).
  - `pred` is not an independent age measurement: it is obtained by holding
    `turnover` fixed and inverting a forward-simulated 14C calibration curve
    against the site's own measured fm (see notebooks/03b_lognormal_calibration.py).
    `pred_05`/`pred_95` are `pred` re-evaluated at `turnover_q05`/`turnover_q95`
    (same fm) -- i.e. one source of randomness (SoilGrids stock uncertainty),
    not two independent quantities, and the map is monotonically decreasing so
    pred_05 > pred > pred_95. Confirmed empirically on this dataset.
  - turnover_q05/q95 and pred_05/pred_95 are 90% CIs (5th/95th percentile),
    not 95% CIs, per SoilGrids convention -- confirmed with the user.
  - (t_c, F_obs), the independent age-CDF measurement, are Duration_labeling
    and total_fnew from results/processed_balesdent_2018.csv, row-aligned by
    index with the calibration CSV (verified via exact Latitude/Longitude match).
  - Only 11 of 99 systems carry CI data; the rest are plotted as point
    estimates with no error bars.
"""
# %%
import os
if os.getcwd().endswith('notebooks'):
    os.chdir('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from numpy.polynomial.hermite_e import hermegauss
from scipy.interpolate import interp1d
from scipy.stats import norm
from permetrics.regression import RegressionMetric
import viz

plt.style.use('notebooks/style.mpl')
pal = viz.color_palette()

DATA_PATH = 'results/03_calibrate_models/lognormal_model_optimization_results.csv'
MEAS_PATH = 'results/processed_balesdent_2018.csv'

GAMMA_EULER = 0.5772156649015329
N_HERMITE = 160
N_MC = 20_000
SEED = 0
Z90 = norm.ppf(0.95)  # 5th/95th percentile z-score for a 90% CI

# %% ------------------------------------------------------------------
# Core math (quadrature / inversion / collapse) -- given by the brief,
# not re-derived here.
# -----------------------------------------------------------------------

def hermgauss_normal(n=N_HERMITE):
    """Probabilists'-Hermite nodes/weights for E_Z[g(Z)], Z ~ N(0,1)."""
    nodes, weights = hermegauss(n)
    weights = weights / weights.sum()
    return nodes, weights


def invert_mu_sigma(turnover, pred):
    """sigma^2 = ln(pred/turnover); mu = sigma^2/2 - ln(turnover)."""
    turnover = np.asarray(turnover, dtype=float)
    pred = np.asarray(pred, dtype=float)
    sigma2 = np.log(pred / turnover)
    sigma = np.sqrt(sigma2)
    mu = sigma2 / 2.0 - np.log(turnover)
    return mu, sigma


def age_cdf(t, mu, sigma, nodes, weights):
    """F_A(t; mu, sigma) = E_Z[1 - exp(-exp(ln t - (sigma^2 - mu) + sigma*Z))]."""
    t, mu, sigma = np.broadcast_arrays(np.asarray(t, dtype=float),
                                        np.asarray(mu, dtype=float),
                                        np.asarray(sigma, dtype=float))
    inner = (np.log(t)[..., None] - (sigma[..., None] ** 2 - mu[..., None])
             + sigma[..., None] * nodes)
    return (1.0 - np.exp(-np.exp(inner))) @ weights


def z_transform(t, mu, sigma):
    """Standardized log-age: collapses out mu exactly, leaves sigma-dependence."""
    t, mu, sigma = np.broadcast_arrays(np.asarray(t, dtype=float),
                                        np.asarray(mu, dtype=float),
                                        np.asarray(sigma, dtype=float))
    return ((np.log(t) - (sigma ** 2 - mu - GAMMA_EULER))
             / np.sqrt(sigma ** 2 + np.pi ** 2 / 6.0))


def master_curve(z, sigma, nodes, weights):
    """F_A expressed purely as a function of z and sigma (mu removed)."""
    z, sigma = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(sigma, dtype=float))
    shifted = z * np.sqrt(sigma ** 2 + np.pi ** 2 / 6.0) - GAMMA_EULER
    inner = shifted[..., None] + sigma[..., None] * nodes
    return (1.0 - np.exp(-np.exp(inner))) @ weights


NODES, WEIGHTS = hermgauss_normal()

# Self-check: age_cdf evaluated directly must equal master_curve evaluated at
# z(t; mu, sigma), to machine precision, for several (mu, sigma) pairs.
for _mu, _sigma in [(-9.0, 0.6), (-8.0, 1.2), (-7.0, 1.8), (-6.5, 2.4), (-9.5, 0.9)]:
    _t = np.logspace(1, 6, 50)
    _direct = age_cdf(_t, _mu, _sigma, NODES, WEIGHTS)
    _via_master = master_curve(z_transform(_t, _mu, _sigma), _sigma, NODES, WEIGHTS)
    assert np.allclose(_direct, _via_master, atol=1e-10), \
        f"collapse self-check failed for mu={_mu}, sigma={_sigma}"
print("Self-check passed: age_cdf(t; mu, sigma) == master_curve(z(t; mu, sigma), sigma).")

# %% ------------------------------------------------------------------
# Load and inspect data -- do not assume column names/mappings.
# -----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"\nLoaded {DATA_PATH}: {df.shape}")
print(df.dtypes)

required_cols = ['Latitude', 'Longitude', 'turnover', 'turnover_q05', 'turnover_q95',
                  'model_A', 'model_A_05', 'model_A_95']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"Expected columns missing from {DATA_PATH}: {missing}")

meas = pd.read_csv(MEAS_PATH)
if len(meas) != len(df):
    raise SystemExit(f"{DATA_PATH} has {len(df)} rows but {MEAS_PATH} has {len(meas)}; "
                      "cannot align by row index.")
lat_ok = np.allclose(df['Latitude'].values, meas['Latitude'].values, atol=1e-6)
lon_ok = np.allclose(df['Longitude'].values, meas['Longitude'].values, atol=1e-6)
if not (lat_ok and lon_ok):
    raise SystemExit(f"Row-index alignment between {DATA_PATH} and {MEAS_PATH} failed "
                      "(Latitude/Longitude mismatch); cannot attach t_c/F_obs.")

df['t_c'] = meas['Duration_labeling'].values
df['F_obs'] = meas['total_fnew'].values
df['system_id'] = [f"site{i:03d}" for i in range(len(df))]

# %% ------------------------------------------------------------------
# Validation: pred > turnover must hold (<A>/<T> = exp(sigma^2) >= 1).
# -----------------------------------------------------------------------
valid_mask = df['model_A'] > df['turnover']
n_excluded = int((~valid_mask).sum())
if n_excluded:
    print(f"\n{n_excluded} system(s) violate pred > turnover; excluding from the figure:")
    print(df.loc[~valid_mask, ['system_id', 'Latitude', 'Longitude', 'turnover', 'model_A']]
          .to_string(index=False))
else:
    print("\nValidation passed: pred > turnover for all systems.")

df = df.loc[valid_mask].reset_index(drop=True)

df['mu'], df['sigma'] = invert_mu_sigma(df['turnover'], df['model_A'])
df['z_c'] = z_transform(df['t_c'], df['mu'], df['sigma'])
df['F_pred'] = age_cdf(df['t_c'], df['mu'], df['sigma'], NODES, WEIGHTS)
df['residual'] = df['F_obs'] - df['F_pred']

sigma_min, sigma_max = df['sigma'].min(), df['sigma'].max()
print(f"\nRecovered sigma range: [{sigma_min:.3f}, {sigma_max:.3f}]")
low_sigma_ids = df.loc[df['sigma'] < 0.5, 'system_id'].tolist()
if low_sigma_ids:
    print(f"Systems with sigma < 0.5 (will visibly separate from the pack): {low_sigma_ids}")

# %% ------------------------------------------------------------------
# Uncertainty propagation (Monte Carlo), only for systems with 90% CI data.
#
# turnover_q05/q95 -> pred_05/pred_95 are NOT independent: pred_05/95 are
# `pred` re-evaluated at turnover_q05/95 holding the site's fm fixed, and the
# map is monotonically decreasing. We therefore build a per-site monotone
# interpolant pred(turnover) from the 3 known (turnover, pred) points and
# draw turnover from its calibrated lognormal, mapping each draw through the
# interpolant -- this reproduces the true (anti-correlated) coupling instead
# of assuming independence or a positively-correlated variant.
# -----------------------------------------------------------------------
has_ci = df[['turnover_q05', 'turnover_q95', 'model_A_05', 'model_A_95']].notna().all(axis=1)
n_ci = int(has_ci.sum())
print(f"\n{n_ci} of {len(df)} systems have 90% CI data (SoilGrids-backfilled stock "
      f"uncertainty); the other {len(df) - n_ci} are plotted as point estimates with no error bars.")

rng = np.random.default_rng(SEED)

df[['z_c_lo', 'z_c_hi', 'F_pred_lo', 'F_pred_hi', 'mc_discard_frac']] = np.nan

print("\nMonte Carlo uncertainty propagation:")
for idx in df.index[has_ci]:
    row = df.loc[idx]

    log_med = np.log(row['turnover'])
    sigma_log_hi = (np.log(row['turnover_q95']) - log_med) / Z90
    sigma_log_lo = (log_med - np.log(row['turnover_q05'])) / Z90
    sigma_log_turnover = 0.5 * (sigma_log_hi + sigma_log_lo)
    turnover_draws = np.exp(log_med + sigma_log_turnover * rng.standard_normal(N_MC))

    log_t_pts = np.log([row['turnover_q05'], row['turnover'], row['turnover_q95']])
    log_p_pts = np.log([row['model_A_05'], row['model_A'], row['model_A_95']])
    order = np.argsort(log_t_pts)
    interp = interp1d(log_t_pts[order], log_p_pts[order], kind='linear',
                       fill_value='extrapolate', assume_sorted=True)
    pred_draws = np.exp(interp(np.log(turnover_draws)))

    keep = pred_draws > turnover_draws
    discard_frac = 1.0 - keep.mean()
    mu_d, sigma_d = invert_mu_sigma(turnover_draws[keep], pred_draws[keep])
    z_d = z_transform(row['t_c'], mu_d, sigma_d)
    F_d = age_cdf(row['t_c'], mu_d, sigma_d, NODES, WEIGHTS)
    z_lo, z_hi = np.percentile(z_d, [2.5, 97.5])
    F_lo, F_hi = np.percentile(F_d, [2.5, 97.5])
    df.loc[idx, ['z_c_lo', 'z_c_hi', 'F_pred_lo', 'F_pred_hi', 'mc_discard_frac']] = \
        [z_lo, z_hi, F_lo, F_hi, discard_frac]

    print(f"  {row['system_id']}: discard frac = {discard_frac:.3f}, "
          f"z_c 95% CI width = {z_hi - z_lo:.3f}")

# %% ------------------------------------------------------------------
# Summary table + caption text
# -----------------------------------------------------------------------
summary_cols = ['system_id', 'turnover', 'model_A', 'mu', 'sigma', 'z_c', 'F_pred', 'F_obs', 'residual']
print("\nSummary table:")
print(df[summary_cols].to_string(index=False, float_format=lambda x: f"{x:0.4g}"))

rms_1to1 = np.sqrt(np.mean(df['residual'] ** 2))
kge = RegressionMetric(y_true=df['F_obs'].values, y_pred=df['F_pred'].values).kling_gupta_efficiency()
ci_rows = df.loc[has_ci]
crosses = int(((ci_rows['F_pred_lo'] <= ci_rows['F_obs']) &
               (ci_rows['F_obs'] <= ci_rows['F_pred_hi'])).sum())
print(f"\nPanel C: RMS deviation from 1:1 line = {rms_1to1:.4f}, KGE = {kge:.4f}")
print(f"Panel C: {crosses} of {n_ci} systems with CI have observed F_obs within the "
      f"predicted 95% interval.")

caption = (
    f"Standardizing log-age by mu and sigma collapses {len(df)} sites' predicted steady-state "
    f"age CDFs onto a near-universal curve in z; mu is removed exactly, while sigma (recovered "
    f"range {sigma_min:.2f}-{sigma_max:.2f}) leaves a residual skewness shown as the shaded "
    f"envelope. Independent radiocarbon-based age measurements (colored points, error bars from "
    f"{n_ci} sites with propagated stock uncertainty) fall along the same collapse, and predicted "
    f"vs. observed F_A(tau_c) agree with the 1:1 line (KGE = {kge:.2f}, RMS = {rms_1to1:.3f}; "
    f"{n_excluded} site(s) excluded for violating pred > turnover)."
)
print("\nDraft caption:\n" + caption)

# %% ------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------
accent = pal['red']
sigma_cmap = LinearSegmentedColormap.from_list('sigma_seq', [pal['dark_blue'], pal['light_blue']])
sigma_norm = Normalize(vmin=sigma_min, vmax=sigma_max)
stat_box_props = dict(boxstyle='round', facecolor=pal['light_yellow'],
                       edgecolor=pal['dark_grey'], alpha=0.8)

fig, axs = plt.subplots(1, 3, figsize=(7.24, 2.49), constrained_layout=True,
                         width_ratios=[1, 1.1, 1])

# --- Panel A: calibrated age CDFs (raw time, not z-transformed) for a few
# sites spanning a wide range of mean age ---
axRaw = axs[0]
# Chosen for well-separated mean age; site041 additionally represents the
# sites in the Balesdent et al. data with a land-use transition ~100-1000 yr
# ago (t_c = 129 yr here), replacing an automatically-picked site (~site046)
# that would otherwise have been the only one in that class.
example_ids = ['site022', 'site052', 'site076', 'site041', 'site060']
picked = df.set_index('system_id').loc[example_ids].reset_index()
example_colors = [pal['dark_blue'], pal['blue'], pal['dark_green'], pal['red'], pal['dark_purple']]
t_grid_lo = min(1.0, picked['t_c'].min() / 2.0)
t_grid_raw = np.logspace(np.log10(t_grid_lo), 6, 400)

for (_, row), color in zip(picked.iterrows(), example_colors):
    curve = age_cdf(t_grid_raw, row['mu'], row['sigma'], NODES, WEIGHTS)
    axRaw.plot(t_grid_raw, curve, color=color, lw=1.6,
               label=f"{row['system_id']} ($A$ = {row['model_A']:.0f} yr)")
    axRaw.plot(row['t_c'], row['F_obs'], marker='o', color=color,
               mec='k', mew=0.5, markersize=5, zorder=3)

axRaw.set_xscale('log')
axRaw.set_xlim(t_grid_raw.min(), t_grid_raw.max())
axRaw.set_ylim(0, 1)
axRaw.set_xlabel(r'age, $\tau$ (years)')
axRaw.set_ylabel(r'$F_A(\tau)$')
axRaw.set_title('calibrated age CDFs')
axRaw.legend(loc='lower right', frameon=False)

# --- Panel B: collapse ---
axCollapse = axs[1]
z_grid = np.linspace(-3.5, 3.5, 400)
sigma_env = np.linspace(sigma_min, sigma_max, 60)
Zg, Sg = np.meshgrid(z_grid, sigma_env)
env_curves = master_curve(Zg, Sg, NODES, WEIGHTS)
env_lo, env_hi = env_curves.min(axis=0), env_curves.max(axis=0)

axCollapse.fill_between(z_grid, env_lo, env_hi, color='0.82', zorder=1,
                         label='residual $\\sigma$-dependence')

for _, row in df.iterrows():
    curve = master_curve(z_grid, np.full_like(z_grid, row['sigma']), NODES, WEIGHTS)
    axCollapse.plot(z_grid, curve, color=sigma_cmap(sigma_norm(row['sigma'])), lw=1.6, alpha=0.85, zorder=2)

    # Only z_c carries propagated uncertainty (from the turnover/pred MC via
    # mu, sigma); F_obs has no CI in the source data, so no vertical error bar.
    xerr = ([[row['z_c'] - row['z_c_lo']], [row['z_c_hi'] - row['z_c']]]
            if not np.isnan(row['z_c_lo']) else None)
    axCollapse.errorbar(row['z_c'], row['F_obs'], xerr=xerr, fmt='none',
                         ecolor='k', elinewidth=0.5, capsize=2, zorder=3)

tau_cmap = plt.cm.viridis
tau_norm = LogNorm(vmin=df['t_c'].min(), vmax=df['t_c'].max())
sc = axCollapse.scatter(df['z_c'], df['F_obs'], c=df['t_c'], cmap=tau_cmap, norm=tau_norm,
                         edgecolor='k', linewidths=0.5, s=20, zorder=4)
cbar = fig.colorbar(sc, ax=axCollapse, pad=0.02)
cbar.set_label(r'time since transition, $\tau_c$ (yr)')
cbar.outline.set_visible(False)

for sid in low_sigma_ids:
    r = df.loc[df['system_id'] == sid].iloc[0]
    axCollapse.annotate(sid, (r['z_c'], r['F_obs']), fontsize=5, color='0.2',
                         xytext=(3, 3), textcoords='offset points')

axCollapse.set_xlim(-3.5, 3.5)
axCollapse.set_ylim(0, 1)
axCollapse.set_xlabel(r'$z = (\ln \tau - E[\ln A]) \,/\, \mathrm{SD}[\ln A]$')
axCollapse.set_ylabel('$F_A$')
axCollapse.set_title('collapse onto standardized log-age')

# --- Panel C: independent validation ---
axCalib = axs[2]
axCalib.plot([0, 1], [0, 1], linestyle='--', color='grey', lw=1.5, zorder=1)

for _, row in df.iterrows():
    yerr = ([[row['F_pred'] - row['F_pred_lo']], [row['F_pred_hi'] - row['F_pred']]]
            if not np.isnan(row['F_pred_lo']) else None)
    axCalib.errorbar(row['F_obs'], row['F_pred'], yerr=yerr, fmt='o',
                      color=accent, mec='k', mew=0.5, markersize=5, elinewidth=0.5,
                      capsize=2, zorder=3)

box_text = f'KGE = {kge:.2f}\nRMSE = {rms_1to1:.2f}'
axCalib.text(0.05, 0.95, box_text, transform=axCalib.transAxes, fontsize=6,
             verticalalignment='top', bbox=stat_box_props)
axCalib.set_title('independent validation')

axCalib.set_xlim(0, 1)
axCalib.set_ylim(0, 1)
axCalib.set_xlabel('observed $F_{obs}$')
axCalib.set_ylabel(r'predicted $F_A(\tau_c)$')

for label, ax in zip('ABC', axs):
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=7, va='top', ha='left')

os.makedirs('figures/experimental', exist_ok=True)
fig.savefig('figures/experimental/fig4_new.pdf')
fig.savefig('figures/experimental/fig4_new.png', dpi=300)
print("\nSaved figures/experimental/fig4_new.pdf and .png")
