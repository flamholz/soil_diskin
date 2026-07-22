#%%
"""
fig3_with_src_models.py

Reproduces fig3.py's main panel (predicted vs observed F_new, y=x line, KGE+RMSE
box per model) over all 99 Balesdent sites, **with the three
SoilBiogeochemModels.jl process models added as panels** (I/J/K): CASA-CNP,
MIMICS, CORPSE.

Predictions in results/04_model_predictions/{CASACNP,MIMICS,CORPSE}.csv are
written by SoilBiogeochemModels.jl/experiments/align_fnew_to_balesdent.jl, one
value per row of processed_balesdent_2018.csv.

Run from the notebooks/ directory so the local `viz` module imports:
    cd notebooks && ../.venv/bin/python fig3_with_src_models.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import viz
from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error

if os.getcwd().endswith("notebooks"):
    os.chdir("..")

all_sites = pd.read_csv("results/processed_balesdent_2018.csv")
plt.style.use("notebooks/style.mpl")
pal = viz.color_palette()

P = "results/04_model_predictions"
powerlaw = pd.read_csv(f"{P}/power_law_model_predictions.csv")
lognormal = pd.read_csv(f"{P}/lognormal_model_predictions.csv")
gamma = pd.read_csv(f"{P}/gamma_model_predictions.csv")
CLM45 = pd.read_csv(f"{P}/CLM45_fnew.csv", header=None, names=["prediction"])
JSBACH = pd.read_csv(f"{P}/JSBACH_fnew.csv", header=None, names=["prediction"])
RCM = pd.read_csv(f"{P}/RCM.csv")
CASACNP = pd.read_csv(f"{P}/CASACNP.csv", header=None, names=["prediction"])
MIMICS = pd.read_csv(f"{P}/MIMICS.csv", header=None, names=["prediction"])
CORPSE = pd.read_csv(f"{P}/CORPSE.csv", header=None, names=["prediction"])

# my added process models (kept visually distinct from continuum/ESM/RCM families)
SRC_COLORS = ["#C44E52", "#DD8452", "#8C6BB1"]


def plot_model_predictions(ax, predictions, model_name, color, pred_err=None):
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="y=x", zorder=-10, lw=1)
    if pred_err is not None:
        ax.errorbar(all_sites["total_fnew"], predictions, yerr=pred_err, fmt="o",
                    color=color, ecolor="k", elinewidth=0.5, capsize=2, mec="k",
                    mew=0.5, markersize=5, alpha=0.9)
    else:
        ax.scatter(all_sites["total_fnew"], predictions, color=color,
                   edgecolor="k", lw=0.5, s=20, alpha=0.9)
    t = all_sites["total_fnew"].values
    p = predictions.values
    m = ~pd.isna(t) & ~pd.isna(p)
    kge = RegressionMetric(y_true=t[m], y_pred=p[m]).kling_gupta_efficiency()
    rmse = root_mean_squared_error(t[m], p[m])
    props = dict(boxstyle="round", facecolor=pal["light_yellow"], edgecolor=pal["dark_grey"], alpha=0.8)
    ax.text(0.05, 0.95, f"KGE = {kge:.2f}\nRMSE = {rmse:.2f}", transform=ax.transAxes,
            fontsize=6, verticalalignment="top", bbox=props)
    ax.set_title(model_name)
    ax.set_xticks(np.arange(0, 1.1, 0.5)); ax.set_yticks(np.arange(0, 1.1, 0.5))


def err_of(df):
    return df[["predicted_fnew_05", "predicted_fnew_95"]].sub(df["predicted_fnew"], axis=0).abs().fillna(0).values.T


#%% 3×4 grid: A–H = published models, I/J/K = the src process models
fig, axs = plt.subplots(3, 4, figsize=(7.24, 5.2), dpi=300,
                        constrained_layout=True, sharex=True, sharey=True)
axs = axs.flatten()

# continuum (A–C)
for ax, df, title, c in zip(axs[:3], [powerlaw, lognormal, gamma],
                            ["power law model", "lognormal model", "gamma model"],
                            [pal["dark_blue"], pal["blue"], pal["light_blue"]]):
    plot_model_predictions(ax, df["predicted_fnew"], title, c, err_of(df))

# ESMs (D–E)
for ax, df, title, c in zip(axs[3:5], [CLM45, JSBACH], ["CLM4.5", "JSBACH"],
                            [pal["dark_purple"], pal["purple"]]):
    plot_model_predictions(ax, df["prediction"], title, c)

# reduced-complexity (F–H)
for i, col in enumerate(RCM.columns):
    plot_model_predictions(axs[5 + i], RCM[col], col + r" ($^{14}C$ corrected)",
                           [pal["dark_green"], pal["green"], pal["light_green"]][i])

# src process models (I–K)
for ax, df, title, c in zip(axs[8:11], [CASACNP, MIMICS, CORPSE],
                            ["CASA-CNP", "MIMICS", "CORPSE"], SRC_COLORS):
    plot_model_predictions(ax, df["prediction"], title, c)

axs[11].set_visible(False)  # 12th cell unused

for j in (8, 9, 10, 7):     # bottom-most visible panel of each column
    axs[j].set_xlabel(r"observed F$_{new}$ ($\delta^{13}C$ based)")
for j in (0, 4, 8):
    axs[j].set_ylabel(r"predicted F$_{new}$")
for j, label in zip(range(11), "ABCDEFGHIJK"):
    axs[j].text(-0.2, 1.12, label, transform=axs[j].transAxes, fontsize=7, va="top", ha="left")

fig.savefig("figures/fig3_with_src_models.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig3_with_src_models.svg", bbox_inches="tight")
print("wrote figures/fig3_with_src_models.png")

#%% standalone 1×3 of just the src models (remake of the earlier figure, 99 sites)
fig, axs = plt.subplots(1, 3, figsize=(7.24, 2.6), dpi=300,
                        constrained_layout=True, sharex=True, sharey=True)
for ax, df, title, c in zip(axs, [CASACNP, MIMICS, CORPSE],
                            ["CASA-CNP", "MIMICS", "CORPSE"], SRC_COLORS):
    plot_model_predictions(ax, df["prediction"], title, c)
    ax.set_xlabel(r"observed F$_{new}$ ($\delta^{13}C$ based)")
axs[0].set_ylabel(r"predicted F$_{new}$")
for ax, label in zip(axs, "ABC"):
    ax.text(-0.18, 1.1, label, transform=ax.transAxes, fontsize=8, va="top", ha="left")
fig.savefig("figures/fig3_src_models.png", dpi=300, bbox_inches="tight")
print("wrote figures/fig3_src_models.png")

#%% metrics table
print("\nmodel      KGE     RMSE")
for name, df in [("CASA-CNP", CASACNP), ("MIMICS", MIMICS), ("CORPSE", CORPSE)]:
    t = all_sites["total_fnew"].values; p = df["prediction"].values
    m = ~pd.isna(t) & ~pd.isna(p)
    kge = RegressionMetric(y_true=t[m], y_pred=p[m]).kling_gupta_efficiency()
    rmse = root_mean_squared_error(t[m], p[m])
    print(f"{name:9s}  {kge:5.2f}  {rmse:5.3f}  (n={m.sum()})")
