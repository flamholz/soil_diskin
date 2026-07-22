#%%
"""
fig3_weibull.py

Reproduces the main panel of fig3.py (predicted vs observed F_new, y=x line,
KGE+RMSE box per model) over all 99 Balesdent sites, **with the Feng (2009)
hockey-stick / Weibull continuum model added as an extra panel (D)**.

The Weibull predictions in
results/04_model_predictions/weibull_model_predictions.csv are written by
notebooks/experimental/04c_collect_weibull_predictions.py, using parameters
calibrated by notebooks/experimental/03h_calibrate_weibull_model.py.

Run from the notebooks/ directory so the local `viz` module imports:
    cd notebooks && ../.venv/bin/python experimental/fig3_weibull.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error

# viz lives in notebooks/; make it importable regardless of where we run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import viz

# allow running from repo root or from notebooks/
if os.getcwd().endswith("experimental"):
    os.chdir("../..")
elif os.getcwd().endswith("notebooks"):
    os.chdir("..")

all_sites = pd.read_csv("results/processed_balesdent_2018.csv")
plt.style.use("notebooks/style.mpl")
pal = viz.color_palette()

P = "results/04_model_predictions"
powerlaw = pd.read_csv(f"{P}/power_law_model_predictions.csv")
lognormal = pd.read_csv(f"{P}/lognormal_model_predictions.csv")
gamma = pd.read_csv(f"{P}/gamma_model_predictions.csv")
weibull = pd.read_csv(f"{P}/weibull_model_predictions.csv")
CLM45 = pd.read_csv(f"{P}/CLM45_fnew.csv", header=None, names=["prediction"])
JSBACH = pd.read_csv(f"{P}/JSBACH_fnew.csv", header=None, names=["prediction"])
RCM = pd.read_csv(f"{P}/RCM.csv")

# the hockey-stick model gets a distinct accent colour
WEIBULL_COLOR = "#C44E52"


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
    p = np.asarray(predictions)
    m = ~pd.isna(t) & ~pd.isna(p)
    kge = RegressionMetric(y_true=t[m], y_pred=p[m]).kling_gupta_efficiency()
    rmse = root_mean_squared_error(t[m], p[m])
    props = dict(boxstyle="round", facecolor=pal["light_yellow"],
                 edgecolor=pal["dark_grey"], alpha=0.8)
    ax.text(0.05, 0.95, f"KGE = {kge:.2f}\nRMSE = {rmse:.2f}", transform=ax.transAxes,
            fontsize=6, verticalalignment="top", bbox=props)
    ax.set_title(model_name)
    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))


def err_of(df):
    return (df[["predicted_fnew_05", "predicted_fnew_95"]]
            .sub(df["predicted_fnew"], axis=0).abs().fillna(0).values.T)


#%% 3x3 grid: A-D continuum (incl. Weibull), E-F ESMs, G-I reduced-complexity
fig, axs = plt.subplots(3, 3, figsize=(5.6, 5.4), dpi=300,
                        constrained_layout=True, sharex=True, sharey=True)
axs = axs.flatten()

# continuum models A-D (Weibull is the new panel)
for ax, df, title, c in zip(
        axs[:4],
        [powerlaw, lognormal, gamma, weibull],
        ["power law model", "lognormal model", "gamma model",
         "hockey-stick model"],
        [pal["dark_blue"], pal["blue"], pal["light_blue"], WEIBULL_COLOR]):
    plot_model_predictions(ax, df["predicted_fnew"], title, c, err_of(df))

# ESMs E-F
for ax, df, title, c in zip(axs[4:6], [CLM45, JSBACH], ["CLM4.5", "JSBACH"],
                            [pal["dark_purple"], pal["purple"]]):
    plot_model_predictions(ax, df["prediction"], title, c)

# reduced-complexity G-I
for i, col in enumerate(RCM.columns):
    plot_model_predictions(axs[6 + i], RCM[col], col + r" ($^{14}C$ corrected)",
                           [pal["dark_green"], pal["green"], pal["light_green"]][i])

for j in (6, 7, 8):
    axs[j].set_xlabel(r"observed F$_{new}$ ($\delta^{13}C$ based)")
for j in (0, 3, 6):
    axs[j].set_ylabel(r"predicted F$_{new}$")
for j, label in zip(range(9), "ABCDEFGHI"):
    axs[j].text(-0.2, 1.12, label, transform=axs[j].transAxes,
                fontsize=7, va="top", ha="left")

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig3_weibull.png", dpi=300, bbox_inches="tight")
fig.savefig("figures/fig3_weibull.svg", bbox_inches="tight")
print("wrote figures/fig3_weibull.png")

#%% metrics table for the continuum models (incl. Weibull)
print("\nmodel          KGE     RMSE")
for name, df in [("power law", powerlaw), ("lognormal", lognormal),
                 ("gamma", gamma), ("hockey-stick", weibull)]:
    t = all_sites["total_fnew"].values
    p = df["predicted_fnew"].values
    m = ~pd.isna(t) & ~pd.isna(p)
    kge = RegressionMetric(y_true=t[m], y_pred=p[m]).kling_gupta_efficiency()
    rmse = root_mean_squared_error(t[m], p[m])
    print(f"{name:13s}  {kge:5.2f}  {rmse:5.3f}  (n={m.sum()})")
