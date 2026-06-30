#%%
"""
Variant of fig3.py that adds MIMICS and CORPSE predictions.

Outputs:
    figures/fig3_mimics_corpse.png
    figures/fig3_mimics_corpse_colored_by_labeling_duration.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import viz

from pathlib import Path
from matplotlib.colors import LogNorm
from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error


#%% Load data and style
all_sites = pd.read_csv("results/processed_balesdent_2018.csv")

plt.style.use("notebooks/style.mpl")
pal = viz.color_palette()


def read_prediction(fname):
    path = Path("results/04_model_predictions") / fname
    if not path.exists():
        print(f"Warning: missing {path}; plotting this model as NaN.")
        return pd.Series(np.nan, index=all_sites.index, name="prediction")

    values = pd.read_csv(
        path,
        header=None,
        names=["prediction"],
    )["prediction"]
    if len(values) != len(all_sites):
        values = values.reindex(all_sites.index)
    return values


powerlaw_predictions = read_prediction("power_law.csv")
lognormal_predictions = read_prediction("lognormal.csv")
gamma_predictions = read_prediction("gamma.csv")
CLM45_predictions = read_prediction("CLM45_fnew.csv")
JSBACH_predictions = read_prediction("JSBACH_fnew.csv")
MIMICS_predictions = read_prediction("MIMICS_fnew.csv")
CORPSE_predictions = read_prediction("CORPSE_fnew.csv")
RCM_predictions = pd.read_csv("results/04_model_predictions/RCM.csv")
RCM_predictions = RCM_predictions.reindex(all_sites.index)


model_specs = [
    ("power law model", powerlaw_predictions, pal["dark_blue"]),
    ("lognormal model", lognormal_predictions, pal["blue"]),
    ("gamma model", gamma_predictions, pal["light_blue"]),
    ("CLM4.5", CLM45_predictions, pal["dark_purple"]),
    ("JSBACH", JSBACH_predictions, pal["purple"]),
    ("MIMICS", MIMICS_predictions, pal["red"]),
    ("CORPSE", CORPSE_predictions, pal["light_red"]),
    ("CESM1 ($^{14}$C corrected)", RCM_predictions["CESM1"], pal["dark_green"]),
    ("IPSL-CM5A-LR ($^{14}$C corrected)", RCM_predictions["IPSL-CM5A-LR"], pal["green"]),
    ("MRI-ESM1 ($^{14}$C corrected)", RCM_predictions["MRI-ESM1"], pal["light_green"]),
]


def prediction_metrics(predictions):
    true_vals = all_sites["total_fnew"].to_numpy(dtype=float)
    pred_vals = pd.Series(predictions).to_numpy(dtype=float)
    mask = np.isfinite(true_vals) & np.isfinite(pred_vals)

    if mask.sum() < 2:
        return np.nan, np.nan, mask.sum()

    rmse = root_mean_squared_error(true_vals[mask], pred_vals[mask])
    kge = RegressionMetric(
        y_true=true_vals[mask],
        y_pred=pred_vals[mask],
    ).kling_gupta_efficiency()
    return kge, rmse, mask.sum()


def plot_model_predictions(ax, predictions, model_name, color):
    true_vals = all_sites["total_fnew"].to_numpy(dtype=float)
    pred_vals = pd.Series(predictions).to_numpy(dtype=float)
    mask = np.isfinite(true_vals) & np.isfinite(pred_vals)

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
            label="y=x", zorder=-10, lw=1)
    ax.scatter(
        true_vals[mask],
        pred_vals[mask],
        label=model_name,
        color=color,
        edgecolor="k",
        lw=0.5,
        s=20,
        alpha=0.9,
    )

    kge, rmse, n = prediction_metrics(predictions)
    props = dict(
        boxstyle="round",
        facecolor=pal["light_yellow"],
        edgecolor=pal["dark_grey"],
        alpha=0.8,
    )
    box_text = f"KGE = {kge:.2f}\nRMSE = {rmse:.2f}\nn = {n}"
    ax.text(
        0.05,
        0.95,
        box_text,
        transform=ax.transAxes,
        fontsize=6,
        verticalalignment="top",
        bbox=props,
    )
    ax.set_title(model_name)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))


def plot_model_predictions_cmap(ax, predictions, model_name, clabel, cmap, norm):
    true_vals = all_sites["total_fnew"].to_numpy(dtype=float)
    pred_vals = pd.Series(predictions).to_numpy(dtype=float)
    color_vals = all_sites[clabel].to_numpy(dtype=float)
    mask = np.isfinite(true_vals) & np.isfinite(pred_vals) & np.isfinite(color_vals)

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
            label="y=x", zorder=-10, lw=1)
    sc = ax.scatter(
        true_vals[mask],
        pred_vals[mask],
        label=model_name,
        c=color_vals[mask],
        cmap=cmap,
        norm=norm,
        edgecolor="k",
        lw=0.5,
        s=20,
        alpha=0.9,
    )

    kge, rmse, n = prediction_metrics(predictions)
    props = dict(
        boxstyle="round",
        facecolor=pal["light_yellow"],
        edgecolor=pal["dark_grey"],
        alpha=0.8,
    )
    box_text = f"KGE = {kge:.2f}\nRMSE = {rmse:.2f}\nn = {n}"
    ax.text(
        0.05,
        0.95,
        box_text,
        transform=ax.transAxes,
        fontsize=6,
        verticalalignment="top",
        bbox=props,
    )
    ax.set_title(model_name)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    return sc


#%% Main 10-panel figure
fig, axs = plt.subplots(
    nrows=2,
    ncols=5,
    figsize=(7.24, 4.0),
    dpi=300,
    constrained_layout=True,
    sharex=True,
    sharey=True,
)
axs = axs.flatten()

for ax, (title, predictions, color) in zip(axs, model_specs):
    plot_model_predictions(ax, predictions, title, color)

for ax in axs[5:]:
    ax.set_xlabel("observed F$_{new}$ ($\\delta^{13}$C based)")
for ax in axs[[0, 5]]:
    ax.set_ylabel("predicted F$_{new}$")

for ax, label in zip(axs, "ABCDEFGHIJ"):
    x = -0.25 if label in ("A", "F") else -0.15
    ax.text(x, 1.1, label, transform=ax.transAxes,
            fontsize=7, va="top", ha="left")

plt.savefig("figures/fig3_mimics_corpse.png", dpi=300, bbox_inches="tight")


#%% Duration-colored diagnostic version
fig, axs = plt.subplots(
    nrows=2,
    ncols=5,
    figsize=(7.24, 4.0),
    dpi=300,
    constrained_layout=True,
    sharex=True,
    sharey=True,
)
axs = axs.flatten()

cmap = "viridis"
norm = LogNorm(
    vmin=all_sites["Duration_labeling"].min(),
    vmax=all_sites["Duration_labeling"].max(),
)

for ax, (title, predictions, _) in zip(axs, model_specs):
    sc = plot_model_predictions_cmap(
        ax,
        predictions,
        title,
        clabel="Duration_labeling",
        cmap=cmap,
        norm=norm,
    )

for ax in axs[5:]:
    ax.set_xlabel("observed F$_{new}$")
for ax in axs[[0, 5]]:
    ax.set_ylabel("predicted F$_{new}$")

for ax, label in zip(axs, "ABCDEFGHIJ"):
    x = -0.25 if label in ("A", "F") else -0.15
    ax.text(x, 1.1, label, transform=ax.transAxes,
            fontsize=7, va="top", ha="left")

plt.colorbar(
    sc,
    ax=axs,
    orientation="vertical",
    label="time since transition (yrs)",
    pad=0.01,
)
plt.savefig(
    "figures/fig3_mimics_corpse_colored_by_labeling_duration.png",
    dpi=300,
    bbox_inches="tight",
)

#%%
