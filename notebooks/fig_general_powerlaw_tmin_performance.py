#%%
from pathlib import Path
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import viz

from permetrics.regression import RegressionMetric
from sklearn.metrics import root_mean_squared_error


#%%
ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT / "results" / "04_model_predictions"
OUT_FNAME = ROOT / "figures" / "general_power_law_tmin_performance.png"

plt.style.use(ROOT / "notebooks" / "style.mpl")
pal = viz.color_palette()


#%%
all_sites = pd.read_csv(ROOT / "results" / "processed_balesdent_2018.csv")


def tmin_from_path(path):
    match = re.search(r"tmin_(\d+p\d+)", path.stem)
    if match is None:
        raise ValueError(f"Could not parse tmin from {path.name}")
    return float(match.group(1).replace("p", "."))


def tmin_label(path):
    match = re.search(r"tmin_(\d+p\d+)", path.stem)
    if match is None:
        raise ValueError(f"Could not parse tmin from {path.name}")
    return match.group(1).replace("p", ".")


prediction_paths = sorted(
    PREDICTIONS_DIR.glob("general_power_law_tmin_*.csv"),
    key=tmin_from_path,
)

if len(prediction_paths) == 0:
    raise FileNotFoundError(
        f"No files matched {PREDICTIONS_DIR / 'general_power_law_tmin_*.csv'}"
    )

predictions = {
    path: pd.read_csv(path, header=None, names=["prediction"])["prediction"]
    for path in prediction_paths
}


# %% Define function to plot model predictions
def plot_model_predictions(ax, observed, prediction, model_name, color):
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
            label="y=x", zorder=-10, lw=1)
    ax.scatter(observed, prediction, label=model_name, color=color,
               edgecolor="k", lw=0.5, s=20, alpha=0.9)

    true_vals = observed.values
    pred_vals = prediction.values
    mask = ~pd.isna(true_vals) & ~pd.isna(pred_vals)
    evaluator = RegressionMetric(y_true=true_vals[mask],
                                 y_pred=pred_vals[mask])
    rmse = root_mean_squared_error(true_vals[mask], pred_vals[mask])
    kge = evaluator.kling_gupta_efficiency()

    props = dict(boxstyle="round", facecolor=pal["light_yellow"],
                 edgecolor=pal["dark_grey"], alpha=0.8)
    box_text = f"KGE = {kge:.2f}\nRMSE = {rmse:.2f}"
    ax.text(0.05, 0.95, box_text,
            transform=ax.transAxes, fontsize=6,
            verticalalignment="top", bbox=props)
    ax.set_title(model_name)
    ax.set_xticks(np.arange(0, 1.1, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.5))


#%%
n_files = len(prediction_paths)
ncols = min(5, n_files)
nrows = int(np.ceil(n_files / ncols))
figsize = (7.24, 1.75 * nrows)

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                        dpi=300, constrained_layout=True,
                        sharex=True, sharey=True, squeeze=False)
axs = axs.flatten()
used_axes = axs[:n_files]

panel_colors = [
    pal["dark_blue"], pal["blue"], pal["light_blue"],
    pal["dark_green"], pal["green"], pal["light_green"],
    pal["dark_purple"], pal["purple"], pal["light_purple"],
    pal["yellow"],
]

for i, (ax, path) in enumerate(zip(used_axes, prediction_paths)):
    color = panel_colors[i % len(panel_colors)]
    title = f"generalized power law\n$t_{{min}}$ = {tmin_label(path)} yr"
    plot_model_predictions(
        ax,
        all_sites["total_fnew"],
        predictions[path],
        title,
        color,
    )

for ax in axs[n_files:]:
    fig.delaxes(ax)

for i, ax in enumerate(used_axes):
    if i // ncols == nrows - 1:
        ax.set_xlabel("observed F$_{new}$ ($\\delta^{13}C$ based)")
    if i % ncols == 0:
        ax.set_ylabel("predicted F$_{new}$")

for i, ax in enumerate(used_axes):
    x = -0.25 if i % ncols == 0 else -0.15
    ax.text(x, 1.1, string.ascii_uppercase[i], transform=ax.transAxes,
            fontsize=7, va="top", ha="left")


#%%
OUT_FNAME.parent.mkdir(exist_ok=True)
plt.savefig(OUT_FNAME, dpi=300, bbox_inches="tight")
