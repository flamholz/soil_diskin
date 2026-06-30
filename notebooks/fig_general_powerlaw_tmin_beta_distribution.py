#%%
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import viz


#%%
ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = ROOT / "results" / "03_calibrate_models"
OUT_FNAME = ROOT / "figures" / "general_power_law_tmin_beta_distribution.png"

plt.style.use(ROOT / "notebooks" / "style.mpl")
pal = viz.color_palette()


#%%
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


calibration_paths = sorted(
    CALIBRATION_DIR.glob("general_powerlaw_model_optimization_results_tmin_*.csv"),
    key=tmin_from_path,
)

if len(calibration_paths) == 0:
    raise FileNotFoundError(
        "No files matched "
        f"{CALIBRATION_DIR / 'general_powerlaw_model_optimization_results_tmin_*.csv'}"
    )

beta_records = []
for path in calibration_paths:
    df = pd.read_csv(path)
    if "beta" not in df.columns:
        raise ValueError(f"{path.name} does not contain a beta column")

    current = df[["beta"]].copy()
    current["t_min"] = tmin_from_path(path)
    current["t_min_label"] = tmin_label(path)
    beta_records.append(current)

beta_df = pd.concat(beta_records, ignore_index=True)


#%%
fig, ax = plt.subplots(figsize=(7.24, 2.4), dpi=300, constrained_layout=True)

tmin_values = [tmin_from_path(path) for path in calibration_paths]
tmin_labels = [tmin_label(path) for path in calibration_paths]
beta_values = [
    beta_df.loc[beta_df["t_min"] == tmin, "beta"].dropna().values
    for tmin in tmin_values
]
positions = np.arange(len(tmin_values)) + 1

box = ax.boxplot(
    beta_values,
    positions=positions,
    widths=0.55,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="k", lw=1),
    whiskerprops=dict(color=pal["dark_grey"], lw=1),
    capprops=dict(color=pal["dark_grey"], lw=1),
    boxprops=dict(edgecolor=pal["dark_grey"], lw=1),
)

panel_colors = [
    pal["dark_blue"], pal["blue"], pal["light_blue"],
    pal["dark_green"], pal["green"], pal["light_green"],
    pal["dark_purple"], pal["purple"], pal["light_purple"],
    pal["yellow"],
]
for i, patch in enumerate(box["boxes"]):
    patch.set_facecolor(panel_colors[i % len(panel_colors)])
    patch.set_alpha(0.55)

rng = np.random.default_rng(0)
for i, values in enumerate(beta_values):
    color = panel_colors[i % len(panel_colors)]
    jitter = rng.uniform(-0.18, 0.18, size=len(values))
    ax.scatter(
        positions[i] + jitter,
        values,
        color=color,
        edgecolor="k",
        lw=0.25,
        s=12,
        alpha=0.55,
        zorder=3,
    )

ax.set_xlabel("$t_{min}$ (yr)")
ax.set_ylabel("$\\beta$")
ax.set_xticks(positions)
ax.set_xticklabels(tmin_labels, rotation=45, ha="right")
ax.set_ylim(-0.03, 1.03)

ax.axhline(np.exp(-np.euler_gamma), color=pal["dark_grey"],
           linestyle="--", lw=1, zorder=-5)
ax.text(
    0.99,
    np.exp(-np.euler_gamma) + 0.02,
    "$e^{-\\gamma}$",
    transform=ax.get_yaxis_transform(),
    ha="right",
    va="bottom",
    fontsize=6,
    color=pal["dark_grey"],
)


#%%
OUT_FNAME.parent.mkdir(exist_ok=True)
plt.savefig(OUT_FNAME, dpi=300, bbox_inches="tight")
