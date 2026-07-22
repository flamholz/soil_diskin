#%%
"""
fig_weibull_vs_others.py

Scatter the hockey-stick (Weibull) F_new predictions against the power-law and
lognormal continuum-model predictions, across all 99 Balesdent sites. Shows how
closely the three continuum models agree despite different survival functions
(all are calibrated to the same per-site turnover + 14C).

Run from anywhere:
    .venv/bin/python notebooks/experimental/fig_weibull_vs_others.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if os.getcwd().endswith("experimental"):
    os.chdir("../..")
elif os.getcwd().endswith("notebooks"):
    os.chdir("..")

sys.path.insert(0, os.path.join("notebooks"))
import viz

plt.style.use("notebooks/style.mpl")
pal = viz.color_palette()

P = "results/04_model_predictions"
weibull = pd.read_csv(f"{P}/weibull_model_predictions.csv")["predicted_fnew"]
powerlaw = pd.read_csv(f"{P}/power_law_model_predictions.csv")["predicted_fnew"]
lognormal = pd.read_csv(f"{P}/lognormal_model_predictions.csv")["predicted_fnew"]


def panel(ax, other, name, color):
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, zorder=-10, label="y=x")
    ax.scatter(weibull, other, color=color, edgecolor="k", lw=0.5, s=22, alpha=0.9)
    r = np.corrcoef(weibull, other)[0, 1]
    bias = np.mean(other - weibull)
    props = dict(boxstyle="round", facecolor=pal["light_yellow"],
                 edgecolor=pal["dark_grey"], alpha=0.8)
    ax.text(0.05, 0.95, f"r = {r:.3f}\nmean({name}\N{MINUS SIGN}HS) = {bias:+.3f}",
            transform=ax.transAxes, fontsize=6, va="top", bbox=props)
    ax.set_xlabel(r"hockey-stick predicted F$_{new}$")
    ax.set_ylabel(rf"{name} predicted F$_{{new}}$")
    ax.set_title(f"{name} vs hockey-stick")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.5)); ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.set_aspect("equal")


fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.9), dpi=300, constrained_layout=True)
panel(axs[0], powerlaw, "power law", pal["dark_blue"])
panel(axs[1], lognormal, "lognormal", pal["blue"])
for ax, label in zip(axs, "AB"):
    ax.text(-0.2, 1.12, label, transform=ax.transAxes, fontsize=8, va="top")

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig_weibull_vs_others.png", dpi=300, bbox_inches="tight")
print("wrote figures/fig_weibull_vs_others.png")
