# Soil carbon dynamics

This project studies soil carbon storage and replacement using decomposition-rate
distributions and isotope information. The active layered model has no transport.

## Language

**Soil column**: Soil from the surface to 100 cm, represented by ten independent
10 cm layers.

**Layer carbon stock**: Carbon mass per ground area within one depth interval,
not the cumulative stock from the surface.

**Calibration profile**: A named Balesdent profile with its own stocks, labeling
duration, and fitted layer parameters. Profiles at the same coordinates remain
separate and may share gridded NPP and Shi radiocarbon targets.

**Complete calibration profile**: Ten positive layer stocks, ten finite radiocarbon
targets after the original nearest-neighbor spatial filling, and positive NPP. New-carbon observations are evaluation data;
they are not required for fitting.

**Input e-folding depth (h)**: The depth increment over which input density drops
by a factor of e. It is the only shared model parameter and is supplied per run.

**Layer input**: The part of site NPP allocated directly to a layer by integrating
the exponential profile over its depth interval and normalizing over 0–100 cm.
The layer inputs sum to NPP. There are no imports or exports between layers.

**Input decomposition-rate distribution**: A normal distribution of log(k), where
k is a decomposition rate in year⁻¹. Each layer has its own mu and sigma.

**Resident-carbon rate distribution**: At steady state without transport, the
normal distribution of log(k) has mean mu - sigma² and standard deviation sigma.
Slow classes accumulate more carbon than fast classes.

**Layer turnover time**: Stock divided by layer input, in years. Without transport,
this equals exp(-mu + sigma²/2) for the modeled steady stock.

**New-carbon fraction (f_new)**: The fraction of a layer's steady-state carbon
that entered after labeling began, under unchanged inputs and decomposition.

**Primary fit**: The optimization start with the smallest stock/radiocarbon
objective for one layer. Its convergence and numerical checks remain explicit.
Different layers choose their primary fits independently.

**Model-selection evaluation**: The observed new-carbon fractions informed the
choice to remove transport. They do not enter local parameter fitting, but the
reported scores are development evaluation rather than independent validation
of that choice.
