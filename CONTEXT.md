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

**Input e-folding depth (h)**: The depth increment over which the distributed
input density drops by a factor of e. It is supplied per run (or per vegetation
coefficient group in the Jackson comparison). A separate experiment can select
h on validation data; local mu/sigma fitting always holds it fixed.

**Surface-input fraction**: The fixed share of soil input placed directly in
0–10 cm. The remaining share follows the depth distribution over all ten layers.

**Input allocation**: The supplied h, surface-input fraction, and soil NPP
fraction together determine every layer input. The code represents these with
`InputAllocation(input_depth_cm, surface_fraction, soil_npp_fraction)`.

**Soil NPP fraction**: The fixed fraction of original site NPP entering the
modeled 0–100 cm column. Defaults to one; the half-NPP experiment uses 0.5.
The remaining NPP is outside the model.

**Layer input**: Site NPP times the soil NPP fraction times a depth weight.
Weights integrate the exponential over 0–100 cm, optionally mixing a fixed
direct surface share into 0–10 cm. They sum to one; layer inputs sum to the
soil share of NPP. There are no imports or exports between layers.

**Input decomposition-rate distribution**: A normal distribution of log(k), where
k is a decomposition rate in year⁻¹. Each layer has its own mu and sigma.

**Resident-carbon rate distribution**: At steady state without transport, the
normal distribution of log(k) has mean mu - sigma² and standard deviation sigma.
Slow classes accumulate more carbon than fast classes.

**Implied layer turnover**: Observed stock divided by modeled layer input, in
years. This depends on the input allocation; it is not a separate observation.

**Modeled layer turnover**: exp(-mu + sigma²/2), equal to modeled steady stock
divided by layer input. It agrees with implied turnover only to the extent that
the fitted stock matches its observation.

**New-carbon fraction (f_new)**: The fraction of a layer's steady-state carbon
that entered after labeling began, under unchanged inputs and decomposition.

**Primary fit**: The optimization start with the smallest stock/radiocarbon
objective for one layer. Its convergence and numerical checks remain explicit.
Different layers choose their primary fits independently.

**Model-selection evaluation**: The observed new-carbon fractions informed the
choice to remove transport. They do not enter local parameter fitting, but the
reported scores are development evaluation rather than independent validation
of that choice.
