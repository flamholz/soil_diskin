# Independent-layer log-normal model

Status: active no-transport model, updated after input-allocation experiments
and the radiocarbon correction. The earlier transport design is retained in Git
at `5ed9ae3`. See the [run guide](layered_lognormal_usage.md) for the interface.

## Decision and parameter counts

Soil layers are independent: diffusion D and downward advection v are removed.
Each retained layer fits its own mu/sigma pair to stock and radiocarbon. NPP is
an observed forcing, not a fitted parameter. For L retained layers there are 2L
locally fitted parameters (20S for S complete ten-layer profiles).

`InputAllocation` contains three supplied settings: input e-folding depth h,
direct surface-input fraction s, and soil NPP fraction q. Defaults are 30 cm,
0, and 1. These remain fixed during a local fit; they are not three extra
unknowns that the stock/radiocarbon observations must identify. If all three
were estimated globally, the parameter count would be 2L + 3. The original
exponential-only case fixed s=0 and q=1, leaving 2L + 1 including h.

The [h-tuning experiment](layered_h_tuning.md) selects h on validation f_new
using a frozen location-level train/validation/test split; it refits local
parameters at each candidate h. The core fitter never sees observed f_new.
The [Jackson comparison](layered_jackson_inputs.md) supplies published beta
coefficients, equivalent to h=-1/log(beta) in cm. Its vegetation-specific mode
uses fixed coefficient groups rather than one universal h. The
[surface mixture](layered_jackson_surface50.md) and
[half-NPP experiment](layered_jackson_npp50.md) supply s and q as scenarios,
without optimizing them on f_new. Pooled scenario scores are descriptive;
observed f_new already informed the earlier choice to remove transport.

## Equations

Ten layers span 0–100 cm in 10 cm steps. With 0≤s<1 and 0<q≤1:

```text
w_i(h) = [exp(-z_top/h) - exp(-z_bottom/h)] / [1 - exp(-100/h)]
a_i = (1-s) w_i(h) + s × indicator(i is the top layer)
I_i = NPP × q × a_i
sum(a_i) = 1; sum(I_i) = NPP × q
```

The distributed component includes the top layer. Thus q=s=0.5 places 25%
of original NPP directly into 0–10 cm and distributes another 25% across all
ten layers. Missing observations never change this full-column allocation.
Jackson uses w_i=(beta^z_top-beta^z_bottom)/(1-beta^100), exactly the same
exponential weights with the converted h. Root biomass is a proxy for input
depth, not a measured carbon-input distribution.

For u=ln(k), new inputs have density p_i(u)=Normal(mu_i,sigma_i):

```text
dc_i(u,t)/dt = I_i p_i(u) - exp(u)c_i(u,t)
T_i = exp(-mu_i + sigma_i²/2)
C_i = I_i T_i
```

The stock-weighted log-rate density is Normal(mu_i-sigma_i²,sigma_i).
Average the following responses over that resident density:

```text
Fm_i = average[k × integral_0^infinity F_atm(a) exp(-(k+lambda)a) da]
f_new_i(t) = average[1 - exp(-kt)]
lambda = 1/8267 year^-1
```

All carbon starts at steady state. Labeling changes only the new/old label.
Atmospheric history, its constant old-age tail, and the year-2000 reference
follow the original model.

## Implementation, fitting, and outputs

`run_profiles` is the entry point: it allocates inputs, calls `_fit_and_predict`
for each layer, and calls `_save_tables` before evaluation. All output tables
come from those fit records; prediction-time arrays expand into rows only when
saving. The public Python interface uses one `InputAllocation` object and
`FitResult` attributes, without legacy keyword or dictionary-access adapters.

The existing `LognormalDisKinFast` owns the numerical evaluator and updates
mu/sigma in place. Its optional cached quadrature integrates the resident
density across the fitting bounds. Its separate survival discretization uses
the input density; those weights are intentionally different. `LayerLognormal`
is a compatibility constructor configuring that existing model for layer fits.
Both lognormal classes share the analytic turnover helper in `lognormal.py`.

- Fit stock residual / (0.1 × observed stock) and fm residual / 0.02.
- Keep mu bounds [-15,10], sigma bounds [0.05,5], without depth regularization.
- Use starting sigmas (2.5,1,4), initializing mu from stock/input.
- Retain all starts as `FitResult` records, including failures. The smallest
  objective is primary; f_new never participates in that choice.
- Report convergence, bound proximity, Jacobian rank, and a twice-finer grid
  check. Keep failed fits visible rather than reducing the evaluation cohort.
- `layers.csv` includes inputs, parameters, predictions, residuals, diagnostics,
  `implied_turnover_years` (observed stock / modeled input), and
  `model_turnover_years` (T_i). `observed_turnover_years` remains a legacy alias
  for the implied quantity, which is not an independent observation.
- Retain all starts, requested-time predictions, near-best prediction ranges,
  exclusions, and settings/source fingerprints. Runs preserve existing output
  directories and mark interruptions or plotting failures explicitly.
- Always produce scatter PNG/PDF and RMSE/KGE (2012) at successful completion.
  With no evaluable observations, write zero pairs, NaN scores, explicit
  `no_evaluable_observations` status, and an explanatory figure. Nonfinite
  predictions invalidate scores; finite failed/unchecked predictions stay
  visible with counts. Hyperparameter selection additionally requires every
  expected layer to converge and pass quadrature checks.

## Data coverage and history

Current corrected inputs support **68 complete profiles at 47 locations
(680 layers)**, or **101 profiles at 65 locations (914 layers)** with partial
profiles enabled. The latter includes 33 partial profiles; 11 input profiles
have no usable layer stock. These counts describe the current local files,
not the original zero-transport cohort.

Complete selection requires ten positive stocks, ten finite Shi targets after
nearest-neighbor spatial filling, and positive cached NPP. Partial selection
applies these requirements per layer, preserving original depth indices.
Stocks, NPP, and excluded-layer parameters are not imputed. Missing f_new
removes only an evaluation pair. Named profiles at shared coordinates stay
separate. The data adapter retains raw metadata for vegetation assignment and
reuses gap-preserving stock differencing from `data_wrangling.py` without the
original analysis's profile grouping or weight imputation.

The [NPP recovery](layered_npp_recovery.md) reconciles Excel/CSV coordinate
roundoff at ten decimal places and rejects conflicting cached values. It does
not extrapolate across locations. Shi filling and float precision now match
line 66 of the original radiocarbon notebook, as documented by the
[parity check](layered_radiocarbon_parity.md).

Historical comparisons used 50 complete profiles/35 locations/500 layers,
then 61 complete profiles after NPP recovery, and 87 profiles/800 layers with
partial selection before corrected spatial filling. See the
[partial-profile analysis](layered_partial_profiles.md) and
[current corrected-target refits](layered_radiocarbon_refit.md). The original
h=30, D=v=0 result had RMSE 0.117376 and KGE 0.577023; those are historical
scores, not current-cohort performance.

## Validation

Tests cover input conservation, independent single-layer reference calculations,
constant/historical atmospheres, f_new limits, broad/narrow distributions,
synthetic fits, failures/interruption, data preparation, and evaluation isolation.

[`check_layered_regression.py`](../../../notebooks/check_layered_regression.py)
refits all 50 historical profiles from their frozen calibration inputs and
compares parameters and predictions with both saved independent-layer and
coupled D=v=0 results, including requested times. This isolates code changes
from later NPP/radiocarbon data corrections. Its optional current-refits check
verifies all ten saved scenario tables (9,140 layer/scenario predictions).
See the [review-fix validation record](layered_review_fixes.md) for results.
