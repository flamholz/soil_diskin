# Layered log-normal model: design interview

Status: accepted for implementation through the user's implement-skill invocation. See the [usage guide](layered_lognormal_usage.md) for the implemented interfaces and run instructions.

## Requested scope

- Fit a vertically resolved extension of the existing log-normal carbon model at Balesdent sites.
- Predict the fraction of new carbon for each layer from fitted parameters.
- Represent 0–100 cm by ten layers, each 10 cm thick, with a mu and sigma for each layer.
- Use finite-volume diffusion and upwind advection between adjacent layers, preserving the transported carbon's energy class.
- Close both transport boundaries, so redistribution conserves column carbon.
- Use diffusion D constant with depth and an effective downward bulk-carbon velocity v; share D and v across sites.
- Distribute site NPP exponentially with depth, with one input e-folding depth h shared across all sites, alongside D and v.
- Use layer carbon stocks from Balesdent and layer radiocarbon values from Shi as calibration targets.

The modeling decisions, equations, and delivery requirements below were agreed in the design interview.

## Findings from the existing project

- `soil_diskin/lognormal.py` uses mu and sigma for the input distribution of log decomposition rates. Its stock and radiocarbon formulas describe an isolated pool.
- `soil_diskin/data_wrangling.py` derives 10 cm layer stocks by differencing cumulative Balesdent stocks. It converts depth-point new-carbon fractions into layer estimates using adjacent-point averages, then produces a column aggregate. It also substitutes a mean stock profile for missing profiles; this is not a measured layer profile.
- `notebooks/02_get_turnover_14C.py` extracts the Shi gridded radiocarbon product and MODIS NPP, then aggregates radiocarbon to a column value. The new fitting pipeline needs to retain depth information.
- The [Shi dataset description](https://zenodo.org/records/3823612) identifies a globally gridded product inferred from 789 profiles, with 0.5-degree spatial resolution and values at 1 cm depth increments to 1 m. Extracted values at Balesdent sites should be distinguished from co-located direct radiocarbon measurements.

## Agreed: shared input depth and parameter counting

The user agreed to share h across all sites, just like D and v. Each site has 20 local distribution parameters and 20 stock/radiocarbon targets, conditional on the shared triple (D, v, h). Measured NPP supplies the input amplitude and is not an additional fitting constraint on these parameters.

For S calibration profiles, fitting all local and shared parameters jointly gives 20S + 3 parameters against 20S targets. The remaining parameter-count deficit is three globally. Matching local counts does not establish identifiability, a unique solution, or the existence of an exact fit for a given triple.

The initial infrastructure accepts specified shared triples and supports sensitivity scans; measured new-carbon fractions are reserved for evaluation, as agreed below.

With transport, the ratio of a layer's stock to its direct external input is not generally its residence time: imports and exports must enter the layer mass balance. The isolated-pool turnover identity cannot simply be applied layer by layer.

## Agreed: energy class and layer parameters

The user confirmed the direct extension of the existing model: use a common physical rate coordinate k across all layers. The external input to layer i has a log-normal decomposition-rate distribution with parameters mu_i and sigma_i. Transported carbon keeps its existing k, regardless of the destination layer's parameters.

A layer's resident carbon reflects local inputs, decomposition, and imports from other layers and is not constrained to a log-normal stock distribution. For example, carbon with k = 0.01 yr^-1 retains that rate when it moves from layer 1 to layer 2. The model does not remap incoming transported carbon to the destination layer's input distribution.

## Agreed: normalization of external inputs

The user confirmed that all site NPP is allocated within the modeled 0–100 cm column. For column depth H = 100 cm, layer boundaries z_i and z_(i+1), and shared h > 0, the input is

    I_(s,i) = NPP_s * [exp(-z_i/h) - exp(-z_(i+1)/h)] / [1 - exp(-H/h)].

This integrates the exponential profile over each layer and allocates all site NPP to the modeled column, so the layer inputs sum to NPP_s. It treats NPP as the effective soil-carbon input, consistent with the existing stock/NPP convention.

No external input is allocated below the modeled column. Inputs are distributed sources inside the column, so they do not conflict with closed transport boundaries.

## Agreed: starting state and new-carbon fraction

The user confirmed using the steady state of the coupled model for total carbon under constant inputs and parameters. At labeling time t = 0, label all subsequent external inputs as new while leaving their amount and decomposition-rate distributions unchanged; initially resident carbon is old. Define f_new_i(t) as the mass of labeled carbon currently in layer i divided by its total modeled steady-state carbon stock.

New carbon can enter any layer and subsequently move to the layer being evaluated. Carbon already in the column at t = 0 remains old even if it moves to another layer. This predicts replacement under unchanged carbon dynamics; it does not model a change in NPP or decomposition caused by a land-use transition. The radiocarbon forcing convention is agreed separately below.

## Agreed: choosing shared hyperparameters

The local fits are conditional on a shared triple (D, v, h). The stock and radiocarbon target count alone does not identify the three shared parameters.

The user's agreement to the recommendation is interpreted as: the initial infrastructure accepts explicit triples and supports sensitivity scans, while keeping measured Balesdent f_new values as independent evaluation data. No optimizer selects a best triple using f_new. Selecting a triple after inspecting its f_new performance would also use those data for tuning and would need to be reported as such.

## Agreed: radiocarbon forcing

The existing `AtmC14`/`load_atm14c` model uses a piecewise-constant atmospheric history indexed by years before 2000, with a constant atmospheric tail beyond the available history. The radiocarbon decay constant in the existing model is 1/8267 yr^-1. `load_atm14c` sets the tail value to the mean of the final 50,000 rows in the input file's original order; the code already notes that the rationale for this convention is unclear.

The user confirmed reusing this atmospheric history and year-2000 reference convention for the first version, with atmospheric radiocarbon supplied to all layer external inputs at the time they enter the soil and no additional vegetation-storage delay. Carry radiocarbon through the same transport and decomposition dynamics as bulk carbon, adding radioactive decay. Total carbon remains at steady state, while the isotope signal reflects the time-varying atmospheric history.

The local Shi NetCDF contains 100 depth levels but no reference-date metadata; its variable is named `temp` and has a `units: year` attribute even though it is used as delta-14C in the current pipeline. Its metadata alone therefore does not establish either the observation reference date or the physical units. The year-2000 convention is documented as inherited from the existing model, not verified as the Shi product's reference date. The adapter verifies the dataset identity against the published delta-14C MD5 (`645aa8d54cbc36cb329c29bcffc4352b`) at the [Zenodo record](https://zenodo.org/records/3823612) and records the unit override and reference-date limitation. The local file matches that checksum.

## Agreed: data eligibility

Inspection of the local Balesdent `Profiles` sheet found 112 profile rows at 74 distinct coordinate pairs. Differencing its cumulative stock columns yields all ten layer stocks for 70 rows; 68 rows have all ten stocks finite and strictly positive, spanning 47 distinct coordinate pairs. The other two complete rows contain a nonpositive layer stock. Of the incomplete rows, 11 have no layer stocks and 31 have only five to nine available layer stocks. These are counts before radiocarbon/NPP eligibility checks or a decision about grouping profiles into sites.

The user confirmed starting with profiles that have all calibration values: ten finite positive layer stocks, ten finite radiocarbon targets, and finite positive NPP. Report excluded profiles and reasons. Do not fill missing layer stocks using a mean profile or silently drop missing residuals, since either would change the information underlying the 20-parameter fit. Missing evaluation-only f_new values do not make a profile ineligible for calibration. Different named profiles at the same coordinates have separate fits, as agreed below.

## Agreed: calibration unit

The 68 profiles with complete positive stocks have 68 distinct nonmissing `Internal_profile_ID` values. Among their 47 coordinate pairs, 31 have one such profile, 12 have two, three have three, and one has four. Shared coordinates can identify different treatments or land uses, not only replicates: examples include grassland versus crops and pine plantation versus pasture. Some separately named profiles also have identical carbon stocks and differ in the reference vegetation used for isotope labeling.

The user confirmed treating each named Balesdent profile (`Internal_profile_ID`) as a separate calibration unit, retaining its own stocks and labeling duration. Obtain gridded NPP and Shi radiocarbon by coordinates, which can be shared by multiple profiles without merging their fitted local parameters. Shared gridded targets do not constitute independent measurements across those profiles. The current aggregated pipeline groups by coordinates and labeling duration; the new workflow preserves profile identity instead.

## Agreed: fitting criterion

A specified (D, v, h) triple need not admit an exact match to all 20 stock/radiocarbon targets, even for complete profiles. The user agreed to allow an approximate fit and expose the mismatch.

Use scaled nonlinear least squares across both types of targets and retain the best found parameters together with layer-level observed/predicted values, residuals, and convergence diagnostics. Numerical convergence alone must not be labeled an adequate scientific fit. Residual scales, handling of multiple solutions, and parameter bounds are agreed below.

## Agreed: residual weighting

The Balesdent profile columns used here do not include layer-specific stock standard errors, and the local Shi file contains a single gridded radiocarbon variable without uncertainty variables. Their numerical units cannot set the relative importance of the two target types.

The user accepted configurable residual scales with provisional defaults of 10% of each observed layer stock and 0.02 in fraction-modern units (20 per mil in delta-14C). Thus a 10% stock discrepancy and a 20 per mil radiocarbon discrepancy contribute equally to the sum of squared scaled residuals:

    r_C_i = (C_pred_i - C_obs_i) / (0.10 * C_obs_i)
    r_F_i = (F_pred_i - F_obs_i) / 0.02

These defaults are working objective weights, not measured uncertainties, priors, or statistically justified confidence thresholds. Fits and their metadata should record the actual scales used; no formal uncertainty claim follows from these weights alone.

## Agreed: depth regularization

The user confirmed preserving the requested 20 independent local parameters, with no smoothness or monotonicity penalty across layers in the first version. This keeps the baseline model's flexibility explicit. Depth smoothness would be an additional assumption about input rate distributions, not an implication of the transport equations; it could suppress sharp layer transitions and would require choosing a penalty strength.

Numerical and physical parameter bounds and checks for poorly constrained or multiple solutions remain necessary independently of whether depth regularization is used.

## Agreed: multiple fitted solutions

Equal numbers of local parameters and targets do not ensure a unique inverse solution. Different parameter sets may have similar objective values while yielding different layer f_new predictions, and a local optimizer can return different results from different starting points.

The user agreed to reproducible multiple starting points, retaining the best found fit as the primary result and preserving the parameters, objective values, diagnostics, and f_new predictions of other distinct converged candidates. Identify comparably good candidates using an explicit configurable objective tolerance and report their prediction spread. This spread describes ambiguity among solutions found by the fitting procedure; it is not a confidence interval or an exhaustive account of possible solutions. Never use the held-out f_new observations to choose among candidates.

## Agreed: numerical search bounds

The 99 valid whole-column fits in the local `03b_lognormal_predictions_calcurve_python.csv` imply mu between approximately -2.71 and 3.72 and sigma between 1.81 and 3.14, with k expressed in yr^-1. These are context for choosing search limits, not evidence that individual layers must lie in the same range.

The user accepted configurable numerical bounds mu in [-15, 10] and sigma in [0.05, 5], with flags for parameters that approach a bound. These are broad optimization limits, not claims about physically plausible ranges or hard bounds on individual decomposition rates. They constrain the log-normal parameters; the continuous rate distribution retains its positive unbounded support. Numerical integration must resolve its relevant tails independently of the optimizer limits.

## Consolidated forward model

Use centimetres for depth, years for time, and kg C per square metre of ground area for layer stocks. Thus D has units cm^2/yr, v has units cm/yr, and h has units cm; require D >= 0, v >= 0, and h > 0. The cached NPP from the existing site table is in g C/m^2/yr and must be converted to kg C/m^2/yr.

Let u = ln(k), with the numerical value of k expressed in yr^-1. For one profile, let c_i(u,t) be layer i's carbon stock density per unit u, and let p_i(u) be the normal density with parameters mu_i and sigma_i. The external source is b_i(u) = I_i p_i(u), with I_i given by the normalized exponential allocation above.

For equal layer thickness dz = 10 cm and downward-positive flux, the interior interface transfer rate is

    J_(i+1/2)(u) = D/dz^2 * [c_i(u) - c_(i+1)(u)] + v/dz * c_i(u).

Set the two boundary fluxes to zero explicitly. The equations are

    dc_i(u)/dt = b_i(u) - exp(u)*c_i(u) + J_(i-1/2)(u) - J_(i+1/2)(u).

Write T for the resulting transport operator on layer stocks, with destination layers indexing rows and source layers indexing columns. Its off-diagonal entries are nonnegative and each column sums to zero. For A(u) = T - exp(u)*Identity,

    c_ss(u) = -solve(A(u), b(u))
    C_i = integral c_ss_i(u) du.

Closed boundaries conserve carbon under transport alone. The full steady column balances total external input against total decomposition; it does not conserve stock in the absence of that balance.

For radiocarbon, use a tracer stock q in atmospheric-reference carbon-equivalent units:

    dq(u,t)/dt = b(u)*F_atm(t) + [A(u) - lambda*Identity] q(u,t),
    lambda = 1/8267 yr^-1,
    F_pred_i = integral q_i(u, reference_year) du / C_i.

The atmospheric history extends into the past with the existing constant-tail convention. Solve its historical response consistently with that tail rather than starting the tracer at zero at an arbitrary finite date. Convert the target delta-14C values with F_obs = 1 + delta14C/1000, following the existing pipeline convention.

For labeled new carbon under constant inputs,

    c_new(u,t) = c_ss(u) - expm(A(u)*t) c_ss(u),
    f_new_i(t) = integral c_new_i(u,t) du / C_i.

An equivalent integrated-source calculation is acceptable and can avoid cancellation near t = 0. A numerically stable evaluation must also cover D = 0, v = 0, and pure advection; it must not assume a well-conditioned transport eigendecomposition in every case. Quadrature accuracy must be checked independently of fit residuals, especially for the broad log-normal tails allowed by the bounds.

## Data preparation and verified starting cohort

- Read Balesdent profile rows without aggregating away `Internal_profile_ID`. Difference cumulative Ctotal stocks to obtain the ten layer stocks.
- Use the nearest Shi grid cell and average its ten 1 cm values within each 10 cm model layer, matching the existing pipeline's depth averaging. This treats carbon density as uniform within each modeled layer for this averaging step. Use explicit dimension names and depth ordering; do not depend on incidental array ordering.
- Apply the complete-data rule to the sampled values. Missing grid-cell targets are exclusions in this baseline; the old pipeline's nearest-neighbor filling of missing raster cells does not turn them into complete native targets.
- Join cached NPP by coordinates, validating that each coordinate pair has one distinct value. Preserve the input source and units in metadata.
- For evaluation, difference cumulative Cnew stocks and divide by the corresponding layer Ctotal stock. This compares layer-integrated observed fractions with the model's layer fractions directly. The source workbook has these columns; averaging the separate point-depth f columns is unnecessary for this cohort.
- Preserve original profile identity, coordinates, layer boundaries, labeling duration, and source provenance in outputs. Missing evaluation data do not affect fitting eligibility.

A read-only check of the current local inputs found:

| Filter | Retained profiles |
| --- | ---: |
| All ten layer stocks finite and positive | 68 |
| Also all ten radiocarbon targets finite in the native nearest cell | 61 |
| Also finite positive cached NPP | 50 |

The 50 eligible profiles span 35 coordinate pairs. All 50 also have finite positive labeling durations and complete layer fractions derived from Cnew/Ctotal; the derived fractions are within [0,1]. These counts describe the current local inputs and must be recomputed by the data adapter rather than hard-coded.

## Proposed implementation and outputs

1. Add a reusable Python forward model for coupled steady stocks, radiocarbon, and layer f_new at caller-specified times, using the repository's existing numerical dependencies and atmospheric data representation.
2. Add profile-level bounded least-squares fitting with configurable scales, bounds, reproducible multiple starts, and retained candidates. Report fit objective, residuals, numerical convergence, boundary proximity, and local sensitivity/rank diagnostics. Do not infer statistical confidence intervals from the provisional objective weights.
3. Add a profile-preserving adapter for the existing local data and a script/CLI that fits a specified shared triple or an explicit list of triples. Hyperparameter values are run inputs; the infrastructure does not silently infer them from evaluation data. No particular nonzero shared triple or scan range has been chosen in this interview.
4. Export profile/layer parameter and residual tables, candidate-level fit diagnostics, exclusions with reasons, run settings/provenance, and f_new predictions at each profile's labeling duration and optional caller-specified times. Every result must identify its profile, shared triple, and solution candidate.
5. Document an end-to-end local example. Reuse the existing single-layer implementation as a reference and retain its current behavior.

Implementation details such as the integration method and number of starts can be chosen based on numerical checks and runtime, with controls exposed and saved in run metadata. If a profile fails to converge, preserve its status and best evaluable candidate without presenting it as a successful fit. Continue with other profiles and triples.

## Validation required before delivery

- Transport conservation and positivity, including zero transport, pure diffusion, pure downward advection, and closed boundary behavior.
- Exact input normalization and agreement with independent isolated-layer calculations when D = v = 0.
- Coupled steady-state layer balances and column decomposition equal to column input.
- Radiocarbon agreement with the existing isolated-pool model in the no-transport limit and with an independent constant-atmosphere calculation.
- f_new(0) = 0, values in [0,1], monotonicity under the agreed constant-input setup, approach toward one at sufficiently long times, and agreement with an independent transient calculation.
- Quadrature convergence for stocks, radiocarbon, and f_new across representative parameter regimes, including broad distributions and bound-adjacent candidates. Numerical errors must be small relative to fitting scales.
- Synthetic fits that recover forward observables, without assuming inverse parameters are uniquely identifiable; verify that observed f_new never enters the fitting objective or candidate selection.
- Data-adapter checks for cumulative-stock differencing, profile identity, NPP conversion, layer alignment, exclusions, and correct association of predictions with labeling duration.
- A smoke run using real eligible profiles and explicitly identified demonstration hyperparameters, plus relevant existing regression tests. A demonstration run is not an estimate of the globally shared hyperparameters.
