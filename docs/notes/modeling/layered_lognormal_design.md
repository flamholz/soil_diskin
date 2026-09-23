# Independent-layer log-normal model

Status: simplified at the user's request after comparing the full-profile runs.
This replaces the earlier transport design, retained in Git at `5ed9ae3`.
See the [run guide](layered_lognormal_usage.md) for the current interface.

## Decision

Remove diffusion D and downward advection v from the active model. Soil layers
are independent. Keep one shared, supplied NPP input e-folding depth h.

There are 20 local parameters per profile (ten mu/sigma pairs), and one shared
h: 20S + 1 parameters for S profiles, rather than 20S + 3. NPP is an observed
forcing. The initial value h = 30 cm remains a run input, not a calibrated estimate.

The three-scenario comparison used 50 profiles at 35 coordinate pairs and 500
layer observations. At h = 30 cm, zero transport gave RMSE 0.117376 and KGE (2012)
0.577023. The user chose the simpler no-transport model. This choice used the
observed new-carbon fractions; those data are still excluded from local parameter
fitting, but are no longer an independent test of the model-selection decision.

## Equations

Ten layers span 0–100 cm in 10 cm steps. Each receives

    I_i = NPP × [exp(-z_top/h) - exp(-z_bottom/h)] / [1 - exp(-100/h)].

For u = ln(k), new inputs have density p_i(u) = Normal(mu_i, sigma_i).
Each layer evolves independently:

    dc_i(u,t)/dt = I_i p_i(u) - exp(u)c_i(u,t).
    C_i = I_i exp(-mu_i + sigma_i²/2).

The stock-weighted log-rate density is Normal(mu_i - sigma_i², sigma_i).
Average the following responses over that density:

    Fm_i = average[k × integral_0^infinity F_atm(a) exp(-(k+lambda)a) da]
    f_new_i(t) = average[1 - exp(-kt)]
    lambda = 1/8267 year^-1.

All carbon starts at steady state. Labeling changes only the new/old label,
not the amount or rate distribution of inputs. The inherited atmospheric history,
constant old-age tail, and year-2000 reference remain unchanged.

## Fitting and outputs

- Fit each layer's two parameters independently using the same scaled least
  squares as before: stock residual / (0.1 × observed stock), fm residual / 0.02.
- Keep mu bounds [-15,10], sigma bounds [0.05,5], and no depth regularization.
- Use three deterministic starting sigmas (2.5,1,4), initializing mu from stock/input.
- Retain all starts; choose the smallest objective without consulting f_new.
  Report convergence, bound proximity, and the two-parameter Jacobian rank.
- Check numerical integration on a twice-finer grid. Keep failed fits visible.
- `layers.csv` is the main table: one row per layer, including inputs, turnover,
  parameters, observations, predictions, residuals, and numerical diagnostics.
- Separate tables retain all starts, requested-time predictions, near-best
  prediction ranges, exclusions, and source/settings metadata. Create the f_new
  scatter plot and RMSE/KGE directly at the end of the run.

Complete-data selection and profile identity are unchanged: ten positive layer
stocks, ten finite native-cell Shi targets, and positive cached NPP. Missing
f_new does not exclude a profile. Different named profiles at shared coordinates
remain distinct. Preserve units, source checksums, and labeling duration.

## Validation

Check input normalization, independent single-layer reference calculations,
constant-atmosphere and historical radiocarbon, f_new bounds and limiting times,
broad/narrow distributions, synthetic fits, alternate solutions, failure statuses,
data preparation, and evaluation isolation. Run all 50 current complete profiles
and compare their parameters and predictions with the saved zero-transport run.
