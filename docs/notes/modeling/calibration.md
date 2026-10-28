# Calibrating continuum models

All continuum models are calibrated by the same code,
[`soil_diskin/calibration.py`](../../../soil_diskin/calibration.py), against
each site's turnover time and steady-state radiocarbon ratio (fm).

## The model contract

Every model in [`continuum_models.py`](../../../soil_diskin/continuum_models.py)
subclasses `AbstractDiskinModel` and provides:

- `T` and `A`: analytic steady-state transit time and mean age;
- `params_valid()`;
- `s(t)`: the survival function, with `s(0) = 1`, accepting scalars and arrays;
- `cdfA(t)`: the CDF of the steady-state age distribution `pA = s / T`;
- `calc_radiocarbon_ratio_ss()`: the steady-state fm, as `(value, error)`.

[`tests/test_model_contract.py`](../../../tests/test_model_contract.py) checks
every model against numerical integrals of its own `s(t)`: `T = ∫ s`,
`A = ∫ a pA`, `cdfA` against the integrated density, and fm against an
age-space integral.

## The atmosphere

`AtmC14` in [`radiocarbon_utils.py`](../../../soil_diskin/radiocarbon_utils.py)
is the only representation of the atmospheric 14C record: a piecewise-constant
lookup `R(a)` over age (years before 2000), holding the last value (at 55 kyr)
beyond the data. `default_atmosphere()` loads it once; models take `atm=None`
and load it on first use. `constants.INTERP_R_14C` is `default_atmosphere().R`.

## Radiocarbon: age space or rate space

By default, fm integrates the age distribution against the atmosphere:
`fm = ∫ pA(a) R(a) exp(-λa) da`, with adaptive quadrature.

A model defined by a rate distribution `p(k)`, whose Laplace transform is
`s(t)`, may define `log_rate_density(u)`, the density of `u = ln k`. Swapping
the order of integration gives `fm = (1/T) ∫ p(k) I(k + λ) dk`, where
`I(α) = ∫ R(a) exp(-αa) da` (`AtmC14.laplace`). `I` is tabulated once per
atmosphere on a fixed grid in `u` (`AtmC14.log_rate_response`), so each fm is a
single sum. The calculation raises if the grid does not hold both the input
mass and the 1/k-weighted stock. `LognormalDisKin` uses this path.

## Calibration

For each observation (turnover `T_obs`, fraction modern `fm_obs`),
`fit_observation` minimizes

    ((T - T_obs) / T_obs)^2 + ((fm - fm_obs) / fm_obs)^2

with bounded `scipy.optimize.least_squares` from several starting points.
A `ModelSpec` holds what differs between models: the free and fixed
parameters, the space they are optimized in, bounds, starting points, an
optional fast fm for the search, and an optional tie-break among exact fits.
Reported values always use the exact fm.

Each fit gets a `calibration_status`:

| status | meaning |
| --- | --- |
| `calibrated` | relative turnover error and absolute fm error both below the spec's `exact_tol` |
| `approximate_fit` | the optimizer converged, but not to within `exact_tol` |
| `fit_failed` | no starting point converged; values are NaN |
| `invalid_input` | turnover or fm missing or not positive; values are NaN |

`calibrate(spec, sites)` fits each site's central turnover and its q05/q95
stock scenarios (suffixes `''`, `_05`, `_95`), keeping every row.
`calibrate_sweep(spec_for, values, sites)` repeats this for each value of a
fixed parameter, such as alpha, writing one row per (value, site). Each site's
fit at one value starts from its fit at the previous value; with
`warm_start_scenarios`, the q05/q95 fits start from the central fit. Warm
starts never cross sites, so sites run in parallel (`n_jobs`, all cores by
default) with results independent of `n_jobs`.

Output columns: the parameters, `model_T`, `model_A`, `model_fm`,
`relative_turnover_error`, `relative_fm_error`, `objective`,
`calibration_status`, each with scenario suffixes.

## Specs and scripts

| script | spec | optimized in | fm during search | `exact_tol` |
| --- | --- | --- | --- | --- |
| `03a_calibrate_powerlaw_model.py` | `power_law_spec()` | log10 t_min, log10(t_max - t_min) | exact (quad) | 1e-6 |
| `03b_lognormal_calibration.py` | `lognormal_spec()` | mu, sigma | exact (rate space) | 1e-10 |
| `03c_calibrate_generalized_powerlaw_model.py` | `general_powerlaw_spec(alpha)` | log10 t_min, log10(t_max - t_min) | age grid (~1e-4) | 1e-3 |
| `03d_calibrate_weibull_model.py` | `weibull_spec()` | log10 k, alpha | exact (quad) | 1e-6 |
| `03e_calibrate_loguniform_model.py` | `loguniform_spec()` | log10 k_min, log10 width | exact (quad) | 1e-6 |
| `03f_calibrate_soft_cutoff_powerlaw_model.py` | `soft_cutoff_powerlaw_spec(alpha)` | log10 t_min, log10(t_max - t_min) | age grid (~1e-4) | 1e-3 |

`lognormal_spec()` keeps the highest-sigma fit when several match both
observations exactly: a fast site can match post-bomb radiocarbon two ways.

To calibrate a new model, add a spec to `calibration.py` and call `calibrate`
or `calibrate_sweep` from a short script, as the 03 scripts do. If the model
has a rate distribution, defining `log_rate_density` gives it an exact, fast fm.
