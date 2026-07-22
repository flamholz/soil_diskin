# Hockey-stick (Weibull) survival model pipeline

This documents an experimental continuum-model pipeline built around the
"hockey stick" survival function of Feng (2009), *Fundamental Considerations
of Soil Organic Carbon Dynamics: A New Theoretical Framework*, Soil Science
174(9):467–481. It mirrors the existing power-law pipeline
([`notebooks/03a_calibrate_powerlaw_model.py`](../03a_calibrate_powerlaw_model.py)
and [`notebooks/04_collect_continuum_model_predictions.py`](../04_collect_continuum_model_predictions.py)):
parameters are fit to each site's steady-state turnover time and radiocarbon
ratio, then used to predict the fraction of "new" carbon (`F_new`) after the
labeling duration at each Balesdent (2018) site.

## 1. The model

Feng writes the SOC decomposition function `R` (the fraction of a carbon
cohort still present at age `tau`) as the "H" / hockey-stick function

```
s(tau) = exp( -(k * tau)^alpha )
```

This is a **stretched-exponential / Weibull survival function** with a
rate-scale parameter `k` (yr⁻¹) and a shape parameter `alpha` (dimensionless).
It is what the codebase calls a survival function `s(tau)` — the fraction of an
input cohort remaining at age `tau`.

* `alpha = 1` recovers first-order (single-exponential) kinetics.
* `alpha < 1` produces the characteristic SOC shape: a sharp early drop and a
  very long, heavy tail. Feng fits `alpha ≈ 0.18–0.38` to real soils.

The model is implemented as `WeibullDisKin` in
[`continuum_models_experimental.py`](continuum_models_experimental.py),
subclassing `AbstractDiskinModel` so it reuses the framework's numerical
radiocarbon integration (`radiocarbon_age_integrand`) and steady-state
age-distribution machinery (`pA(tau) = s(tau)/T`).

## 2. Relating the parameters to steady-state observables

To calibrate `(k, alpha)` from the turnover time and ¹⁴C data we need the
steady-state relationships. At steady state with unit input the age
distribution is `pA(tau) = s(tau)/T`, where `T` is the mean transit time.

**Mean transit time (turnover) `T`.** Using the substitution `u = (k tau)^alpha`:

```
T = ∫₀^∞ s(tau) dtau = ∫₀^∞ exp(-(k tau)^alpha) dtau = Γ(1 + 1/alpha) / k
```

**Mean age `A`.**

```
A = (1/T) ∫₀^∞ tau · s(tau) dtau = Γ(1 + 2/alpha) / ( 2 k Γ(1 + 1/alpha) )
```

**Stabilization coefficient `A/T`.** Dividing the two:

```
A/T = Γ(1 + 2/alpha) / ( 2 Γ(1 + 1/alpha)^2 )
```

This depends on **`alpha` only**. It is exactly Feng's stabilization
coefficient `beta = Ta/MRT0`. So `alpha` controls the *shape* of the age
distribution (how heavy the tail / how much old carbon), while `k` sets the
overall *timescale*. `alpha = 1` gives `A/T = 1` (no stabilization, the
exponential case); smaller `alpha` gives `A/T >> 1`.

**Age-distribution CDF (the prediction we need).** The fraction of
steady-state carbon younger than age `t`:

```
cdfA(t) = (1/T) ∫₀^t s(tau) dtau = P(1/alpha, (k t)^alpha)
```

where `P` is the **regularized lower incomplete gamma function**
(`scipy.special.gammainc`). This is the model's `F_new(t)` after a step change
to zero old-carbon input — the same quantity the other continuum models report
via `cdfA`.

All four closed forms were verified against direct numerical quadrature (agree
to ≥6 significant figures), and the `alpha = 1` limit reproduces the
exponential model (`A = T`, `cdfA(t) = 1 - exp(-t/T)`).

`WeibullDisKin` also exposes `from_age_and_transit_time(A, T)`, which inverts
these relations (solving the `A/T` equation for `alpha` with a 1-D root find on
the log-gamma form to avoid overflow, then `k = Γ(1+1/alpha)/T`).

## 3. Pipeline scripts

All scripts are run from the **repository root** (the `constants` module
resolves data paths relative to the working directory).

### 3a. Calibration — [`03h_calibrate_weibull_model.py`](03h_calibrate_weibull_model.py)

For each of the 99 sites in `results/all_sites_14C_turnover.csv`, minimize the
same objective as the power-law pipeline:

```
objective = relative_diff_14C² + relative_diff_turnover²
```

where the predicted ¹⁴C ratio is `∫ pA(tau) · R_atm(tau) · exp(-lambda·tau) dtau`
(inherited `radiocarbon_age_integrand`) and the predicted turnover is the
closed-form `T`.

**Robust initialization.** A plain L-BFGS-B start (as used for the power-law
model) strands ~14% of sites at the initial guess, because the finite-difference
gradient is dominated by the `epsabs=1e-3` noise of the ¹⁴C quadrature. Instead,
each site is initialized with a **1-D scan over `alpha`** in which `k` is pinned
to the observed turnover via `k = Γ(1+1/alpha)/T` (so turnover is matched
exactly and only the ¹⁴C residual drives the scan). The best scan point and the
shared default guess are then both polished with Nelder-Mead, and the lower
optimum is kept. After this, **all 99 sites converge** (max objective ≈ 2.4e-6,
none stuck), with fitted `alpha ∈ [0.14, 0.30]` — squarely in Feng's reported
hockey-stick range.

The 5%/95% turnover-uncertainty bounds are calibrated the same way, mirroring
the power-law script. Output:
`results/03_calibrate_models/weibull_model_optimization_results.csv`.

> Note: the radiocarbon quadrature is occasionally slow (a few sites trigger
> roundoff-limited integration), so a full run takes on the order of tens of
> minutes.

### 3b. Predictions — [`04c_collect_weibull_predictions.py`](04c_collect_weibull_predictions.py)

Loads the calibrated parameters and, for each site, evaluates
`F_new = cdfA(Duration_labeling) = P(1/alpha, (k·Duration)^alpha)`, plus the
5%/95% columns from the uncertainty calibrations. Output:
`results/04_model_predictions/weibull_model_predictions.csv`
(same schema as the other `*_model_predictions.csv` files).

### 3c. Figure — [`fig3_weibull.py`](fig3_weibull.py)

Reproduces the main panel of `notebooks/fig3.py` (predicted vs. observed
`F_new`, `y=x` reference line, KGE + RMSE box per model) as a 3×3 grid and adds
the hockey-stick model as **panel D** (red), next to the power-law, lognormal
and gamma continuum models, the CLM4.5/JSBACH ESMs, and the three
reduced-complexity models. Output: `figures/fig3_weibull.png` / `.svg`.

## 4. Result

Skill against the 99 observed `F_new` values:

| model         | KGE  | RMSE  |
|---------------|------|-------|
| power law     | 0.61 | 0.115 |
| lognormal     | 0.83 | 0.107 |
| gamma         | 0.38 | 0.162 |
| **hockey-stick (Weibull)** | **0.56** | **0.137** |

Calibrated only from each site's turnover time and radiocarbon ratio, the
hockey-stick model predicts `F_new` with skill comparable to the power-law
model and better than the gamma model, though below the lognormal model. This
is consistent with Feng's argument that a two-parameter stretched-exponential
`R` captures the essential fast-initial-loss / heavy-tail structure of SOC
decomposition.
