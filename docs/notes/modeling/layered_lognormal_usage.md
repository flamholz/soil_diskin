# Independent-layer log-normal pipeline

Each site has ten independent 10 cm layers. Each layer has its own `mu` and
`sigma`. There is **no transport**. The input e-folding depth `h` is shared,
supplied in centimetres (default: 30). An optional fixed surface-input fraction
can add a direct input to the first layer (default: zero).

## Run all complete profiles

From the repository root:

```sh
uv run python -m soil_diskin.layered_workflow \
  --input-depth 30 --output-dir results/layered_no_transport
```

This fits all eligible profiles and creates an observed-versus-predicted
`f_new` plot with RMSE and KGE. Use a new output directory for each run.
Add `--limit 2` for a small run or `--times 1 10 100` for extra prediction times.
Each profile's observed labeling duration is always included.

The current local inputs contain 68 complete profiles at 47 locations after
matching the original radiocarbon spatial filling. Each contributes ten layer
fits, so a complete-profile run contains 680 layer fits.

## Include partial profiles

```sh
uv run python -m soil_diskin.layered_workflow --input-depth 10 --allow-partial \
  --max-nfev 1000 --output-dir results/my_partial_profiles
```

`--allow-partial` includes every 10 cm layer with positive stock, finite spatially filled
radiocarbon, and positive site NPP, even if other layers in its profile are missing.
Depth indices stay unchanged: a 60–70 cm layer receives the original 60–70 cm
share of NPP. Inputs are still normalized over 0–100 cm, never over the observed
subset. Stocks, NPP, and local parameters are not imputed. Shi spatial gaps are
filled at each one-cm depth, matching the original analysis.

The current inputs support **101 profiles at 65 locations and 914 layers** this way.
See the [radiocarbon parity check](layered_radiocarbon_parity.md). Earlier saved
comparisons used 87 profiles and 800 layers before spatial filling was aligned;
their performance metrics have not been rerun as part of this data check.
The [NPP recovery report](layered_npp_recovery.md) and
[partial-profile report](layered_partial_profiles.md) document those earlier cohorts.

## Follow the code in this order

1. **Load data:** [`layered_data.py`](../../../soil_diskin/layered_data.py)
   reads Balesdent stocks, Shi radiocarbon, and cached NPP. It selects complete
   profiles or usable individual layers, differences cumulative stocks, converts NPP units,
   and records exclusions.
2. **Fit and predict one layer:**
   [`layered_lognormal.py`](../../../soil_diskin/layered_lognormal.py)
   contains `input_weights`, `LayerLognormal.predict`, and `fit_layer`.
3. **Run everything:** [`layered_workflow.py`](../../../soil_diskin/layered_workflow.py)
   contains `run_profiles`, with five numbered steps. It allocates NPP, fits
   each layer, predicts new carbon, saves the tables, and draws the scatter plot.

There are no transport matrices, matrix exponentials, coupled 20-parameter
optimizers, or D/v sensitivity scans. The old transport implementation and API
are available in Git at commit `5ed9ae3`; earlier saved results are preserved.

## What the model calculates

For layer top and bottom depths `z_top`, `z_bottom`, the annual carbon input is

```text
input = NPP × [exp(-z_top/h) - exp(-z_bottom/h)] / [1 - exp(-100/h)]
```

All site NPP is allocated within 0–100 cm. With no exchange between layers,

```text
turnover = observed_stock / input
predicted_stock = input × exp(-mu + sigma²/2)
```

An optional fixed `--surface-fraction s` puts that fraction of NPP directly into
0–10 cm and distributes the remainder with the same exponential over **all ten
layers**, including the top layer. If `w` is the exponential allocation above,
the mixed weights are `(1-s)*w`, with `s` added to the first weight. Thus `s=0.5`
gives the top layer **more than 50%** of total NPP. The default is zero, preserving
the original model. The fraction must lie in `[0,1)` so deeper layers retain input.
It is supplied, not fitted; `h` then describes the distributed component only.

`mu` and `sigma` describe the normal distribution of **log decomposition rates
in new inputs**, with rates in year⁻¹. Slow carbon accumulates at steady state:
the resident carbon's log-rate distribution has mean `mu - sigma²` and the
same standard deviation `sigma`.

`predict` averages two quantities over that resident distribution:

- Radiocarbon: the atmospheric-history response for each rate, including
  radioactive decay with mean life 8267 years.
- New carbon: `1 - exp(-k × time)` for each rate `k`.

`fit_layer` varies only `mu` and `sigma` to match the observed stock and fraction
modern. It retains the existing least-squares weights: 10% of observed stock and
0.02 fraction modern. These are working weights, not measured uncertainties.
Three fixed starting sigmas (2.5, 1, 4) make the procedure reproducible. Starting
mu uses the turnover identity above. The lowest objective is the primary result.
All starts are retained, including duplicates and unconverged results.

Thus there are **20 local parameters per profile, plus one shared h**, replacing
20 local parameters plus shared D, v, and h. Supplying h fixes it for that run;
the pipeline does not estimate it from the new-carbon observations.
For a partial profile, only the two parameters of each retained layer are fitted;
no parameters or predictions are inferred for its excluded layers.

## Start with `layers.csv`

One row per profile/layer, containing:

- Identity, depth, labeling duration, and observed stock, radiocarbon, and `f_new`.
- Allocated input, `observed_turnover_years`, fitted `mu` and `sigma`.
- Predicted stock, radiocarbon, and `fnew_pred` at the labeling duration.
- Residuals, optimizer `success`, bound flags, and `quadrature_ok`.

Check both `success` and `quadrature_ok`; convergence alone does not mean a good
scientific fit. Integration is checked by halving the log-rate spacing. Allowed
differences are 2e-5 fraction modern and 1e-6 new-carbon fraction; stocks are analytic.

Other outputs:

| File | Purpose |
| --- | --- |
| `fnew_scatter.png`, `.pdf` | Observed versus predicted layer fractions |
| `metrics.csv` | RMSE in fraction units and KGE (2012), with equal layer weights |
| `fits.csv` | All three optimization starts for every layer, best first |
| `predictions.csv` | Every start's predictions at each requested time |
| `prediction_spread.csv` | Min/max predictions from converged, numerically checked, near-best starts |
| `exclusions.csv` | Excluded profiles/layers and reasons; a blank `layer` means the whole profile |
| `run.json` | Settings, source fingerprints, counts, and run status |

Candidate IDs now refer to starts **within a layer**, not coupled whole-profile
solutions. Near-best means objective ≤ `best × 1.01 + 1e-6`. Prediction spreads
are not confidence intervals. The Jacobian rank is a local diagnostic of the
two-parameter layer fit, not proof of global uniqueness.

Tables are written when the fit loop finishes; a handled interruption saves
partial tables and marks the run incomplete. A hard process termination may
leave only the settings and exclusions. A fresh run never overwrites old files.

## Use one layer in Python

```python
from soil_diskin.layered_lognormal import LayerLognormal, fit_layer, input_weights
from soil_diskin.radiocarbon_utils import load_atm14c

model = LayerLognormal(load_atm14c())
layer_input = 0.5 * input_weights(30)[0]  # Site NPP 0.5 kg C/m²/year; top layer.
fits = fit_layer(model, stock=2.0, fm=0.95, input_rate=layer_input)
best = fits[0]
prediction = model.predict(best['mu'], best['sigma'], layer_input, times=(20.,))
print(prediction.fnew[0])
```

Bounds remain mu in [-15,10] and sigma in [0.05,5], configurable when constructing
`LayerLognormal`. `fit_layer` exposes the residual scales and evaluation budget.
The old `--hyper D V H` command is replaced by `--input-depth H`; the old
`layered_fitting` module and `LayeredLognormal(D,v,h,...)` interface were removed.

## Data conventions

- Default selection requires ten positive stocks, ten finite spatially filled
  radiocarbon targets, and positive NPP. With `--allow-partial`, the same checks
  apply per layer. Missing evaluation-only `f_new` is allowed for fitting, but
  those layers cannot contribute to observed-versus-predicted scores.
- Layer stocks use differences of adjacent cumulative stocks. A missing boundary
  invalidates both adjacent differences; it is never bridged or interpolated.
- Different named Balesdent profiles at the same coordinates remain separate.
- NPP is converted from g C/m²/year to kg C/m²/year. Stocks use kg C/m².
- NPP join keys round latitude/longitude to ten decimal places to reconcile
  workbook/CSV roundoff. Original coordinates used for Shi sampling are preserved.
  Conflicting NPP values at the same normalized coordinates raise an error.
  This recovers existing cached values; it does not extrapolate NPP across sites.
- Shi spatial gaps are filled with `rio.interpolate_na(method='nearest')` at each
  one-cm depth before nearest-cell site selection. The ten values are then averaged,
  using the original line-66 arithmetic and precision. No values are borrowed from
  other depths. Layers using a filled value have `radiocarbon_spatially_filled=True`.
  These are gridded estimates, not direct measurements at each Balesdent site.
- The adapter verifies the [published Shi file checksum](https://zenodo.org/records/3823612):
  the NetCDF incorrectly labels its delta-14C variable as years.
- Atmospheric history, its constant old-age tail, and the year-2000 reference
  follow the existing model. The Shi file does not specify a reference date.

Default files are `data/balesdent_2018/balesdent_2018_raw.xlsx`,
`data/shi_2020/global_delta_14C.nc`, `results/all_sites_14C_turnover.csv`, and
`data/14C_atm_annot.csv`. Override them with `--balesdent`, `--shi`, `--npp`, and
`--atmosphere`. See `--help` for the remaining run controls.

Observed `f_new` never enters local mu/sigma fitting. However, these observations
informed the choice to remove transport, so the reported scores are development
evaluation, not an independent test of model selection.

## Tune the shared input depth

The separate, readable experiment in
[`notebooks/tune_layered_input_depth.py`](../../../notebooks/tune_layered_input_depth.py)
reuses this pipeline. It groups profiles by location, fixes a train/validation/test
split, refits local parameters for each candidate h, selects h using validation
f_new, and then evaluates only that h and the predeclared 30 cm baseline on test.
The test f_new values are masked during local fitting. No shared regression for
mu/sigma is learned: a new profile still needs its own stock, radiocarbon, and NPP.

```sh
uv run python notebooks/tune_layered_input_depth.py --output-dir results/my_h_search
```

Defaults: seed 42; 60/20/20 by location; h in [5,10,15,20,30,40,60,80,120,200] cm;
selection by validation RMSE. `--metric kge_2012` changes the selection criterion.
Every candidate uses the same observations; a failed validation fit makes it
ineligible, rather than removing difficult observations from its score.
Settings, split assignments, validation scores, the locked selection, and final
test scores are saved separately. The [experiment report](layered_h_tuning.md)
records the completed study and explains its retrospective test split.
That historical study used 50 profiles before the NPP matching fix. A new search
now uses the expanded complete-data cohort and produces a different split.

## Compare published root-depth allocations

```sh
uv run python -m notebooks.compare_jackson_inputs --output-dir results/my_jackson_comparison
```

This runs h = 10 cm, Jackson et al. (1996)'s global coefficient, and published
coefficients assigned by vegetation group on the same usable layers, including
partial profiles. The paper's `1 - beta**depth` distribution is equivalent to
the existing exponential with `h = -1 / log(beta)`, so the core fitter is reused.
Root biomass is assumed to represent input depth; all NPP remains allocated
within 0–100 cm. No published coefficient or vegetation assignment is fitted to
f_new. See the [comparison report](layered_jackson_inputs.md) for the exact
assignments, results, limitations, and output files.

To additionally compare 50% direct surface input plus 50% Jackson input against
all three existing alternatives:

```sh
uv run python -m notebooks.compare_jackson_inputs --surface-fraction 0.5 \
  --output-dir results/my_jackson_surface50
```

The [surface-input comparison](layered_jackson_surface50.md) records the results
and exact interpretation. The surface fraction and full ten-layer weights are
saved in each group's `run.json`, even when some layers lack observations.

## Verification of the simplification

In the historical 50-profile cohort, all 500 layer fits at h = 30 cm converged and passed the finer-grid check.
Compared with the saved coupled implementation run at D = v = 0, the largest
mu/sigma difference was below 1e-12 and the largest f_new difference across
1,950 profile/layer/time predictions was below 1.5e-13. RMSE remains 0.117376
and KGE (2012) remains 0.577023.

The focused model, fitting, data, workflow, and existing log-normal tests passed
(19 tests), as did Mypy and Ruff. The full suite had 103 passing tests, 5 skips,
and one unchanged failure because the Wolfram installation is not activated.

### Standards review

No material findings against repository conventions or the code-smell baseline.

### Spec review

One workflow status issue was fixed: completion is recorded only after evaluation
and plotting succeed. A regression test checks that plotting failure preserves
fit tables while marking the run incomplete. Independent numerical probes found
no additional issues. Review totals: Standards 0; Spec 1 resolved, 0 remaining.
