# Layered pipeline: where to start

Read [`run_profiles`](../../../soil_diskin/layered_workflow.py) first.
It follows four steps:

1. Allocate site NPP over ten 10 cm layers.
2. Fit each layer's `mu` and `sigma` to stock and radiocarbon.
3. Predict `f_new` and check numerical accuracy.
4. Save tables and compare predicted versus observed `f_new`.

In the same file, `_fit_and_predict` handles one layer and `_save_tables`
creates every output from the fit records. No separate prediction-record list
needs to be kept in sync. Output rows are sorted by profile and layer.

For the fitting equations, read `fit_layer` in
[`layered_lognormal.py`](../../../soil_diskin/layered_lognormal.py).
`layer_model(atmosphere)` configures the existing `LognormalDisKinFast`; its `predict`
method in [`continuum_models.py`](../../../soil_diskin/continuum_models.py)
calculates stock, radiocarbon, and new carbon. The
[design note](layered_lognormal_design.md) contains the equations and assumptions.

## Run the pipeline

From the repository root, using a new output directory:

```sh
uv run python -m soil_diskin.layered_workflow \
  --input-depth 10 --allow-partial --max-nfev 1000 \
  --output-dir results/my_layered_run
```

Omit `--allow-partial` to require ten usable layers per profile. Add `--limit 2`
for a small run, or `--times 1 10 100` for extra prediction times. Valid labeling
durations are always included. See `--help` for input-file overrides.

Current data support 68 complete profiles/680 layers, or 101 profiles/914 layers
with partial profiles enabled. Each retained layer needs positive stock and NPP
and finite radiocarbon. Missing `f_new` only removes an evaluation observation.
Missing layers retain their original depths; inputs are always allocated over
all ten layers. The adapter preserves profile identities, never fills stock
or NPP gaps, and matches the original radiocarbon spatial filling. See the
[radiocarbon check](layered_radiocarbon_parity.md) and
[current results](layered_radiocarbon_refit.md).

## One way to specify inputs

```python
from soil_diskin.layered_data import load_profiles
from soil_diskin.layered_lognormal import InputAllocation
from soil_diskin.layered_workflow import run_profiles
from soil_diskin.radiocarbon_utils import load_atm14c

allocation = InputAllocation(input_depth_cm=10, surface_fraction=0.5, soil_npp_fraction=0.5)
run_profiles(load_profiles(allow_partial=True), load_atm14c(),
             "results/my_python_run", allocation=allocation)
```

- `input_depth_cm` is the exponential e-folding depth h (default 30 cm).
- `surface_fraction` is the share of soil input placed directly in 0–10 cm
  (default 0; allowed range `[0,1)`). The remainder is distributed over **all**
  ten layers, including the top layer.
- `soil_npp_fraction` is the share of site NPP entering soil (default 1;
  allowed range `(0,1]`). The other share is outside the model.

With both fractions set to 0.5, 25% of original NPP enters the surface directly
and another 25% follows the depth distribution. `soil_input_fractions` sum to
one; `npp_fractions` sum to `soil_npp_fraction`; `layer_inputs(npp)` returns
kg C/m²/year. Read weights through `allocation.soil_input_fractions`; the separate
`input_weights` wrapper has been removed. The Python function accepts only
`allocation=...`; the old separate
`input_depth`, `surface_fraction`, and `soil_npp_fraction` arguments were removed.
CLI flags and saved CSV field names are unchanged.

## Fit and predict one layer

```python
from soil_diskin.layered_lognormal import layer_model, fit_layer

model = layer_model(load_atm14c())
layer_input = allocation.layer_inputs(npp=0.5)[0]
best = fit_layer(model, stock=2.0, fm=0.95, input_rate=layer_input)[0]
prediction = model.predict(best.mu, best.sigma, layer_input, times=(20.,))
print(prediction.fnew[0])
```

`fit_layer` returns typed records: use `best.mu`, not the former `best['mu']`.
The three deterministic starts, parameter bounds, residual scales, and numerical
checks are unchanged. Only stock and radiocarbon enter local fitting.

## Read the results

Start with `layers.csv`: one row per profile/layer, with inputs, fitted
parameters, predicted stock/radiocarbon/`f_new`, and diagnostics. Check both
`success` and `quadrature_ok`. Convergence does not guarantee a good scientific fit.

`implied_turnover_years` is observed stock divided by modeled input.
`model_turnover_years` is `exp(-mu + sigma²/2)`. The saved
`observed_turnover_years` column remains an alias for implied turnover.

| Output | Contains |
| --- | --- |
| `fits.csv` | All three starts, including unsuccessful fits |
| `predictions.csv` | Each start's predictions at requested and labeling times |
| `prediction_spread.csv` | Range across near-best, converged, checked starts; not a confidence interval |
| `metrics.csv`, `fnew_scatter.png` / `.pdf` | RMSE, KGE (2012), and observed versus predicted `f_new` |
| `exclusions.csv` | Excluded profiles/layers and reasons |
| `run.json` | Settings, counts, source hashes, and completion status |

`score(layers)` and `plot_comparison(layers, output)` both consume the primary
layer-fit table. The plot uses its labeling-time predictions directly, so no
second predictions-table join is needed. Metrics also include convergence and
calibration diagnostics from that same scoring function.

With no evaluable observations, the metrics contain zero pairs and NaN scores,
and the figure explains why. Failed predictions remain visible and invalidate
scores rather than shrinking the cohort. Completed layers are saved on handled
interruption; existing output directories are protected.

## Optional analyses

- [Tune h](layered_h_tuning.md): `notebooks/tune_layered_input_depth.py` selects
  on validation locations, then evaluates the selected h on test locations.
- [Jackson inputs](layered_jackson_inputs.md): `notebooks/compare_jackson_inputs.py`
  compares published coefficients. `--surface-fraction 0.5` adds surface mixtures;
  `--soil-npp-fraction 0.5` runs the half-NPP scenario.
- Check numerical agreement with saved runs:

```sh
uv run python notebooks/check_layered_regression.py \
  --output-dir results/my_regression \
  --current-refits results/layered_radiocarbon_refit
```

The regression uses frozen inputs for the original 50 profiles and optionally
checks all 9,140 saved scenario predictions. Those saved result files must be
present locally. Reported pooled scenario scores are development evaluations:
observed `f_new` previously informed the choice to remove transport.
