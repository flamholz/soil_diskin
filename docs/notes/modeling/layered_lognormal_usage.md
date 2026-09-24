# Layered pipeline: where to start

## Use the reported sampling intervals

The new analysis uses the workbook's **Layers** sheet: `Cstock` in kg C/m²,
`ratio_newCtoC` as observed `f_new`, and the supplied `zmid` for Shi radiocarbon.
NPP is integrated over the reported `z1–z2` interval. Fractional `zmid` values
linearly interpolate adjacent one-cm Shi depths after the existing spatial filling.

```sh
uv run python notebooks/01_preprocess_balesdent_data.py --sampled-layers
uv run python notebooks/02_get_turnover_14C.py --depth-resolved \
  --input results/processed_balesdent_2018_sampled.csv \
  --output results/all_sites_14C_turnover_sampled.csv --input-depth 10
uv run python -m soil_diskin.layered_workflow --allow-partial --max-nfev 1000 \
  --input-table results/all_sites_14C_turnover_sampled.csv \
  --output-dir results/my_sampled_fit
```

There are 615 usable reported intervals from 101 profiles, with 592 observed
fractions in [0, 1]. Missing/negative fractions do not exclude calibration rows;
the 23 negative observations are retained in `fnew_reported` for sensitivity checks.
Intervals must lie wholly within 0–100 cm, and `zmid` must lie inside both the
interval and Shi's 0–99 cm coordinates. Unsupported intervals are logged, never
clipped or assigned extrapolated radiocarbon. Layers retain their sheet order and
Excel row number. No stock values are backfilled.

Pass the same `--input-table` to `notebooks.compare_jackson_inputs` to compare
input allocations. See the [results versus the 10 cm analysis](layered_sampled_intervals.md).
Existing 10 cm files and defaults remain available for reproducing the earlier analysis.

## Previous 10 cm workflow

The pipeline now has three separate steps:

1. `01_preprocess_balesdent_data.py --depth-resolved` creates one row per original
   profile and 10 cm layer, including stock, `f_new`, coordinates, land use, and vegetation.
2. `02_get_turnover_14C.py --depth-resolved` attaches cached site NPP and layer
   radiocarbon, allocates NPP, and saves stock/input turnover in a reusable CSV.
3. [`run_profiles`](../../../soil_diskin/layered_workflow.py) fits those prepared
   inputs, predicts `f_new`, checks numerical accuracy, and saves/evaluates results.

Without `--depth-resolved`, both original preparation scripts retain their bulk
calculations. Depth mode uses the existing site NPP cache and requires no Earth
Engine connection. It retains missing observations; the CSV reader selects either
complete profiles or usable layers from partial profiles.

For the fitting equations, read `fit_layer` in
[`layered_lognormal.py`](../../../soil_diskin/layered_lognormal.py).
`layer_model(atmosphere)` configures the existing `LognormalDisKinFast`; its `predict`
method in [`continuum_models.py`](../../../soil_diskin/continuum_models.py)
calculates stock, radiocarbon, and new carbon. The
[design note](layered_lognormal_design.md) contains the equations and assumptions.

## Run the pipeline

From the repository root (use new output paths if these already exist):

```sh
uv run python notebooks/01_preprocess_balesdent_data.py --depth-resolved
uv run python notebooks/02_get_turnover_14C.py --depth-resolved
uv run python -m soil_diskin.layered_workflow --allow-partial --max-nfev 1000 \
  --output-dir results/my_layered_run
```

The default prepared file is `results/all_sites_14C_turnover_depth.csv`, with
`h=30 cm`, no direct surface input, and all NPP entering soil. Set `--input-depth`,
`--surface-fraction`, and `--soil-npp-fraction` on **script 02**, not the fitter.
For example:

```sh
uv run python notebooks/02_get_turnover_14C.py --depth-resolved --input-depth 10 \
  --output results/my_turnover_h10.csv
uv run python -m soil_diskin.layered_workflow --input-table results/my_turnover_h10.csv \
  --allow-partial --output-dir results/my_fit_h10
```

Omit `--allow-partial` to require ten usable layers per profile. Add `--limit 2`
for a small run, or `--times 1 10 100` for extra prediction times. Valid labeling
durations are always included. Each fit saves its actual `prepared_inputs.csv`.

Current data support 68 complete profiles/680 layers, or 101 profiles/914 layers
with partial profiles enabled. Each retained layer needs positive stock and NPP
and finite radiocarbon. Missing `f_new` only removes an evaluation observation.
Missing layers retain their original depths; inputs are always allocated over
all ten layers. The adapter preserves profile identities, never fills stock
or NPP gaps, and matches the original radiocarbon spatial filling. See the
[radiocarbon check](layered_radiocarbon_parity.md) and
[current results](layered_radiocarbon_refit.md).

## Change the NPP distribution without repeating spatial extraction

`allocate_inputs` takes a table and a callable. The callable receives seven named
columns: `latitude`, `longitude`, `z_top_cm`, `z_bottom_cm`, `land_use`,
`vegetation`, and `npp_kg_m2_yr`. It returns one layer input per row, in the same
order, in kg C/m²/year. It receives no stock, radiocarbon, or `f_new` observations.
Use absolute depth intervals and normalize over 0–100 cm, including missing layers.

```python
import numpy as np
from soil_diskin.layered_data import allocate_inputs, load_profiles
from soil_diskin.layered_lognormal import InputAllocation
from soil_diskin.layered_workflow import run_profiles
from soil_diskin.radiocarbon_utils import load_atm14c

# Example only: choose a deeper exponential for forest rows.
def npp_by_land_use(**columns):
    shallow = InputAllocation(10)(**columns)
    deep = InputAllocation(30)(**columns)
    return np.where(columns['land_use'].eq('FOREST'), deep, shallow)

prepared = load_profiles(allow_partial=True)
prepared.profiles = allocate_inputs(prepared.profiles, npp_by_land_use)
run_profiles(prepared, load_atm14c(), 'results/my_custom_inputs')
```

For a fixed exponential, simply use `allocate_inputs(layers, InputAllocation(10))`.
This updates `input_kg_m2_yr` and `implied_turnover_years` without rereading the
workbook or Shi raster. To save the alternative table, call `.to_csv(new_path,
index=False)` and pass that path as `--input-table` in future runs. Retain the full
prepared table when saving if you also need its excluded layers and profile counts.

`InputAllocation(h, s, q)` has three settings:

- `h`: exponential e-folding depth in cm (default 30).
- `s`: share placed in the top 10 cm (default 0, allowed `[0,1)`). The rest
  follows the exponential over all ten layers, including the top one.
- `q`: share of site NPP entering soil (default 1, allowed `(0,1]`).

Both `s=q=0.5` place 25% of original NPP directly in the surface layer and
another 25% along the exponential. `soil_input_fractions` sum to one;
`npp_fractions` sum to `q`; `layer_inputs(npp)` returns kg C/m²/year.

## Fit and predict one layer

```python
from soil_diskin.layered_lognormal import layer_model, fit_layer

model = layer_model(load_atm14c())
layer_input = InputAllocation(10).layer_inputs(npp=0.5)[0]
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
| `prepared_inputs.csv` | Exact layer inputs consumed by this fit |
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
