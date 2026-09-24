# Jackson inputs with only 50% of NPP entering soil

Historical 800-layer comparison. See [corrected-target refits](layered_radiocarbon_refit.md)
for the current 914-layer results after matching the original radiocarbon extraction.

Reducing total soil input to **0.5 × site NPP** leaves Jackson-only f_new RMSE
nearly unchanged and improves KGE. For the surface-plus-Jackson mixtures,
KGE also improves, but RMSE increases by about 2%. Thus halving input does not
uniformly improve predictive agreement.

All five depth allocations from the [previous comparison](layered_jackson_surface50.md)
were refitted on the same **87 profiles, 57 locations, and 800 observed layers**.
The Jackson coefficients, vegetation assignments, observed targets, labeling
durations, and original site NPP are unchanged. Every layer's mu and sigma were
refitted to stock and radiocarbon before predicting f_new; predictions were not
simply multiplied by 0.5.

## Input definition

Let `w` be the Jackson depth weight, normalized across the complete 0–100 cm column:

```text
w = (beta**z_top - beta**z_bottom) / (1 - beta**100)

Jackson only: input = 0.5 × NPP × w

Surface-plus-Jackson mixture:
  top 0–10 cm: input = 0.5 × NPP × (0.5 + 0.5*w)
  deeper layers: input = 0.5 × NPP × 0.5*w
```

For every scheme, full-column inputs sum to 50% of original NPP. The other
50% is outside the modeled soil column. In the optional mixture, the soil input
is split equally: 25% of original NPP enters the surface directly, and 25%
follows Jackson over all depths, including the top layer. For global Jackson,
the top layer therefore receives 15.10% of original NPP without the surface
component and 32.55% with it.

Missing layer observations never redistribute inputs. For each retained layer,
observed stock divided by input is exactly twice its value in the full-NPP run.
The fixed soil fraction is a requested scenario assumption, not a fitted
coefficient or an inference from Jackson's root data.

## Results

Each metric pools the same 800 observed layers with equal layer weights.
RMSE is in fraction units; KGE uses the 2012 definition.

| Depth allocation | RMSE, 100% NPP | RMSE, 50% NPP | KGE, 100% NPP | KGE, 50% NPP |
| --- | ---: | ---: | ---: | ---: |
| Exponential h = 10 cm | 0.115305 | 0.119798 | 0.699147 | 0.561642 |
| Jackson global | 0.114521 | 0.114338 | 0.593133 | 0.655182 |
| Jackson vegetation | 0.114078 | 0.114089 | 0.613862 | 0.672098 |
| Surface + Jackson global | 0.109502 | 0.111655 | 0.694056 | 0.721407 |
| Surface + Jackson vegetation | 0.109402 | 0.111887 | 0.711995 | 0.725249 |

Within the half-NPP runs, the global surface mixture has the lowest RMSE,
while the vegetation surface mixture has the highest KGE. Relative to its own
full-NPP counterpart, global Jackson-only RMSE decreases by 0.16%; vegetation
Jackson-only RMSE increases by 0.01%. The surface mixtures' RMSE increases by
1.97% and 2.27%, respectively.

The depth diagnostics show how pooled scores can conceal opposing changes.
For vegetation-specific Jackson alone, top-layer RMSE rises from 0.17931 to
0.19687, while RMSE improves in every deeper depth interval. Its top-layer
mean prediction bias becomes more negative, from −0.05026 to −0.09450.
Complete depth diagnostics are saved in `metrics_by_depth.csv`.

All 800 primary fits per scheme converged and passed the finer-grid integration
check. All four Jackson alternatives have zero layers at parameter bounds and
stock/radiocarbon calibration errors below 1e-10 percent/per mil, respectively.
For h = 10 cm, halving input increases layers at bounds from 74 to 120; stock
relative RMSE rises from 12.23% to 20.82%, and radiocarbon RMSE from 10.49‰ to
13.96‰. Optimizer convergence therefore does not imply an exact calibration fit.

These are descriptive comparisons on the existing development dataset, not
independent validation or a statistical significance test. Root biomass remains
a proxy for the depth distribution of carbon inputs.

## Reproduce and inspect

```sh
uv run python -m notebooks.compare_jackson_inputs \
  --soil-npp-fraction 0.5 --surface-fraction 0.5 \
  --output-dir results/my_jackson_npp50 --max-nfev 1000
```

Omit `--surface-fraction` to run only the exponential baseline and the two
Jackson-only alternatives. `--soil-npp-fraction` defaults to 1, preserving
earlier behavior. The standard layered workflow accepts the same option.

Completed outputs are in `results/layered_jackson_npp50/`:

- `comparison.png/.pdf`: observed versus predicted f_new for the five half-NPP schemes.
- `npp_sensitivity.png/.pdf` and `npp_sensitivity.csv`: paired full-NPP/half-NPP scores.
- `input_profiles.png/.pdf` and `input_weights.csv`: allocation as fractions of original NPP.
- `metrics.csv`, `metrics_by_depth.csv`, and `<scheme>/layers.csv`: scores and layer results.
- `protocol.json`, `vegetation_assignments.csv`, and group `run.json` files: inputs,
  settings, hashes, and completion status. Original site NPP and its soil fraction
  are saved separately; normalized depth weights still sum to one.
- `compare_previous.py`: independently verifies the saved runs and recreates
  the paired comparison and depth summaries from the repository root.

## Verification

All **25 relevant tests passed**, including both soil fractions, both surface
allocations, partial profiles, original NPP preservation, f_new exclusion from
fitting, and invalid fractions. Ruff passed for the changed analysis code.
Mypy was unavailable in the environment; no type-check result is claimed.

Independent calculations confirmed unchanged source metadata, assignments,
cohorts and observations; exactly halved inputs; doubled stock/input turnover;
the Jackson cumulative-depth formula; full-column input conservation; and saved
RMSE values. The comparison, allocation, and paired sensitivity figures were
visually checked.
