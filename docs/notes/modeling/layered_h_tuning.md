# Shared input-depth hyperparameter experiment

The fixed-grid experiment selected **h = 10 cm** by validation RMSE. On the
predeclared test split, it reduced f_new RMSE by **10.24%** relative to h = 30 cm.
It also worsened the stock/radiocarbon calibration, which is a material trade-off.

## Protocol fixed before the scan

- No-transport model; shared h in **5, 10, 15, 20, 30, 40, 60, 80, 120, 200 cm**.
- Seed 42; split the 35 distinct locations 60/20/20. Profiles at the same location
  stay together. The 35 locations also correspond to 35 distinct nearest Shi cells.
- Training: **26 profiles / 21 locations / 260 layers**.
- Validation: **12 profiles / 7 locations / 120 layers**.
- Test: **12 profiles / 7 locations / 120 layers**.
- For every h, fit each profile's local mu/sigma using only its own stock,
  radiocarbon, and NPP. Keep the original bounds, residual weights, and three
  deterministic starts, with at most 1,000 function evaluations per start.
- Select the lowest validation f_new RMSE; report KGE (2012) as a secondary metric.
  Equal weight per layer gives each ten-layer profile equal weight. Locations
  containing multiple profiles therefore contribute more observations.
- Failed or numerically unchecked validation layers make an h ineligible; never
  improve its score by dropping those layers. Every candidate was eligible here.
- Write `selection.json` before test fitting. Mask test f_new during local fits;
  reveal it only for final evaluation. Compare the selected h and the fixed
  h = 30 baseline; do not compare all h values on test or select from test scores.

This is conditional prediction at a new location with its stock/radiocarbon/NPP
available. The training subset supplies local fits and diagnostic curves; no
shared regression of local parameters is trained across locations. h is the only
quantity selected across profiles, using validation observations.

The split is **retrospective**: earlier full-data f_new comparisons informed the
choice to remove transport. It isolates this h search but does not undo that
prior model-selection use. The test is small (seven locations), not an external
validation dataset or 120 independent sites.

## Validation results

| h (cm) | RMSE | KGE (2012) |
| ---: | ---: | ---: |
| 5 | 0.144080 | 0.187998 |
| **10** | **0.126144** | **0.724975** |
| 15 | 0.130381 | 0.655433 |
| 20 | 0.135853 | 0.563452 |
| 30 | 0.142353 | 0.481358 |
| 40 | 0.145740 | 0.446373 |
| 60 | 0.149201 | 0.415286 |
| 80 | 0.150986 | 0.400937 |
| 120 | 0.152838 | 0.387127 |
| 200 | 0.154390 | 0.376278 |

Both validation criteria favor 10 cm on this grid. The training curve favors
15 cm (RMSE 0.110799, KGE 0.787491); training scores did not choose h.
The experiment identifies the best **tested grid value**, not a continuous optimum.

## Final test comparison

| h (cm) | RMSE | KGE (2012) | Converged layers |
| ---: | ---: | ---: | ---: |
| 10, selected | 0.083834 | 0.786183 | 120/120 |
| 30, baseline | 0.093401 | 0.419223 | 120/120 |

All test predictions passed the numerical accuracy check. The relative RMSE
reduction is 10.24%, or 0.00957 in fraction units (0.957 percentage points).

## Calibration trade-off

Selecting h by f_new rewards its predictive performance, not exact agreement
with calibration targets. h = 10 reaches parameter bounds in **11/120 validation
layers** and **14/120 test layers**; the test bound hits are at the sigma lower bound.
Its test stock relative RMSE is **16.14%**, and radiocarbon RMSE is **12.11 per mil**.
At h = 30, both calibration errors are essentially zero with no bound hits.

h = 15 has no validation bound hits and essentially exact calibration, with
validation f_new RMSE 0.130381. It was not evaluated on test because it was not
the selected candidate or predeclared baseline. Requiring a specified calibration
accuracy would be a different selection rule and should be fixed before a new
validation exercise; this report does not retroactively change the rule.

## Reproduce and inspect

```sh
uv run python notebooks/tune_layered_input_depth.py \
  --output-dir results/my_h_search --metric rmse --seed 42 --max-nfev 1000
```

The completed run is in `results/layered_h_tuning_seed42/`:

- `protocol.json`: frozen grid, criterion, split policy, solver settings, data and code hashes.
- `splits.csv`: profile/location assignments; `validation_scores.csv`: training and validation scores.
- `selection.json`: decision recorded before test fitting; `test_scores.csv`: final comparison.
- `validation_curve.png/.pdf`: the search curves.
- `test_comparison.png/.pdf`: the side-by-side test scatter plot, reproducible with the adjacent plotting script.
- `development/h_*/`: complete fitting outputs for each candidate on training/validation profiles.
- `test/h_10/` and `test/h_30/`: blinded fitting outputs, `evaluated_layers.csv`, and evaluation plots.

Checks confirmed disjoint location groups, stable splits under row reordering,
validation-only selection, complete fixed scoring cohorts, source/split hashes,
and that test fitting started after selection while its f_new inputs were masked.
The existing no-transport pipeline was reused without changes.

Verification: 21 focused tests passed; Ruff and type checking passed. The full
suite finished with 105 passed, 5 skipped, and one existing failure because the
Wolfram integration test requires an activated Mathematica license. Separate
standards and specification reviews found no issues.
