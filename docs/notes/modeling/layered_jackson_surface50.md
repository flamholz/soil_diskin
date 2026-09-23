# Half surface input, half Jackson root profile

Adding a fixed 50% direct surface input improves pooled f_new prediction on the
same **87 profiles, 57 locations, and 800 usable layers**. The vegetation-specific
mixture has RMSE **0.109402** and KGE (2012) **0.711995**, compared with 0.115305
and 0.699147 for h = 10 cm. All mixtures retain essentially exact stock and
radiocarbon calibration fits, with no parameters at their bounds.

## Interpretation and allocation

The experiment puts half of NPP directly into the top 10 cm, then distributes
the other half using Jackson across **all of 0–100 cm, including the top layer**.
This interpretation was stated before running the experiment. It means the top
layer receives more than 50% in total; an exactly-50%-top model with the remainder
restricted to 10–100 cm has not been evaluated here.

If the previously normalized Jackson weight for a layer from a to b cm is

```text
w = (beta**a - beta**b) / (1 - beta**100)
```

then the new layer inputs are

```text
top 0–10 cm: input = NPP × (0.5 + 0.5*w)
all deeper layers: input = NPP × 0.5*w
```

The fractions sum to one across all ten layers. Missing observations do not
redistribute their allocated inputs, including when the top layer is missing.
The direct surface fraction is fixed at the user's requested 0.5; it is not an
additional fitted parameter and was not tuned to f_new.

The [published coefficients and vegetation assignments](layered_jackson_inputs.md)
are unchanged. Root biomass is a proxy for the distributed component's input
depth. The 50/50 split is our experimental assumption, not a result from the paper.

| Jackson group | Total NPP entering 0–10 cm |
| --- | ---: |
| Global | 65.10% |
| Crops | 66.72% |
| Grasses | 69.57% |
| Trees | 63.78% |
| Shrubs | 61.18% |

## Results

| Input allocation | f_new RMSE | KGE (2012) | Layers at parameter bounds |
| --- | ---: | ---: | ---: |
| Exponential h = 10 cm | 0.115305 | 0.699147 | 74 |
| Jackson global only | 0.114521 | 0.593133 | 0 |
| Jackson vegetation only | 0.114078 | 0.613862 | 0 |
| 50% surface + 50% Jackson global | 0.109502 | 0.694056 | 0 |
| 50% surface + 50% Jackson vegetation | **0.109402** | **0.711995** | 0 |

The vegetation mixture reduces RMSE by **5.12% versus h = 10 cm** and by
**4.10% versus vegetation-specific Jackson alone**. Its KGE improves against both.
The global mixture also improves substantially over global Jackson alone, but
its KGE remains slightly below h = 10 cm.

All 800 layer fits in each alternative converged and passed the integration-grid
check. Both mixed-input alternatives have stock relative RMSE below 1e-10 percent
and radiocarbon RMSE below 1e-10 per mil. The h = 10 comparison retains stock
relative RMSE 12.23% and radiocarbon RMSE 10.49 per mil. These are calibration
targets, so exact agreement does not constitute independent validation.

This is a descriptive comparison on the available dataset, with equal weight per
layer. It is not a new independent test, and the observed improvements do not
establish statistical significance. Local mu and sigma were refitted from stock,
radiocarbon, and allocated NPP; f_new only enters evaluation.

## Reproduce and inspect

```sh
uv run python -m notebooks.compare_jackson_inputs --surface-fraction 0.5 \
  --output-dir results/my_jackson_surface50 --max-nfev 1000
```

Completed outputs: `results/layered_jackson_surface50/`.

- `comparison.png/.pdf` and `metrics.csv`: all five alternatives.
- `input_profiles.png/.pdf` and `input_weights.csv`: original and mixed allocations.
- `<scheme>/layers.csv`: primary layer fits and predictions.
- `<scheme>/<group>/`: all starts and predictions from the existing pipeline.
- `protocol.json`, `vegetation_assignments.csv`, and group `run.json` files:
  frozen settings, source hashes, assignments, direct surface fractions, and
  complete ten-layer weights.

The existing comparison script adds two alternatives when `--surface-fraction`
is supplied. The standard workflow also accepts that option with any fixed h.
The allocation helper contains the mixture calculation; local fitting and
prediction equations are unchanged. Omitting the option reproduces the original
behavior.

## Verification

Independent cumulative-root calculations reproduce every mixed layer input.
All five alternatives retain the same cohort. The three original alternatives
reproduce all previously saved layer-fit columns exactly. Tests cover input
conservation, invalid fractions, the top layer's additional Jackson share,
partial profiles without a top observation, and f_new exclusion from fitting.
Both figures were visually checked.

Ruff and Mypy pass. The full suite completed with **110 passed, 5 skipped**, and
the existing Wolfram integration failure because Mathematica is not activated.

### Standards review

No findings. The allocation remains in one helper, defaults preserve existing
behavior, and saved settings record the surface fraction and full-column weights.

### Spec review

No findings. Independent checks confirmed conservation, partial-profile handling,
identical cohorts, baseline reproduction, and the reported metrics and interpretation.

Review totals: Standards 0; Spec 0.
