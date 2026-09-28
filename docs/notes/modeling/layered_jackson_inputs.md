# Published root profiles as soil-carbon input profiles

Using Jackson et al. (1996) coefficients gives slightly lower pooled f_new RMSE
than h = 10 cm, but lower KGE. Both published-coefficient runs match the stock
and radiocarbon calibration targets to numerical precision, with no parameters
at their bounds. The comparison uses the same **87 profiles, 57 locations, and
800 observed layers**, including partial profiles and the recovered cached NPP.

## What changes

The paper's cumulative root distribution is `Y(d) = 1 - beta**d`, with depth
in centimetres. It is algebraically the existing exponential with
`h = -1 / log(beta)`. Thus this experiment introduces published coefficients and
vegetation assignments, not a new mathematical distribution. See the
[verified source note](jackson1996_source.md) for the paper, coefficient sources,
and limitations of its functional groups.

For each 10 cm layer, we allocate inputs as

```text
input = NPP × (beta**z_top - beta**z_bottom) / (1 - beta**100)
```

We assume that root biomass distribution approximates the distribution of
total carbon inputs. This does not separately represent aboveground litter or
root turnover. As in the existing model, all NPP is allocated within 0–100 cm;
missing layers do not cause inputs to be redistributed to observed layers.

Three fixed alternatives were run:

- **h = 10 cm:** the previous baseline, rerun on exactly the same data.
- **Jackson global:** the published global coefficient, equivalent to h = 28.91 cm.
- **Jackson vegetation:** published crop, grass, tree, and shrub coefficients,
  assigned from Balesdent land use and vegetation metadata. Ambiguous mixed
  vegetation and clover use the global coefficient.

The vegetation assignments are a coarse assumption, not a classification of
all eleven biomes in the paper. Forest stands in this cohort were treated as
temperate/tropical trees based on their location, climate, and recorded
vegetation; this mapping should be revisited before adding boreal stands.

| Assigned group | Profiles | Layers | Equivalent h (cm) |
| --- | ---: | ---: | ---: |
| Crops | 37 | 335 | 25.14 |
| Grasses | 21 | 200 | 20.33 |
| Trees | 22 | 197 | 32.83 |
| Shrubs | 1 | 10 | 44.95 |
| Global fallback | 6 | 58 | 28.91 |

The six fallback profiles are the Van Kessel FACE Trifolium profile, four Haile
sylvopastoral profiles, and Pessenda REBIO III Tabuleiro savanna. Their assignments
are explicitly flagged in `vegetation_assignments.csv`.

Coefficients and assignments were fixed before fitting and do not use observed
f_new. Each alternative refits local mu and sigma from stock, radiocarbon, and
allocated NPP, then predicts f_new at each profile's labeling duration. The
no-transport model, optimizer, bounds, and numerical checks are unchanged.

## Results

| Input allocation | f_new RMSE | KGE (2012) | Layers at bounds | Stock relative RMSE (%) | Radiocarbon RMSE (‰) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exponential h = 10 cm | 0.115305 | 0.699147 | 74 | 12.2301 | 10.4870 |
| Jackson global | 0.114521 | 0.593133 | 0 | <1e-10 | <1e-10 |
| Jackson vegetation | 0.114078 | 0.613862 | 0 | <1e-10 | <1e-10 |

All 800 layer fits in each alternative converged and passed the integration-grid
check. Every observed layer contributes equally to the f_new scores; partial
profiles contribute fewer layers than complete profiles. Stock and radiocarbon
are calibration targets, so their near-zero errors are not independent validation.

The vegetation alternative reduces f_new RMSE by **1.06%** relative to h = 10 cm,
while KGE decreases from **0.699 to 0.614**. The global alternative reduces RMSE
by 0.68% but has the lowest KGE. Neither published alternative improves both
metrics. Their clearest benefit is fitting the calibration targets without
bound constraints becoming active.

These are descriptive comparisons on the full available dataset. No coefficient
was tuned or selected on these results, and this is not a new independent test:
the observations already informed earlier model choices and the h = 10 baseline.
The small RMSE differences alone do not establish a robust predictive improvement.

## Reproduce and inspect

```sh
uv run python -m notebooks.compare_jackson_inputs \
  --output-dir results/my_jackson_comparison --max-nfev 1000
```

The [comparison script](../../../notebooks/compare_jackson_inputs.py) has four
steps: load the common cohort, freeze assignments, reuse the existing fitter,
and report every alternative. It adds no core-model code or dependencies.

Completed outputs are in `results/layered_jackson_1996/`:

- `comparison.png/.pdf`: observed versus predicted f_new for all three alternatives.
- `input_profiles.png/.pdf`, `input_weights.csv`: the depth allocation curves.
- `metrics.csv`: prediction scores and calibration diagnostics.
- `vegetation_assignments.csv`: profile metadata, assigned group, and coefficient.
- `<scheme>/layers.csv`: combined primary layer fits and predictions.
- `<scheme>/<group>/`: full existing pipeline outputs, including all fit starts.
- `protocol.json`: assumptions, frozen coefficients, source and assignment hashes,
  and completion status. Each group also retains its own `run.json`.

## Verification

The published cumulative formula independently reproduces existing exponential
layer weights. Tests also check metadata-only assignments and fallback flags.
All alternatives contain exactly the same 800 profile/layer pairs and observed
targets. The new h = 10 run reproduces the earlier saved mu, sigma, inputs, and
f_new predictions exactly. Both figures were visually checked.

Ruff and Mypy passed for the new code. The full test suite completed with
**108 passed, 5 skipped**, and the existing Wolfram integration failure because
the local Mathematica installation is not activated.

### Standards review

One minor provenance finding was resolved: the run now fingerprints the shared
scoring module as well as the model and comparison script. No documented
standards violations or actionable code-smell findings remain.

### Spec review

No findings. Independent checks confirmed the input formula, common cohort,
baseline reproduction, assignment independence, and reported metrics.

Review totals: Standards 1 resolved, 0 remaining; Spec 0.
