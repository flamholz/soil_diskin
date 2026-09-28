# Layered workflow review fixes

Historical record for commit `b7f6de5`. The later readability cleanup removed
legacy Python keyword/dictionary adapters; see the current
[short run guide](layered_lognormal_usage.md). Scientific calculations, CLI
flags, and saved CSV fields remain the same.

The supplied review was checked against `08c5d3f` plus the related uncommitted
half-NPP changes. This change includes those allocation changes and their
matching documentation; unrelated workspace files are outside its scope.

## Disposition of the review

| Finding | Resolution |
| --- | --- |
| Glossary and design disagree with input scenarios | Updated equations, three supplied allocation settings, parameter counts, validation selection, current and historical cohorts, and links to experiment notes. |
| Independent model duplicates built-in calculations | Added cached resident-density quadrature and in-place updates to existing `LognormalDisKinFast`; `LayerLognormal` preserves the prior constructor. Analytic turnover is shared by both lognormal models. The input-density survival discretization is intentionally distinct. |
| Duplicate cumulative-stock differencing | Both data adapters use `balesdent_layer_stocks`, preserving gaps and avoiding grouping/imputation in the layered adapter. |
| Allocation fields travel and validate separately | `InputAllocation` owns validation, soil/NPP fractions, physical layer inputs, and metadata; both experiment drivers use it. |
| Depth/weight vocabulary varies across files | Documented canonical API names and the historical CLI/CSV aliases, with units and distinct meanings. |
| Duplicate metrics and output handling | Shared RMSE/KGE in `layered_evaluation.py`; shared output protection and lifecycle in `run_output.py`. |
| Partial result dictionaries and vague workflow names | `FitResult` supplies a complete schema for success/failure; workflow uses `layer_input`, `layer_record`, `new_carbon`, and `near_best`. |
| Workbook reread through nested metadata | `PreparedProfiles.raw_profiles` retains metadata for Jackson vegetation assignment. |
| No evaluation output without observed f_new | Always write a metrics record and explanatory PNG/PDF, with zero pairs, NaN scores, and explicit status. Failed predictions cannot silently reduce the cohort. |
| Missing modeled turnover | Write `model_turnover_years` and `implied_turnover_years`; retain `observed_turnover_years` as a compatibility alias for the implied value. |
| Historical zero-transport comparison exists only in prose | Added executable frozen-input refit and comparison against both saved independent and coupled D=v=0 outputs. |

## Numerical regression

Reproduce from the repository root with the original saved results present:

```sh
uv run python notebooks/check_layered_regression.py \
  --output-dir results/my_layered_regression \
  --current-refits results/layered_radiocarbon_refit
```

The 50-profile/500-layer frozen-input refit exactly reproduced all saved
independent-layer mu, sigma, stock, fraction-modern, and f_new values. Against
the original coupled D=v=0 run, maximum absolute differences were:

| Quantity | Maximum absolute difference |
| --- | ---: |
| mu | 9.31e-13 |
| sigma | 4.97e-13 |
| stock, kg C/m² | 6.19e-14 |
| fraction modern | 5.22e-15 |
| f_new at labeling duration | 1.37e-13 |
| f_new at all saved times | 1.46e-13 |

All ten corrected-target scenario tables (914 layers each; 9,140 layer/scenario
predictions) also exactly reproduced saved stock, fraction modern, and f_new
with the refactored forward model. This latter check uses saved parameters;
it is not a new fit of all 9,140 cases. Original RMSE/KGE conclusions remain
unchanged. The regression artifact records reference-file and source hashes.

## Verification

Targeted checks cover model reuse, input conservation including partial profiles,
shared stock differencing, retained vegetation metadata, evaluation isolation,
absent observations, complete failure records, output protection, interrupted
fitting, and plotting failures. The full suite returned **124 passed, 5 skipped,
1 failed**. The failure is the existing Wolfram integration: the installed
Wolfram product reports that it is not activated. This is not a model assertion
failure. After fixing the review-discovered import dependency, the 47 affected
model/workflow tests passed, including a fresh-process test from outside the
repository with an explicitly supplied atmosphere.

Ruff passes for the changed layered modules, drivers, constants, utilities, and
layered tests. The existing `continuum_models.py` still has six unrelated lint
findings (an unused import and five legacy `I` parameter names); none was added
by this change. Mypy passes on the nine checked workflow/driver modules with
untyped external imports ignored. `git diff --check` passes. The scatter and
no-observation placeholder were inspected visually.

The final local artifact is
[`regression.json`](../../../results/layered_review_verification/regression.json),
with refreshed source hashes. Existing saved runs were preserved.

## Standards review

No remaining documented-standard breaches or actionable findings. The reviewer
identified an import-time dependency on the default atmospheric CSV and source
hash keys that were not resolvable paths. Both were fixed and independently
verified: default atmosphere loading is lazy, caller-supplied atmospheres work
outside the repository, and source records use absolute paths.

## Spec review

No remaining actionable findings. The reviewer confirmed all attached review
requirements and independently verified the import fix. The original default
atmosphere still loads and caches when requested by existing models.

Final review counts: Standards 0; Spec 0.
