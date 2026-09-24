# Filling missing reported-layer stocks with SoilGrids

All **46 eligible missing stocks** were filled, restoring **112 profiles at 74
locations**. The sampled-layer model now fits **661 intervals**, versus 615 before
backfilling. There are **634 f_new observations in [0, 1]**, comprising 592 with
reported stocks and 42 with SoilGrids stocks. All ten input-allocation scenarios
were refitted, giving 6,610 converged and numerically checked primary layer fits.

## Method and provenance

The existing WCS reader downloaded SOC and bulk-density means at nine locations,
using the same projected ±125 m sampling box as the bulk pipeline. Values for
the five SoilGrids depth bands spanning 0–100 cm are cached in
`results/soilgrids_layer_cache.json`, with coordinates, retrieval times and WCS
settings. The published [SoilGrids units and depth bands](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html)
specify SOC in dg/kg and bulk density in cg/cm³; the existing reader converts
these to g/kg and g/cm³ respectively.

For each reported layer, use its overlap with each SoilGrids depth band:

```text
stock (kg C/m²) = sum(overlap_cm × SOC_g/kg × bulk_density_g/cm³ × 0.01)
```

This treats each SoilGrids band's SOC and density as constant within that band.
It follows the existing bulk calculation, including its omission of a
coarse-fragment correction. These are mean-map stock estimates; SoilGrids
uncertainty was not propagated through these layer fits. No uncertainty bounds
were inferred by multiplying independent marginal quantiles.

Summing the downloaded bands over 0–100 cm reproduces the original bulk
SoilGrids stocks at **all nine locations**, with maximum relative difference
2.22e-16. The native intervals need not cover a full meter, so sums of retained
interval stocks can be smaller than these bulk totals.

Only missing stocks in supported intervals are filled. Reported stocks, including
zero-thickness surface markers, remain untouched; depth exclusions from the
sampled-layer analysis remain in force. `stock_kg_m2_reported`, `stock_source`,
and `stock_fill_error` retain provenance. Missing required SoilGrids bands or
failed requests would leave a stock missing and record an error; none failed in
this run. No observations of f_new, NPP, labeling duration, or radiocarbon changed.

## Full NPP comparison

The before column evaluates 592 measured-stock observations; after evaluates all
634 observations, adding 42 observations from 11 profiles at 9 locations.

| Input allocation | Before RMSE | After RMSE | Before KGE | After KGE |
| --- | ---: | ---: | ---: | ---: |
| Exponential h = 10 cm | 0.128454 | 0.139173 | 0.787730 | **0.804046** |
| Jackson global | 0.136147 | 0.139769 | 0.629686 | 0.692877 |
| Jackson vegetation | 0.135494 | 0.139133 | 0.635595 | 0.698227 |
| 50% surface + Jackson global | 0.131557 | 0.136318 | 0.716560 | 0.760049 |
| 50% surface + Jackson vegetation | 0.128894 | **0.133909** | 0.727881 | 0.769484 |

The vegetation/surface mixture again has the lowest pooled RMSE, while h = 10 cm
retains the highest KGE. All original 615 layers have **exactly unchanged f_new
predictions** across all ten scenarios; calibration inputs and parameters also
match. Thus the pooled score changes arise entirely from the expanded cohort.
These are descriptive comparisons, not an independent held-out evaluation.

The 42 newly included observations alone show a different ranking:

| Full NPP, added observations only | RMSE | KGE |
| --- | ---: | ---: |
| Exponential h = 10 cm | 0.244551 | 0.470338 |
| Jackson global | 0.183365 | **0.766823** |
| Jackson vegetation | **0.182881** | 0.764500 |
| 50% surface + Jackson global | 0.191204 | 0.735027 |
| 50% surface + Jackson vegetation | 0.191068 | 0.729317 |

## Half NPP comparison

| Input allocation | Before RMSE | After RMSE | Before KGE | After KGE |
| --- | ---: | ---: | ---: | ---: |
| Exponential h = 10 cm | 0.132437 | 0.144298 | 0.724491 | 0.739774 |
| Jackson global | 0.138396 | 0.142838 | 0.670918 | 0.739645 |
| Jackson vegetation | 0.137342 | 0.141839 | 0.682051 | 0.747577 |
| 50% surface + Jackson global | 0.131161 | 0.137452 | 0.746483 | 0.785943 |
| 50% surface + Jackson vegetation | 0.130559 | **0.136921** | 0.752095 | **0.789014** |

The vegetation/surface mixture remains best within the half-NPP group. The
surface fraction refers to soil input, so this scenario puts 25% of original
NPP directly in 0–10 cm and 25% along the Jackson distribution.

## Calibration and out-of-range observations

All 6,610 primary fits converged and passed the finer-quadrature checks. However,
h = 10 cm still hits parameter bounds in 24 of the 634 evaluation layers with full
NPP, and 46 with half NPP. Its stock relative RMSE is 8.589% / 13.994%, and its
radiocarbon RMSE is 6.525‰ / 9.520‰. Jackson alternatives have no bound fits and
match both calibration targets to numerical precision.

Of the 46 newly calibrated layers, four report f_new outside [0, 1]: two negative
and two above one. They remain unchanged in `fnew_reported`. A sensitivity check
includes all 661 reported fractions, including the 23 earlier negatives. The
overall ranking remains: full-NPP vegetation/surface has the lowest RMSE
(0.134624), and full-NPP h = 10 cm has the highest KGE (0.809378).

## Reproduce and inspect

Use the [backfilled preparation commands](layered_lognormal_usage.md), then:

```sh
uv run python -m notebooks.compare_jackson_inputs \
  --input-table results/all_sites_14C_turnover_sampled_soilgrids.csv \
  --surface-fraction 0.5 --soil-npp-fraction 1 --max-nfev 1000 \
  --output-dir results/layered_sampled_soilgrids/npp100
uv run python -m notebooks.compare_jackson_inputs \
  --input-table results/all_sites_14C_turnover_sampled_soilgrids.csv \
  --surface-fraction 0.5 --soil-npp-fraction 0.5 --max-nfev 1000 \
  --output-dir results/layered_sampled_soilgrids/npp50
PYTHONPATH=. uv run python results/layered_sampled_soilgrids/summarize.py
```

Use new destinations when completed fit directories already exist.

- [Scatter plot distinguishing stock sources](../../../results/layered_sampled_soilgrids/stock_source_comparison.png)
- [All cohort/scenario scores](../../../results/layered_sampled_soilgrids/comparison_metrics.csv)
- [The 46 filled intervals](../../../results/layered_sampled_soilgrids/filled_layers.csv)
- [Preserved measured-layer predictions](../../../results/layered_sampled_soilgrids/regression_checks.csv)
- [Bulk stock parity at nine locations](../../../results/layered_sampled_soilgrids/bulk_stock_parity.csv)
- [Coverage and source fingerprints](../../../results/layered_sampled_soilgrids/summary.json)

Validation: 14 focused tests passed. The full suite had 131 passed, 5 skipped,
and the existing Wolfram integration failure because the local product is not
activated. New tests cover depth-overlap units, preserving measurements,
cache reuse, missing bands, and request failures. Python compilation and focused
Ruff checks passed. The source-colored scatter plot was visually checked.
Standards and specification reviews found no actionable issues. Independent
checks reproduced the stock calculation in SI units, all 50 metric records,
the unchanged measured-layer fits, and current artifact/source fingerprints.
