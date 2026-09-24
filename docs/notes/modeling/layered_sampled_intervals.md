# Reported sampling intervals versus the previous 10 cm analysis

The analysis now supports the workbook's **Layers** sheet without rebinning it.
All ten previous input allocations were rerun: five depth distributions, each
with full or half NPP entering soil. The lognormal equations, three starting
points, optimizer settings, bounds, atmospheric history, site NPP, and Jackson
coefficients/mapping are unchanged. Stocks and radiocarbon fit each layer's
mu/sigma; observed `f_new` is used only for evaluation.

## Data and pairing

- Read `Profiles` with `skiprows=7` and `Layers` with `header=9`.
- Join `Layers.Identifier_1` to `Profiles.Identifier_1` to obtain the original
  profile identity, coordinates, labeling duration, land use, and vegetation.
  Only the sheet's profile identifier is forward-filled across a profile block.
- Use summary `Cstock` directly (kg C/m²), and `ratio_newCtoC` directly. Do not
  difference cumulative stocks or derive observed fractions from the Profiles sheet.
- Use the **supplied zmid**, which is not always `(z1+z2)/2`. At integer depths,
  sample that Shi level; at fractional depths, linearly interpolate the two
  adjacent levels. Spatial nearest selection and nearest filling match the
  existing workflow. Convert delta-14C to Fm as `1 + delta14C/1000`.
- Integrate NPP over each full `z1–z2` interval, still normalized over 0–100 cm.
  The direct surface component remains uniform over 0–10 cm and is split by
  interval overlap; it is not all assigned to the first reported interval.
  Missing/excluded intervals never receive redistributed inputs.
- Retain only positive-stock intervals wholly within 0–100 cm, with supplied
  `zmid` inside the interval and Shi's depth range 0–99 cm. Do not truncate an
  interval crossing 100 cm while keeping its full stock, or extrapolate Shi at depth.

The sheet contains 754 depth observations from 112 profiles. **615 layers from
101 profiles and 65 locations** have usable calibration data. This is the same
set of profiles/locations as the previous 914-layer analysis. All **6,150 primary
fits** converged and passed the finer-quadrature check.

The main comparison follows the previous [0, 1] observation-validity rule:
**592 reported f_new observations**. The other 23 are negative, remain unchanged
in `fnew_reported`, and are also evaluated in a separate sensitivity check. They
are not clipped to zero, and their calibration fits are retained.

## Full NPP results

Previous = 914 interpolated 10 cm observations; sampled = 592 valid reported
observations. These are pooled development scores, with equal weight per layer;
they are not an independent test, and different layer widths change the weighting.

| Input allocation | Previous RMSE | Sampled RMSE | Previous KGE | Sampled KGE |
| --- | ---: | ---: | ---: | ---: |
| Exponential h = 10 cm | 0.117198 | **0.128454** | 0.735200 | **0.787730** |
| Jackson global | 0.119231 | 0.136147 | 0.615121 | 0.629686 |
| Jackson vegetation | 0.118529 | 0.135494 | 0.635066 | 0.635595 |
| 50% surface + Jackson global | 0.113472 | 0.131557 | 0.715366 | 0.716560 |
| 50% surface + Jackson vegetation | 0.113059 | 0.128894 | 0.733661 | 0.727881 |

The h = 10 cm exponential now has the lowest RMSE and highest KGE among all ten
alternatives. Its RMSE advantage over the full-NPP vegetation/surface mixture is
only 0.000440; no significance claim is made. Previously the full-NPP vegetation
mixture had the lowest pooled RMSE. All full-NPP RMSE values increase with the
reported observations; KGE does not move uniformly with RMSE.

## Half NPP results

| Input allocation | Previous RMSE | Sampled RMSE | Previous KGE | Sampled KGE |
| --- | ---: | ---: | ---: | ---: |
| Exponential h = 10 cm | 0.121755 | 0.132437 | 0.596373 | 0.724491 |
| Jackson global | 0.118985 | 0.138396 | 0.687697 | 0.670918 |
| Jackson vegetation | 0.118480 | 0.137342 | 0.704155 | 0.682051 |
| 50% surface + Jackson global | 0.115447 | 0.131161 | 0.753111 | 0.746483 |
| 50% surface + Jackson vegetation | 0.115376 | **0.130559** | 0.756884 | **0.752095** |

The vegetation/surface mixture remains best within the half-NPP alternatives.
The surface share here refers to soil input: 25% of original NPP is direct
surface input and 25% follows Jackson.

## Separate the radiocarbon change from the layer-grid change

There are **213 exactly matching intervals**, from 60 profiles and 37 locations.
Their observed stocks, f_new, labeling times, site NPP, and allocated inputs
agree with the previous analysis to numerical precision. Thus on this subset,
only the radiocarbon target changes from a 10 cm mean to the supplied zmid value.
Mean absolute radiocarbon change is 11.503‰, with a maximum of 55.064‰.

| Full NPP, exact same 213 intervals | Previous RMSE | At-zmid RMSE | Previous KGE | At-zmid KGE |
| --- | ---: | ---: | ---: | ---: |
| Exponential h = 10 cm | 0.129274 | 0.136958 | 0.743100 | 0.730799 |
| Jackson vegetation | 0.140441 | 0.146098 | 0.556165 | 0.547828 |
| 50% surface + Jackson vegetation | 0.131888 | 0.141697 | 0.650627 | 0.642458 |

The higher pooled KGE of h = 10 cm therefore does **not** show that midpoint
radiocarbon improves predictions on unchanged observations. Its matched-interval
scores worsen slightly; the pooled result also changes the depth observations
and their relative weight. Shi's depth profiles can vary sharply: for Jaouadi's
10–20 cm interval, the 10 cm mean is Fm 0.966141, whereas its reported zmid of
15 cm gives Fm 1.021205. Both values were checked directly against the NetCDF.

Including all 615 reported fractions (including negatives) leaves full-NPP
h = 10 cm best, with RMSE **0.127376**, KGE **0.790828**. The vegetation/surface
mixture gives RMSE 0.129042, KGE 0.710676. No refitting is needed for this check.

Convergence does not imply exact calibration. On the 592 evaluation layers,
h = 10 cm still has 17 parameter-bound fits with full NPP and 38 with half NPP;
stock relative RMSE is 6.414% / 12.048%, and radiocarbon RMSE is 5.651‰ / 8.965‰.
The Jackson alternatives have no bound fits and match both calibration targets
to numerical precision.

## Reproduce and inspect

Prepare the sampled table using the commands in the [pipeline guide](layered_lognormal_usage.md).
Then run each NPP alternative with fresh output paths:

```sh
uv run python -m notebooks.compare_jackson_inputs \
  --input-table results/all_sites_14C_turnover_sampled.csv \
  --surface-fraction 0.5 --soil-npp-fraction 1 --max-nfev 1000 \
  --output-dir results/layered_sampled_intervals/npp100
uv run python -m notebooks.compare_jackson_inputs \
  --input-table results/all_sites_14C_turnover_sampled.csv \
  --surface-fraction 0.5 --soil-npp-fraction 0.5 --max-nfev 1000 \
  --output-dir results/layered_sampled_intervals/npp50
PYTHONPATH=. uv run python results/layered_sampled_intervals/summarize.py
```

- [Previous versus sampled scatter plots](../../../results/layered_sampled_intervals/previous_vs_sampled.png)
- [All five full-NPP plots](../../../results/layered_sampled_intervals/npp100/comparison.png)
- [All five half-NPP plots](../../../results/layered_sampled_intervals/npp50/comparison.png)
- [Every cohort/scenario metric](../../../results/layered_sampled_intervals/comparison_metrics.csv)
- [Matched-interval checks](../../../results/layered_sampled_intervals/matched_interval_checks.csv)
- [Exclusions](../../../results/layered_sampled_intervals/exclusions.csv)
- [Coverage and fingerprints](../../../results/layered_sampled_intervals/summary.json)

Each scenario retains all starts, fitted parameters, prediction spread, numerical
checks, the actual input table, and source hashes. Original 10 cm data/results
are unchanged. The new plots have also been saved as PDFs.

Validation: 30 focused tests passed, including exact parity with the original
10 cm radiocarbon extraction, supplied-midpoint interpolation, depth-range
exclusion, fractional surface-input allocation, and fitting a layer index above 9.
The full suite had 129 passed, 5 skipped, and the existing Wolfram integration
failure caused by its inactive local license. Ruff and Python compilation passed.
Standards and specification reviews found no actionable issues. Independent review
matched all 754 observations to the workbook, verified all 508 retained unfilled
Shi targets at supplied zmid, checked interval inputs in all ten scenarios, and
reproduced all 50 comparison metric records within 4.5e-16. All 26 run manifests
match the current scientific source files.
