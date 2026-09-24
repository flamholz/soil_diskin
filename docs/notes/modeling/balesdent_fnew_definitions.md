# Balesdent depth fractions and layer stock ratios

The two preprocessing paths use different observation constructions. The
`balesdent_layer_stocks` helper only differences cumulative stocks;
`balesdent_layers` divides the resulting new-carbon stock by total-carbon stock.
`process_balesdent_data` instead averages fractions at the two layer boundaries,
then combines layers using total-carbon weights. See
[the implementations](../../../soil_diskin/data_wrangling.py).

## Primary-source evidence

Source workbook:
[balesdent_2018_raw.xlsx](../../../data/balesdent_2018/balesdent_2018_raw.xlsx).
Cell references below use Excel's one-based row/column coordinates.

| Location | Evidence |
| --- | --- |
| `Profiles!A1:A2` | Identifies this workbook as supplementary information for Balesdent et al. (2018). |
| `Profiles!A4` | Describes profile data interpolated at 10 cm increments. |
| `Comments on variables!B5:C6` | Defines `f_0` and `f_10 ...` as new-carbon/total-carbon ratios **at a depth**, not averages over the preceding layer. |
| `Profiles!AG8:AQ8` | Headers for the point fractions `f_0` through `f_100`. |
| `Profiles!AT7:BD8`, `BF7:BP8` | Cumulative `Ctotal_0-z` and `Cnew_0-z` stocks, in kg/m². |
| `Layers!A4`, `F10:M11` | Raw sampled intervals (`z1`, `z2`, `zmid`), interval and cumulative stocks, and `ratio_newCtoC`. |
| `Comments on variables!B19:C21` | Describes stock provenance: author-reported stocks or calculations using carbon concentration and bulk density; new-carbon fractions may come from stock ratios or isotope data, with new stock then calculated as fraction × total stock. |

The publisher also lists the [supplementary workbook](https://www.nature.com/articles/s41586-018-0328-3)
as containing the raw, calculated, and ancillary data for the study. The local
workbook contains three sheets and **no Excel formulas or cell comments**;
the authors' exact interpolation code is not embedded in it.

## Why the calculations differ

For layer `[a,b]`, the layered path uses

```text
f_stock = (Cnew_0-b - Cnew_0-a) / (Ctotal_0-b - Ctotal_0-a)
```

The bulk path first uses

```text
f_endpoint = (f_a + f_b) / 2
```

Mathematically, a layer's new/total stock ratio is a carbon-mass-weighted fraction:
`integral(f(z) c(z) dz) / integral(c(z) dz)`, where `c(z)` is total carbon per
unit depth. The endpoint mean is a trapezoidal approximation to a depth average.
Even if carbon density were constant, interpolation of point fractions can
smooth changes between sampled layers and need not reproduce their stock ratios.
Constant carbon density and a linear fraction throughout a layer would make the
continuous expressions agree; the workbook does not impose that identity on its
separately tabulated point fractions and cumulative stocks. This paragraph is a
mathematical interpretation, not a claim about undocumented interpolation code.

A direct workbook example is **Dalal C4 buffel grassland, 10–20 cm**:

- `Profiles!BG9:BH9`: cumulative new stocks 0.2372366667 and 0.3701255556 kg/m².
- `Profiles!AU9:AV9`: cumulative total stocks 0.9831 and 1.7631 kg/m².
- Their difference ratio is **0.1703703704**, matching the sampled interval's
  fraction at `Layers!K15` (interval bounds `F15:G15` are 10 and 20 cm).
- `Profiles!AH9:AI9`: point fractions 0.2074074074 and 0.1574074074; their mean
  is **0.1824074074**, higher by **0.0120370370** (1.20 percentage points).

This example supports the stock-ratio interpretation for that observed interval.
It does not prove that every interpolated 10 cm interval is an exact original
observation or establish one universal interpolation method.

An especially large difference occurs for **Wilcke Pasture Brachiaria vs
Cerrado, 0–10 cm**. The workbook shows why:

| Quantity | Value | Workbook cells |
| --- | ---: | --- |
| Fraction at zero depth | 0.9253888014 | `Profiles!AG17`; `Layers!K133` |
| Fraction at 10 cm | 0.1850777603 | `Profiles!AH17` |
| Mean of those endpoints | 0.5552332808 | Calculated from the two cells above |
| New/total stock in top 10 cm | 0.5938250122 / 3.1155158359 = **0.1906024695** | `Profiles!BG17/AU17` |
| Sampled 0–15 cm fraction | **0.1906024695** | `Layers!F134:G134`, `K134` |

The zero-depth record has **zero thickness and zero carbon stock**
(`Layers!F133:J133`, `L133:M133`). The 0–10 cm total and new stocks are each
two-thirds of their 0–15 cm values (`Layers!I134`, `L134`), so their ratio retains
the sampled 0–15 cm fraction exactly. In contrast, averaging the two boundary
fractions gives half the weight to the high zero-depth value. This accounts for
the **0.3646308113** difference. The workbook does not identify the zero-depth
record's physical material explicitly; it should not be called a measured
mineral-soil layer or litter measurement without further source evidence.

The bulk routine also pools profiles sharing coordinates and labeling duration,
and substitutes average carbon weights when a profile has no stock weights.
The layered routine preserves original profiles and masks invalid fractions.
Those are additional processing differences, separate from the two formulas.

## Effect on the saved analysis

Audit: 2026-09-24, using the same **101 profiles and 914 layers** in all ten
corrected-radiocarbon scenarios. We held all fitted parameters and predictions
fixed, changing only the evaluation observation. The current h=10 refit and
the older corrected-target h=10 table agree exactly in parameters, inputs,
radiocarbon, stock, and predicted `f_new`.

Observed differences (endpoint mean minus stock ratio):

- 795 of 914 values differ by more than `1e-10`.
- Mean signed change: **+0.004703** (+0.47 percentage points).
- Mean absolute change: **0.016551** (1.66 percentage points).
- Median absolute change: **0.009135** (0.91 percentage points).
- 430 layers differ by more than one percentage point; 59 by more than five.
- Maximum absolute change: **0.364631** (36.46 percentage points), the Wilcke case above.

Endpoint means are available for 1,030 of the workbook's 1,120 rows, while stock
ratios are available for 914. All 116 extra endpoint observations lack positive
finite layer stocks, so they do **not** add calibratable layers to this model.

The two local calibration targets are stock and radiocarbon, with layer NPP
supplied as forcing. Observed `f_new` does not enter `fit_layer`, so **mu, sigma,
turnover, and predicted `f_new` do not change for a fixed input distribution**.
Evaluation and hyperparameter selection can change. See
[the fitting call](../../../soil_diskin/layered_workflow.py) and
[the objective](../../../soil_diskin/layered_lognormal.py).

### Fixed predictions, alternative evaluation labels

All scores below use exactly the same 914 pairs and KGE (2012), with equal weight
per layer. These are pooled development evaluations, not an independent test set.
“Surface” means half of soil input goes directly to 0–10 cm and the rest follows
Jackson roots; the percentage in the first column is the fraction of original
NPP entering soil.

| Soil NPP | Depth distribution | RMSE: stock ratio | RMSE: endpoints | KGE: stock ratio | KGE: endpoints |
| --- | --- | ---: | ---: | ---: | ---: |
| 100% | Exponential h=10 cm | 0.117198 | 0.114215 | 0.735200 | 0.724603 |
| 100% | Jackson global | 0.119231 | 0.116348 | 0.615121 | 0.640647 |
| 100% | Surface + Jackson global | 0.113472 | 0.110719 | 0.715366 | 0.737692 |
| 100% | Jackson vegetation | 0.118529 | 0.115493 | 0.635066 | 0.660800 |
| 100% | Surface + Jackson vegetation | 0.113059 | 0.110245 | 0.733661 | 0.755022 |
| 50% | Exponential h=10 cm | 0.121755 | 0.120428 | 0.596373 | 0.575276 |
| 50% | Jackson global | 0.118985 | 0.117491 | 0.687697 | 0.703769 |
| 50% | Surface + Jackson global | 0.115447 | 0.114213 | 0.753111 | 0.757860 |
| 50% | Jackson vegetation | 0.118480 | 0.116836 | 0.704155 | 0.719553 |
| 50% | Surface + Jackson vegetation | 0.115376 | 0.114090 | 0.756884 | 0.758736 |

The minimum-RMSE scenario remains **100% NPP, surface + vegetation-dependent
Jackson**. The maximum-KGE scenario remains **50% NPP, surface + vegetation-dependent
Jackson**. Some intermediate rankings change: within the 100%-NPP scenarios,
h=10 has the highest KGE under stock ratios, whereas surface + vegetation-dependent
Jackson has the highest KGE under endpoint means. Better agreement with a different
observation construction is not evidence that the model predictions improved.

The earlier h-search's **historical 50-profile cohort and fixed split** still select
**h=10 cm** by validation RMSE under either definition. Its 120 validation pairs
change from RMSE **0.126144** to **0.121658**. Its 120 test pairs change from RMSE
**0.083834** to **0.092378** and KGE **0.786183** to **0.759141**. This uses the
historical radiocarbon targets and is not a new search on today's 101-profile cohort.
Thus switching labels can improve pooled RMSE while worsening the earlier test score.

### Effect on the original bulk observation

To isolate the formula difference, we also kept the original routine's carbon
weights, default stock filtering, and profile grouping. The 101 stock-bearing
profiles become **88 location/label-duration groups**. Replacing just the layer
endpoint means with stock ratios changes their bulk observations by **0.007230**
on average in absolute value (0.72 percentage points), with maximum **0.060810**
(6.08 percentage points). The endpoint reconstruction matches the original
`process_balesdent_data` output exactly.

These 88 groups are not the same units as the 101 independently fitted profiles.
The cached bulk analysis additionally contains sites backfilled for missing
stocks, for which this raw-stock-ratio comparison is unavailable. Bulk weighting,
missing-data behavior, and grouping must therefore be held fixed when comparing
observation definitions.

## Conclusion and reproducibility

For a model predicting the fraction of the carbon **stock** that is new, the
ratio of layer new-carbon and total-carbon stocks directly matches that quantity.
The endpoint average is an approximation based on point-depth fractions. The
sampled-interval and zero-stock boundary examples above support retaining the
stock-ratio construction for the layered analysis, rather than changing it solely
to reproduce the original bulk approximation. This is a modeling judgment based
on the workbook evidence, not a claim that every interpolated stock is exact.

**No production calculation, fitted parameter, saved fit, or observation table
was changed for this audit.**

- [Three-panel comparison](../../../results/layered_fnew_definition_audit/fnew_definition_comparison.png)
- [All scenario scores](../../../results/layered_fnew_definition_audit/scenario_scores.csv)
- [Every layer observation](../../../results/layered_fnew_definition_audit/layer_observations.csv)
- [Differences by depth](../../../results/layered_fnew_definition_audit/differences_by_depth.csv)
- [Historical h-search scores](../../../results/layered_fnew_definition_audit/historical_h_scores.csv)
- [Source hashes and summary](../../../results/layered_fnew_definition_audit/summary.json)

Reproduce from the repository root with existing local data/results:

```sh
uv run python results/layered_fnew_definition_audit/reproduce.py
```

The script asserts the original-observation reconstruction, identical evaluation
cohorts, saved/current fit agreement, and exact bulk preprocessing parity. It
re-scores saved predictions and never calls the fitting optimizer.

## Follow-up: the original bulk lognormal model

The user requested a direct bulk observation `Cnew_0-100 / Ctotal_0-100`.
This follow-up evaluates the **original bulk lognormal predictions**, not the
layered model and not a stock-weighted aggregation of layer predictions.

The literal `0-100` columns provide valid ratios for **70 raw profiles**, grouped
into **62 coordinate/label-duration records** by the original arithmetic-mean
rule. Compute the ratio for each raw profile before averaging these ratios;
this preserves the original treatment of replicate profiles rather than replacing
it with a ratio of pooled stocks. The alternative using the authors' separate
`Cnew_0-100estim / Ctotal_0-100estim` columns covers **101 profiles / 88 records**.
The 11 SoilGrids-backfilled records have no corresponding new-carbon stock in the
workbook and cannot enter either stock-ratio comparison.

For each cohort, the original endpoint method was recomputed on **exactly the same
raw profiles**. All bulk predictions, carbon/radiocarbon calibration inputs,
and parameters were held fixed:

| Observation coverage | Bulk records | Original RMSE | Stock-ratio RMSE | Original KGE (2012) | Stock-ratio KGE (2012) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Literal `0-100` columns | 62 | 0.110783 | 0.114196 | 0.725378 | 0.716062 |
| Including authors' `0-100estim` columns | 88 | 0.103716 | 0.103041 | 0.723732 | 0.712238 |

The literal-stock definition makes agreement slightly worse: RMSE rises by
**0.003414 (3.1%)**, while KGE falls by **0.009316**. Including the authors' 1 m
stock estimates gives nearly unchanged RMSE (0.000676 lower) and KGE 0.011494
lower. Mean absolute changes in the bulk observations are 0.007900 and 0.016506,
respectively. The estimate-based ratio is not identical to the preceding audit's
weighted ratio over available layers: it uses the authors' entire estimated 1 m
stocks, including unobserved depth extensions.

### Why the matched baseline matters

The original saved 99-record bulk result is RMSE **0.106745**, KGE **0.827698**.
It includes observations lacking a stock-ratio alternative. Comparing that KGE
directly with the 62- or 88-record ratio scores would confound the observation
formula with changes in the evaluated population.

There is one further grouping detail. In two of the 62 literal-stock groups,
only some constituent profiles have both 1 m stocks. The original saved
observations still include their other profiles, whereas the new ratio cannot.
Those mixed groups are `(10.1666666667, -83.5666666667, 25 years)` and
`(26.7466666667, 115.0702777778, 19 years)`.

- Using the **unmodified saved observations** for those same 62 bulk rows gives
  RMSE **0.108943**, KGE **0.732583**; the ratio scores remain 0.114196 and 0.716062.
- The primary table above removes this within-group composition change by
  recomputing endpoint observations only for the 70 profiles contributing ratios.
- Excluding the two mixed groups entirely leaves 60 records: RMSE changes from
  **0.109885 to 0.113832**, and KGE from **0.735951 to 0.723615**. The direction of
  the result is therefore unchanged.

The original model fits turnover and radiocarbon, not `f_new`. Changing only the
bulk evaluation observation does not require refitting. We independently
reconstructed all 99 saved predictions from the stored lognormal C(t) curves,
turnover, and labeling durations using the original interpolation: the maximum
absolute discrepancy is **1.11e-16**. No production code, source data, calibration
inputs, or existing model result was overwritten.

Artifacts:

- [Matched scatter plots](../../../results/bulk_lognormal_fnew_ratio/observed_vs_predicted.png)
- [All metrics, including saved-baseline and unmixed-group checks](../../../results/bulk_lognormal_fnew_ratio/metrics.csv)
- [Literal 0–100 comparison rows](../../../results/bulk_lognormal_fnew_ratio/direct_0_100_comparison.csv)
- [Authors' 1 m estimate comparison rows](../../../results/bulk_lognormal_fnew_ratio/authors_0_100_estimates_comparison.csv)
- [Raw-profile ratios](../../../results/bulk_lognormal_fnew_ratio/profile_observations.csv)
- [Audit summary and source fingerprints](../../../results/bulk_lognormal_fnew_ratio/summary.json)

Reproduce from the repository root:

```sh
uv run python results/bulk_lognormal_fnew_ratio/reproduce.py
```
