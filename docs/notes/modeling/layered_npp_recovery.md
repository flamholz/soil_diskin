# Recovering existing NPP values and extending the h = 10 cm analysis

Correcting NPP coordinate matching adds **17 profiles at eight locations and 158
layers** to the previous partial-profile analysis. Holding h = 10 cm fixed, the
expanded result contains **87 profiles at 57 locations and 800 layers**, with
f_new **RMSE 0.115305 and KGE (2012) 0.699147**.

## Correction to the earlier missing-data diagnosis

All 21 profiles previously labelled as missing NPP already had positive values
in `results/all_sites_14C_turnover.csv`. The cache covers all 74 locations in the
112-profile workbook. Coordinates in the workbook and cache differed by up to
1.42e-14 degrees, causing the exact-coordinate join to miss 21 profiles at 11
locations. This was a data-matching bug, not a lack of NPP data.

The original analysis, [`02_get_turnover_14C.py`](../../../notebooks/02_get_turnover_14C.py),
uses nearest-neighbour filling for **Shi radiocarbon**, at line 44. Its NPP comes
from MODIS/061/MOD17A3HGF, averaged and sampled at 10 km scale at lines 57–79.
It does not extrapolate these cached NPP values across Balesdent sites.

The loader now rounds only the NPP join coordinates to ten decimal places,
preserves the original coordinates for Shi sampling, and rejects conflicting
NPP values at the same normalized coordinates. This recovers the original cache
values without extrapolation, new spatial assumptions, or a new NPP product.
Source fingerprints and the matching precision are recorded in `run.json`;
`npp_imputed` is false.

## Fixed-h results

| Cohort | Profiles | Locations | Layers | RMSE | KGE (2012) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous partial-profile cohort | 70 | 49 | 642 | 0.109559 | 0.749440 |
| Newly included through NPP recovery | 17 | 8 | 158 | 0.136183 | 0.465169 |
| All usable profiles | 87 | 57 | 800 | 0.115305 | 0.699147 |

The original 642 NPP inputs, fitted mu/sigma pairs, and f_new predictions are
exactly unchanged. The newly included profiles have larger prediction errors;
the pooled score changes because coverage expands. Every layer is equally
weighted, so partial profiles contribute fewer observations than full profiles.
These are descriptive scores, not a new holdout test. h remains the value
selected using the earlier 50-profile study; it was not retuned here.

All 800 fits converged and passed the numerical accuracy check. There are 74
layers at parameter bounds (58 previously, 16 newly added). Pooled stock relative
RMSE is 12.23%, and radiocarbon RMSE is 10.49 per mil. Solver convergence should
not be interpreted as exact agreement with these calibration targets.

## Coverage and remaining exclusions

Current coverage is **61 complete profiles, 26 partial profiles, and 25 excluded
profiles**. Complete profiles alone span 42 locations and 610 layers. Seventy
layers are unavailable within the partial profiles, reconciling all 1,120
possible profile/layer combinations: 800 fitted + 70 omitted + 25 × 10 excluded.

Recovered NPP permits fitting 17 of the affected 21 profiles. The remaining four
still lack other calibration data:

- Osher Pasture 2500 mm and Osher Cane 2500 mm: no native-cell Shi radiocarbon.
- Schwartz MAYOMBE 7 and Schwartz NIARI 8: no usable layer stocks.

There are now **zero exclusions due to missing NPP**. Among the 25 remaining
excluded profiles, 11 lack usable stocks alone, seven lack radiocarbon alone,
and seven have both problems. Shi targets and stock gap handling are unchanged.

## Reproduce and inspect

Prepare the depth-resolved h=10 table first using the [run guide](layered_lognormal_usage.md).

```sh
uv run python -m soil_diskin.layered_workflow \
  --input-table results/all_sites_14C_turnover_depth_h10.csv --allow-partial \
  --max-nfev 1000 --output-dir results/my_npp_recovered_run
```

Completed output: `results/layered_no_transport_h10_npp_recovered/`.

- `layers.csv`, `fits.csv`, `predictions.csv`: the complete fitting output.
- `fnew_scatter.png/.pdf`, `metrics.csv`: results for all 800 layer observations.
- `npp_recovered/fnew_scatter.png/.pdf`: newly included profiles alone.
- `cohort_metrics.csv`: previous, recovered, and pooled metrics.
- `npp_matching_audit.csv`: all 112 profiles, original/cache coordinates, NPP,
  numerical discrepancies, recovery flags, coverage, and exclusion reasons.
- `profile_coverage.csv`, `exclusions.csv`: coverage and excluded layer indices.
- `run.json`, `recovery_provenance.json`: settings and source fingerprints.
- `summarize.py`: reproducible audit and cohort summary, run with `PYTHONPATH=.`
  from the repository root.

The audit independently compared coordinate differences without rounding,
verified a unique cache match for all 112 profiles, checked that each fitted NPP
equals its cached value divided by 1,000, and confirmed the original predictions
are exactly unchanged. The regression test covers floating-point mismatch,
genuinely missing NPP, a nearby distinct site, and conflicting normalized keys.

Verification: 22 focused tests, Ruff, and Mypy passed. The full suite completed
with 106 passed, five skipped, and the existing Wolfram integration failure
because Mathematica is not activated. Standards and specification reviews found
no material issues.
