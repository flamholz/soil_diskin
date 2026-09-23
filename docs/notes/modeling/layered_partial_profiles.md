# Extending h = 10 cm to partially observed profiles

The no-transport model can fit available layers independently. Holding the
previously selected **h = 10 cm** fixed adds **20 profiles and 142 layer pairs**,
for **70 profiles at 49 locations and 642 evaluated layers** in total.

## Data and fitting rule

- Every retained layer needs a positive carbon stock, a finite native-cell Shi
  radiocarbon target, and positive cached site NPP. Coordinates must be valid.
- Missing stock or radiocarbon excludes that layer. A profile with no usable
  layers or no site NPP is excluded altogether. No values are imputed.
- Stocks are differences of adjacent cumulative values. Missing boundaries
  invalidate both adjacent differences, rather than bridging the missing depth.
- Keep the ten original 10 cm intervals. The exponential inputs remain normalized
  over 0–100 cm, with no redistribution toward observed layers.
- Fit each layer's mu/sigma using stock and radiocarbon only, with the same
  residual scales, bounds, three starts, and 1,000-evaluation limit as the h search.
- Missing f_new or labeling duration does not prevent calibration, but prevents
  scoring that layer. Here all 642 fitted layers also have evaluation observations.
- h was not retuned on the added data. No parameters are borrowed from other
  layers or profiles. Excluded layers remain unpredicted.

## Results

Scores give equal weight to each evaluated layer. Partial profiles therefore
contribute fewer observations than complete profiles.

| Cohort | Profiles | Locations | Layers | RMSE | KGE (2012) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original complete profiles | 50 | 35 | 500 | 0.111592 | 0.712603 |
| Added partial profiles | 20 | 15 | 142 | 0.102077 | 0.724068 |
| All usable profiles | 70 | 49 | 642 | 0.109559 | 0.749440 |

One added-profile location is shared with the original cohort, so the extension
adds 14 distinct locations. The original 500 fitted mu/sigma pairs and f_new
predictions are **exactly unchanged**. Changes in pooled performance reflect
the expanded cohort, not a model change on the original observations.

All 642 fits converged and passed the finer-grid numerical check. Calibration
still has the h = 10 trade-off:

| Cohort | Layers at parameter bounds | Stock relative RMSE | Radiocarbon RMSE |
| --- | ---: | ---: | ---: |
| Original complete profiles | 56/500 | 13.45% | 11.58 per mil |
| Added partial profiles | 2/142 | 1.20% | 2.06 per mil |
| All usable profiles | 58/642 | 11.88% | 10.26 per mil |

These are descriptive evaluation scores, not a new independent test. The pooled
cohort includes the validation observations used to select h. The added profiles
have incomplete depth coverage and one shared location, and should not be
interpreted as an equivalent random sample of complete profiles.

## Remaining missing data

Of the 112 input profiles, 50 are complete, 20 are partially usable (five to nine
layers each), and **42 cannot supply any complete calibration layer** using the
current inputs. In the 20 partial profiles, another **58 layers** are excluded
because their stocks are missing or nonpositive.

Among the 42 wholly excluded profiles, 21 lack positive cached NPP, 14 have
incomplete native-cell radiocarbon, and 24 have incomplete/nonpositive stocks.
These reasons overlap. `profile_coverage.csv` lists all 112 profiles and the
whole-profile reasons; `exclusions.csv` also identifies excluded depth indices.

Predicting the remaining layers would require additional data or an explicitly
specified pooling/imputation model: two local parameters cannot be uniquely
estimated from a single calibration target with the current independent fits.

## Reproduce and inspect

```sh
uv run python -m soil_diskin.layered_workflow --input-depth 10 --allow-partial \
  --max-nfev 1000 --output-dir results/my_partial_profiles
```

Completed output: `results/layered_no_transport_h10_partial/`.

- `layers.csv`, `fits.csv`, `predictions.csv`: inputs, local fits, and predictions.
- `fnew_scatter.png/.pdf`, `metrics.csv`: pooled observed-versus-predicted results.
- `added_partial/fnew_scatter.png/.pdf`: the newly included profiles alone.
- `cohort_metrics.csv`: original, added, and pooled scores plus calibration errors.
- `profile_coverage.csv`, `exclusions.csv`: complete coverage accounting.
- `run.json`, `extension_provenance.json`: settings, sources, and comparison checks.
- `summarize.py`: reproduce cohort summaries with `PYTHONPATH=. .venv/bin/python
  results/layered_no_transport_h10_partial/summarize.py` from the repository root.

Regression checks cover adjacent cumulative-stock gaps, missing native-cell
radiocarbon, missing NPP, missing evaluation data, unchanged layer depths, and
unchanged inputs/predictions when other layers are removed.

All 21 focused tests, Ruff, and Mypy checks passed. The full suite finished with
105 passed, five skipped, and one existing Wolfram integration failure because
Mathematica is not activated.
