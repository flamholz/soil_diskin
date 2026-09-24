# Fits rerun with the original radiocarbon extraction

All five input-depth allocations were refitted with the corrected Shi targets,
for both full and half NPP entering the soil. Every alternative uses the same
**101 profiles, 65 locations, and 914 layers**. All **9,140 primary layer fits**
converged and passed the finer-grid integration check.

The vegetation-specific surface/Jackson mixture has the lowest RMSE with full
NPP and the highest KGE with half NPP. The precision correction has negligible
effects on the original 800 layers; the visible changes in pooled scores come
from adding 114 previously excluded layers.

## Results on all 914 layers

| Input-depth allocation | RMSE, full NPP | KGE, full NPP | RMSE, half NPP | KGE, half NPP |
| --- | ---: | ---: | ---: | ---: |
| Exponential h = 10 cm | 0.117198 | 0.735200 | 0.121755 | 0.596373 |
| Jackson global | 0.119231 | 0.615121 | 0.118985 | 0.687697 |
| Jackson vegetation | 0.118529 | 0.635066 | 0.118480 | 0.704155 |
| Surface + Jackson global | 0.113472 | 0.715366 | 0.115447 | 0.753111 |
| Surface + Jackson vegetation | **0.113059** | 0.733661 | **0.115376** | **0.756884** |

Surface mixtures put 50% of **soil input** directly into 0–10 cm and distribute
the remaining 50% with Jackson over 0–100 cm, including the top layer. Full NPP
means all site NPP enters soil. Half NPP means total modeled input is 0.5 × NPP;
in its surface mixture, 25% of original NPP enters directly at the surface and
25% follows Jackson. Depth allocations never renormalize to available layers.

The full-NPP vegetation mixture improves RMSE over h = 10 cm but has slightly
lower KGE (0.7337 versus 0.7352). The half-NPP vegetation mixture has the highest
KGE among these ten alternatives, at the cost of higher RMSE than its full-NPP
counterpart. These are descriptive pooled comparisons, with equal layer weights,
not an independent test or a new hyperparameter search.

## What changed on the original observations?

Matching original profile/layer identities isolates the effect of the precision
correction from the expanded cohort. Across all ten comparisons on those same
800 layers:

- Maximum absolute change in fitted mu: **3.19e-7**.
- Maximum absolute change in fitted sigma: **7.10e-7**.
- Maximum absolute change in predicted f_new: **2.16e-8**.
- Absolute RMSE changes are below **1.1e-10**; absolute KGE changes are below **1.4e-9**.

Original stocks, site NPP, allocated layer inputs, labeling durations, observed
f_new, and coordinates are exactly unchanged. Coefficients, vegetation mappings,
optimizer settings, and model equations were held fixed. All newly retained
layers are flagged as using spatially filled radiocarbon.

For the full-NPP vegetation mixture, the earlier 800-layer result was RMSE
0.109402 and KGE 0.711995. The refit on those same 800 layers has the same displayed
scores. Including the additional observations changes the pooled result to RMSE
0.113059 and KGE 0.733661; this is a cohort change rather than a substantial change
in predictions at the original sites.

## Calibration diagnostics

All eight Jackson alternatives have zero layers at parameter bounds and match
stocks and radiocarbon to numerical precision: stock relative RMSE below 1e-10
percent and radiocarbon RMSE below 1e-10 per mil. Those are calibration targets,
so this is not independent predictive validation.

The h = 10 cm alternatives still have calibration limitations:

| Soil NPP fraction | Layers at bounds | Stock relative RMSE (%) | Radiocarbon RMSE (‰) |
| --- | ---: | ---: | ---: |
| 1.0 | 85 | 12.2848 | 10.5912 |
| 0.5 | 139 | 20.9259 | 14.0236 |

## Reproduce and inspect

```sh
uv run python -m notebooks.compare_jackson_inputs --surface-fraction 0.5 \
  --soil-npp-fraction 1 --max-nfev 1000 --output-dir results/my_refit/npp100
uv run python -m notebooks.compare_jackson_inputs --surface-fraction 0.5 \
  --soil-npp-fraction 0.5 --max-nfev 1000 --output-dir results/my_refit/npp50
```

Completed outputs: `results/layered_radiocarbon_refit/`.

- `npp100/comparison.png/.pdf`, `npp50/comparison.png/.pdf`: all observed-versus-predicted plots.
- `npp100/`, `npp50/`: complete fitting outputs, assignments, input-depth plots,
  and protocols with exact source fingerprints for the workspace code used.
- `metrics.csv`: all ten pooled scores and calibration diagnostics.
- `cohort_metrics.csv`: previous 800, refitted original 800, added 114, and all 914
  layer scores for every alternative.
- `changes_on_original_layers.csv`: parameter, prediction, and score changes
  restricted to the original observations.
- `summary.json`: comparison checks and fingerprints of the old and new tables.
- `summarize.py`: reproduces the cohort comparisons from saved layer tables;
  run from the repository root with `PYTHONPATH=.`.

Both comparison figures were visually checked. All scenarios have identical
profile/layer identities, and the saved source and assignment hashes match the
code and inputs used. The existing model code was rerun without changes in this task.

### Standards review

No findings. Commands, outputs, source fingerprints, and comparison checks are
consistent with the saved runs.

### Spec review

No findings. Independent calculations reproduced all RMSE/KGE values, verified
the common cohort and input fractions, and confirmed the negligible changes on
the original layers. Review totals: Standards 0; Spec 0.
