# Matching the original radiocarbon extraction

`layered_data.py` now gives **exactly the same fraction-modern values** as
`02_get_turnover_14C.py` lines 36–67 for all **914 usable layers**. The maximum
absolute difference is **0.0**, checked against an independent reproduction of
the original raster-reading, filling, selection, and averaging steps.

## What differed

The original notebook fills spatial gaps in the Shi grid at line 44 with
`rio.interpolate_na(method='nearest')`, then selects the nearest grid cell for
each site. Line 66 reshapes the 100 one-cm values into ten 10-cm layers and takes
an ordinary arithmetic mean. These are per-layer delta-14C values, not the
stock-weighted column signal calculated separately at line 69.

The layered adapter previously selected from the unfilled grid. It therefore
excluded sites whose nearest cell was missing. Also, the original interpolation
promotes values from float32 to float64 before averaging, whereas the unfilled
adapter averaged float32 values. Even where no spatial value was missing, this
caused tiny differences: at most **0.00004883 per mil**, or about **4.88e-8 fraction
modern** across the raw workbook.

The corrected adapter uses the same rioxarray interpolation and raster ordering
(north-to-south latitude, increasing longitude). Matching that order also preserves
nearest-neighbor tie behavior. It retains original site coordinates, averages ten
one-cm values with the same precision, and converts with `fm = 1 + delta14C/1000`.
It does not interpolate across depth. A depth with no finite spatial source still
remains missing. Stocks and NPP are unchanged.

## Coverage

| Quantity | Previous unfilled targets | Matching original extraction |
| --- | ---: | ---: |
| Profiles with usable layers | 87 | 101 |
| Locations with usable layers | 57 | 65 |
| Usable layers | 800 | 914 |
| Complete profiles | 61 | 68 |
| Partial profiles | 26 | 33 |
| Excluded profiles | 25 | 11 |

Of all 1,120 possible profile/layer combinations in the 112-profile workbook,
140 lacked native-cell radiocarbon. The original filling procedure supplies all
140; 114 also have usable stocks and therefore become eligible for fitting.
Those 114 layers come from 14 newly included profiles. Remaining exclusions are
due to missing or nonpositive stocks. All 914 usable layers also have observed f_new.

Every retained layer records `radiocarbon_spatially_filled`, indicating whether
any constituent one-cm target came from spatial filling. Metadata separately
counts filled layers in the raw workbook and among retained fitting layers.

**This check updates the input adapter, not saved model fits.** Previously
reported RMSE/KGE values describe the old targets and cohort; they have not been
recomputed in this task. New runs automatically use the corrected targets.

## Reproduce

```sh
uv run python -m notebooks.check_layered_radiocarbon \
  --output-dir results/my_radiocarbon_parity
```

Completed outputs are in `results/layered_radiocarbon_parity/`:

- `radiocarbon_comparison.csv`: original-reference and unfilled targets for all
  1,120 profile/layer combinations, alongside the adapter's eligible targets and
  exact differences.
- `prepared_layers.csv`, `exclusions.csv`: corrected adapter outputs.
- `summary.json`: parity statistics, coverage, data checksums, and source hashes.

The reference reproduces only the local radiocarbon portion of the original
notebook. It needs no Earth Engine authentication, NPP download, or modification
of the existing NPP cache.

Regression tests compare a float32 synthetic raster through the two independent
ingestion paths, including a missing cell and a spatial tie. They also verify
per-depth filling, retention of truly unavailable depths, missing evaluation
data, and unchanged stock/NPP handling.

Verification: the four data-adapter tests, Ruff, and Mypy pass. The workspace's
full suite completed with **118 passed, 5 skipped**, and the existing Wolfram
integration failure because Mathematica is not activated.

### Standards review

No findings. The original interpolation operation is reused, depth separation
is preserved, and filling flags and source hashes are recorded.

### Spec review

No findings. Independent checks confirmed exact equality, matching selected
cells for the original processed coordinates, and unchanged stocks, NPP,
labeling durations, and f_new for the previous cohort.

Review totals: Standards 0; Spec 0.
