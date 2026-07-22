# JSBACH $F_{new}$: steady-state normalization and a per-site indexing bug

Companion to [`notebooks/04_JSBACH_model_predictions.py`](../../../notebooks/04_JSBACH_model_predictions.py)
and the `JSBACH` class in
[`soil_diskin/compartmental_models.py`](../../../soil_diskin/compartmental_models.py).
Tests live in [`tests/test_JSBACH_model.py`](../../../tests/test_JSBACH_model.py).

This note records two independent changes to the JSBACH $F_{new}$ predictions:
a genuine bug fix, and the same steady-state normalization applied to CLM4.5
(see [`clm45_fnew_sasu_normalization.md`](clm45_fnew_sasu_normalization.md)).
It also documents an approximation cost that is materially larger for JSBACH
than for CLM4.5.

---

## 1. The model

JSBACH's soil module is linear, with 18 pools and no vertical transport:

$$
\frac{dX}{dt} = I(t)\,u + \bigl(A\,K(t)\bigr)X ,
$$

where $I(t)$ is scalar NPP for the month, $u$ the 18-vector allocation over
litter pools, $A$ the transfer matrix, and $K(t)$ the diagonal decomposition
matrix. The annual-mean operator and input are therefore

$$
M = A\bar{K},
\qquad
\bar{I} = \overline{I(t)}\;u .
$$

---

## 2. Bug: every site received site 1's curve

The old script computed per-site $F_{new}$ with

```python
interp1d(JSBACH_predictions.columns, JSBACH_predictions.iloc[1])(
    site_data.iloc[i]['Duration_labeling'])
```

`.iloc[1]` is hardcoded — it should have been `.iloc[i]`. Every one of the 99
sites was evaluated on **site 1's** age CDF, merely sampled at its own labeling
duration. The symptom is visible in the output: the old `JSBACH_fnew.csv` held
only **49 unique values across 99 sites**; the corrected file has 93.

This means the JSBACH panel in previously generated fig3/fig4 was wrong. The
same bug is present in
[`notebooks/archive/04_model_predictions.py`](../../../notebooks/archive/04_model_predictions.py)
(line 285), which is archived and was left alone.

---

## 3. Normalization change

As with CLM4.5, the old script obtained the CDF from a tracer run normalized by
the tracer's own total at `tmax`. It now uses the analytical steady-state age
CDF,

$$
F_{new}(a) = 1 - \mathbf{1}^{\mathsf T}e^{Ma}\eta,
\qquad
\eta = \frac{X_{ss}}{\mathbf{1}^{\mathsf T}X_{ss}},
\qquad
X_{ss} = -M^{-1}\bar I ,
$$

evaluated by `calc_age_dist_cdf` directly at each site's `Duration_labeling`.

---

## 4. The annual-mean approximation is much costlier here than for CLM4.5

JSBACH's slowest mode has an e-folding time of only **167 yr**, so $F_{new}$
saturates at 1.0 well before the longest labeling durations; 4 of the 99 sites
now sit above 0.999. Ages beyond ~1000 yr carry no information for this model.

More importantly, replacing the monthly forcing with its annual mean costs far
more for JSBACH than for CLM4.5. Measured at the same grid cell (Balesdent site
19), analytical vs. a tracer driven by the model's own `_dX`:

| age (yr) | 5 | 10 | 25 | 50 | 100 | 200 |
| --- | --- | --- | --- | --- | --- | --- |
| \|difference\| | 2.3e-3 | 4.2e-3 | 9.2e-3 | 1.28e-2 | 1.31e-2 | **1.49e-2** |

For CLM4.5 at the same cell the corresponding figure is ~4e-5, roughly **350x
smaller**.

The error is systematic and grows with age rather than being phase noise. The
reason is Jensen's inequality: $X_{ss} = -(A\bar K)^{-1}\bar I$ averages $K$
*before* inverting, which is not the same as averaging the inverse. JSBACH's
climate modifier

$$
k_{clim} = e^{\,b_1 T + b_2 T^2}\bigl(1 - e^{\gamma P}\bigr)
$$

swings strongly over the seasonal cycle, so the two differ appreciably.
CLM4.5's environmental scalars are milder relative to its much slower pools.

This is a real cost of the analytical method for JSBACH and should be kept in
mind when interpreting its predictions.

---

## 5. Results

| | KGE | RMSE |
| --- | --- | --- |
| original (with the `.iloc[1]` bug) | 0.277 | 0.2261 |
| corrected + SASU normalization | **0.258** | **0.2299** |

Predictions change by a mean of +0.0113 (median +0.0164, max 0.190). Runtime is
about 11 s.

**Skill goes down, and that is expected.** The old number was flattered by a
bug that collapsed 99 sites onto a single curve; a near-constant predictor can
score better by accident than a correct one. The new value is what JSBACH
actually predicts per site.

---

## 6. Validation

`TestJSBACHFnewAccuracy` (5 tests, ~20 s) mirrors the CLM4.5 structure and
separates the two error sources:

| ground truth | agreement | what it isolates |
| --- | --- | --- |
| tracer on the **same annual-mean operator** | ~1e-11 | the analytical solution itself |
| tracer on the model's own **`_dX`** | 2.3e-3 – 1.5e-2 | the annual-mean approximation |

Tolerances are set from those measurements, not by taste, and the measured
values are recorded in the test docstrings.

Additional guards:

- `test_fnew_is_informative_over_the_tested_ages` — ages are capped at 200 yr
  and $F_{new}$ must not be pinned at 0 or 1, so the comparison cannot pass
  vacuously through saturation. 90 of the 99 sites have durations in this
  informative range.
- Total carbon during labeling is asserted against $\mathbf{1}^{\mathsf T}X_{ss}$
  (rtol 2e-2). Under monthly forcing the system rides a periodic orbit: total
  carbon circles $X_{ss}$ with a measured amplitude of 0.30% and the totals
  spread by up to 1.2% across sampled phases; the worst per-pool amplitude is
  3.8%.

The class skips cleanly if the JSBACH forcing files are absent, since
`data/model_params/JSBACH/*.nc` are downloaded rather than committed.

Note that the two pre-existing failures in `tests/test_JSBACH_model.py`
(`test_initialization`, `test_comparison_with_fortran`) are unrelated: their
`setUp` shells out to `tests/test_data/jsbach/testing_src/run_yasso_test.sh`,
and that fixture directory does not exist in the repo.

---

## 7. Pipeline notes

`results/04_model_predictions/JSBACH.csv` (the full CDF grid) is no longer
written. Nothing reads it — the figures use `JSBACH_fnew.csv` — but the
Snakefile still lists it as an output of the JSBACH rule and will fail under
Snakemake until that line is removed. `interp1d` is now an unused import.

Because both JSBACH and CLM4.5 predictions changed, fig3 and fig4 need
regenerating.
