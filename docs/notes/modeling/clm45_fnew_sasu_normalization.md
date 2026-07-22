# CLM4.5 $F_{new}$: normalizing to the semi-analytical steady state

Companion to [`notebooks/04_CLM45_model_predictions.py`](../../../notebooks/04_CLM45_model_predictions.py)
and the `CLM5` class in
[`soil_diskin/compartmental_models.py`](../../../soil_diskin/compartmental_models.py).
Tests live in [`tests/test_CLM_models.py`](../../../tests/test_CLM_models.py).

This note records why the CLM4.5 $F_{new}$ predictions changed, what they were
normalized to before and after, and how the new method was validated.

---

## 1. The model

The repo's `CLM5` class is the CLM4.5bgc matrix soil-carbon model (Huang et al.
2018; Lu et al. 2020). Its state equation is linear:

$$
\frac{dX}{dt} = I(t) + \bigl(A\,K(t) - V\bigr)X ,
$$

with $X$ a 70-vector (7 pools x 10 soil levels, pool-major), $A$ the transfer
matrix, $K(t)$ the diagonal decomposition-rate matrix carrying the monthly
environmental scalars $\xi$ and depth attenuation, and $V$ vertical transport.

**Sign convention.** Liao et al. (2023) write the dynamics as
$I + (A\xi K + V)C$, with $V$ carrying a negative diagonal. This repo's
`make_V_matrix` returns $V$ with a **positive** diagonal and the model
subtracts it. The two are negatives of one another, $V_{repo} = -V_{paper}$,
and produce the identical dissipative operator. All expressions below use the
repo's convention.

---

## 2. What changed

### Before

`predict_fnew` ran a tracer from bare soil and normalized by the tracer's own
total at the end of the simulation:

```python
age_CDF = labeled.y.sum(axis=0) / labeled.y.sum(axis=0)[-1]
```

This assumes the tracer has fully replaced the standing soil stock by `tmax`.
It has not. The slowest eigenmode of the CLM4.5 operator has an e-folding time
of **18,000–38,000 yr** across the 99 Balesdent sites, implying:

| convergence target | years of native-dynamics spin-up |
| --- | --- |
| within 1% | 83,000 – 175,000 |
| within 0.1% | 124,000 – 262,000 |

At `tmax = 100,000` the tracer is still a few percent short, so the denominator
is too small and $F_{new}$ is inflated.

### After

Following the Semi-Analytical Spin-Up of Liao et al. (2023, Eq. 7) and Xia et
al. (2012), average the time-dependent operators over the forcing loop and set
$dX/dt = 0$:

$$
X_{ss} \,=\, -\bigl(A\bar{K} - V\bigr)^{-1}\bar{I} .
$$

Because the model is linear ($K$ does not depend on $X$; there is no online
nitrogen feedback), this single 70x70 solve is the **exact** steady state of the
annual-mean system. No AD phase or post-AD native run is needed — those are
required in the paper only for the nonlinear C–N coupling.

For that same linear, time-invariant mean system the tracer has a closed form,
so no integration is needed at all:

$$
X(t) = \bigl(\mathrm{Id} - e^{Mt}\bigr)X_{ss},
\qquad
F_{new}(t) = 1 - \mathbf{1}^{\mathsf T}e^{Mt}\eta,
\qquad
M = A\bar{K} - V .
$$

This is the Sierra et al. (2018) steady-state age CDF, i.e. exactly
`soil_diskin.age_dist_utils.calc_age_dist_cdf`, which derives
$X_{ss} = -M^{-1}\bar I$ internally. See
[`age_dist_cdf_eigendecomposition.md`](age_dist_cdf_eigendecomposition.md).
By construction $F_{new}(\infty) = 1$.

The script now calls `calc_age_dist_cdf` **directly at each site's
`Duration_labeling`**, rather than building a 1000-point log grid and
interpolating. That removes an interpolation error of up to 5e-6, which was
systematically negative (linear interpolation undershoots this concave curve).

---

## 3. Results

| | KGE | RMSE |
| --- | --- | --- |
| original (tracer-end normalization) | 0.230 | 0.2313 |
| SASU normalization | **0.273** | **0.2201** |

$F_{new}$ falls by a mean of 0.0165 (median 0.0124, max 0.0608) — the expected
direction, since the old denominator was too small. Whole-ensemble runtime is
about 5 s.

CLM4.5 remains biased high against observations; this correction reduces that
bias but does not explain it.

**Caveat on the baseline.** The previously committed `CLM45_fnew.csv` was
byte-identical to `CLM45_fnew_10_000.csv`, i.e. generated with
`tmax = 10_000`, not the 100,000 the script specified. Since the old
normalization divided by the tracer total at `tmax`, that baseline was
inflated somewhat more than the 0.0165 shift alone suggests.

---

## 4. Validation

Ground truth is a labeled/unlabeled tracer pair in the style of
[`notebooks/clm_tracer_test.ipynb`](../../../notebooks/clm_tracer_test.ipynb):
spin up from bare soil, then at $t=0$ make all incoming carbon labeled while
the standing stock decays unlabeled, and take
$F_{new} = \text{labeled}/(\text{labeled}+\text{unlabeled})$. Spin-up is run
numerically rather than seeded with $X_{ss}$, so the ground truth stays
independent of the solution under test.

Two test classes, at Balesdent site 19 (Amazon, 4000-yr labeling duration —
long enough to exercise the slow passive SOM pool):

| class | ground truth driven by | agreement | runtime |
| --- | --- | --- | --- |
| `TestCLM45FnewAccuracy` | annual-mean operator $M$ | 1.1e-8 | **0.2 s** |
| `TestCLM45FnewAccuracyMonthlyForcing` | the model's own `_dX` | ≤1e-3 | **25 min**, opt-in |

The slow class is gated behind `RUN_SLOW_TESTS=1`; the default suite skips it.

The fast test has ~878x headroom between its measured agreement and its
tolerance, yet still fails on a 0.1% perturbation and on the old
normalization's 1.6% bias — i.e. it would have caught the bug this work fixed.

### Periodic orbits under monthly forcing

The monthly-forced system settles onto a periodic orbit rather than a fixed
point, so per-pool equality with $X_{ss}$ cannot hold tightly. Measured orbit
amplitudes: Litter1 **5.3%** (fastest pool, tracks the forcing most closely),
Litter2/3 ~1.5%, CWD 0.8%, SOM1 0.9%, SOM2 0.4%, SOM3 0.01%. Total carbon —
the quantity $F_{new}$ is normalized by — is far steadier at **6.4e-4**.

The steady-state assertion is therefore split: a tight bound on the total
(rtol 5e-3) and a loose per-pool bound (rtol 1e-1) justified by those measured
amplitudes.

---

## 5. Related changes

- **`CLM5._dX` optimization.** `self.I` became a property whose setter keeps an
  ndarray cache in sync, and the 12 monthly operators $A K_t - V$ are
  precomputed in `__init__`. `_dX` went from 116.2 µs to 3.4 µs per call
  (**33.8x**), bit-identical output, full suite green. The cache lives in a
  property setter rather than `__init__` because
  [`tests/test_CLM_models.py`](../../../tests/test_CLM_models.py) reassigns
  `model.I` after construction.

- **Month indexing.** `_dX` computes its month index as
  `int((t % (1/12)) * 144)`, which cycles all 12 months within each 1/12 of a
  year — 144 times per year rather than once. This looks like a bug but was
  left untouched. It has a convenient side effect: because the forcing then
  varies far faster than any pool turnover, the system responds to the
  annual-mean operator, which is exactly the operator used above. If it is ever
  fixed, the mean-operator equivalence is no longer guaranteed and the
  validation should be re-run.

- **Pipeline.** `results/04_model_predictions/CLM45.csv` (the full CDF grid) is
  no longer written. Nothing reads it, but the Snakefile still lists it as an
  output of the `CLM45_model_predictions` rule and will fail under Snakemake
  until that line is removed. `interp1d` is now an unused import.
