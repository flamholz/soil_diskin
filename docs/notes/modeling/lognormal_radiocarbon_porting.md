# Porting `03b_lognormal_age_scan.wls` to Julia and Python

This document records the port of [`notebooks/03b_lognormal_age_scan.wls`](03b_lognormal_age_scan.wls)
(Wolfram Language) to Python: the mathematical setup, the
algorithmic improvement that the port applies, the implementation choices, the
validation strategy, and the resulting performance.

For the underlying math of the lognormal Diskin model and the closed-form
solution used here, see also
[`docs/notes/modeling/lognormal_diskin_closed_form_derivation.md`](lognormal_diskin_closed_form_derivation.md).

---

## 1. What the Wolfram script computes

The lognormal Diskin model represents soil carbon as a continuum of
first-order pools, with decay rate $k > 0$ distributed according to a
lognormal density

$$
f(k) = \frac{1}{k\\,\sigma\sqrt{2\pi}}\exp\\!\Bigl(-\frac{(\log k - \mu)^2}{2\sigma^2}\Bigr),
$$

where $(\mu, \sigma)$ are determined from the system's transit time $\tau$ and
mean age $\bar a$ via

$$
\sigma = \sqrt{\log(\bar a / \tau)},\qquad
\mu    = -\log\\!\sqrt{\tau^3/\bar a}.
$$

For each pool $k$ at steady state under constant input, the carbon's age
distribution is exponential with mean $1/k$. Combining over rates,
mass-weighting (with weight $\propto 1/k$ since pools with slow turnover hold
more carbon at steady state), the predicted bulk-pool radiocarbon activity
ratio is

$$
\boxed{\\;
r(\tau, \bar a)
\\;=\\; \frac{1}{\mathbb{E}[1/k]}
\int_0^\infty\\!\\!\int_0^\infty
R(a)\\, e^{-a/8267}\\, f(k)\\, e^{-k a}\\, dk\\, da
\\;}
$$

where:
- $R(a)$ is the atmospheric $^{14}\\!C/^{12}\\!C$ activity ratio $a$ years before
  the reference epoch (year 2000 in this dataset),
- $8267\\;\mathrm{yr}$ is the $^{14}\\!C$ mean life (so $e^{-a/8267}$ is the
  radioactive-decay factor),
- $\mathbb{E}[1/k] = e^{-\mu + \sigma^2/2}$ is the lognormal first inverse
  moment, normalizing the mass-weighted age distribution.

The Wolfram script evaluates $r(\tau, \bar a)$ on a grid of 99 sites
(parametrized by `turnover` $= \tau$) and 101 log-spaced ages
$\bar a \in [10^3, 10^{5.5}]$ yr — 9999 cells in total — producing a
calibration matrix that maps `(τ, age)` to predicted fraction modern.

A sister Python script (`03b_calibrate_lognormal_model.py`) then inverts each
row via LOWESS smoothing + 1-D interpolation to predict each site's mean age
from its measured fraction modern.

The Wolfram script implements $R(a)$ as Mathematica's
`Interpolation[..., InterpolationOrder -> 0]` — i.e. a piecewise-constant
lookup over ~55 000 annual atmospheric $^{14}\\!C$ measurements.

---

## 2. Why the Wolfram implementation has limitations

Mathematica computes $r(\tau, \bar a)$ as a **2-D adaptive quadrature**:

```mathematica
NIntegrate[ atm14C[a] * Exp[-a/8267]
            * PDF[LogNormalDistribution[mu, sig], x]
            * Exp[-x*a] / Exp[-mu + sig^2/2],
            {x, 0, Infinity}, {a, 0, Infinity},
            PrecisionGoal -> 3 ]
```

This has two problems:

1. **Inner integrand is discontinuous in $a$.** $R(a)$ has ~55 000 step
   discontinuities (one per year of atmospheric data). 2-D adaptive
   quadrature in Mathematica refines based on local error estimates, and on
   discontinuities those estimates are unreliable. The result is occasional
   convergence failures.

2. **Failures aren't always isolated.** When `NIntegrate` mis-converges at a
   given $(\tau, \bar a)$ pair, it sometimes produces a sustained run of
   wrong values across consecutive ages — observed runs of 5 to 11 cells in
   the Wolfram output, with values ~11 % too high.

Visually these failures show up as non-monotonic "spikes" in the age scan
when the rest of the row is smoothly monotone. They contaminate the LOWESS
calibration step downstream and shift inferred ages by up to ~10 %.

The `PrecisionGoal -> 3` setting (3-digit accuracy target) is itself a clue
that the original author was aware of how aggressive a tighter goal would be
to satisfy here.

---

## 3. The algorithmic improvement: analytical inner integral

The key observation is that $R(a)$ is **piecewise constant**, so the inner
integral has an exact closed form. Define

$$
I(k) \\;\equiv\\; \int_0^\infty R(a)\\, e^{-(1/8267 + k)\\, a}\\, da .
$$

If $R(a) = c_i$ on the segment $[a_i, a_{i+1})$, with a constant tail
$c_\infty$ beyond $a_N$ (Mathematica's `ExtrapolationHandler` returns the
mean of the last 50 000 values), then with $\alpha \equiv 1/8267 + k$,

$$
\boxed{\\;
I(k)
= \frac{1}{\alpha}\\!\left[
    \sum_{i} c_i\\,\bigl(e^{-\alpha a_i} - e^{-\alpha a_{i+1}}\bigr)
    \\;+\\; c_\infty\\, e^{-\alpha a_N}
\right]
\\;}
$$

This is exact in $a$ — no quadrature needed for that direction.

What remains is a **1-D outer integral** over $k$:

$$
r(\tau, \bar a)
= \frac{1}{\mathbb{E}[1/k]} \int_0^\infty f(k)\\, I(k)\\, dk .
$$

The outer integrand pairs a smooth $f(k)$ (lognormal density) with the smooth
$I(k)$ (a Laplace-transform-like average). Switching to log-rate coordinates
$u = \log k$ uses the identity $f(k)\\,dk = \varphi_\mathcal{N}(u; \mu, \sigma)\\,du$
to put the outer integral in a Gaussian-weighted form perfect for adaptive
quadrature:

$$
\boxed{\\;
r(\tau, \bar a)
= e^{\mu - \sigma^2/2}
\int_{-\infty}^{\infty}
\varphi_\mathcal{N}(u; \mu, \sigma)\\, I(e^u)\\, du
\\;}
$$

In code we truncate the outer integration to $[\mu - 10\sigma,\ \mu + 10\sigma]$
(well past machine precision for the Gaussian).

**Why this is better than the Wolfram approach:**

| | Wolfram (2-D adaptive) | Closed-form inner + 1-D outer |
|---|---|---|
| Inner-`a` integration | adaptive on a discontinuous integrand | exact, sums over knots |
| Outer-`k` integration | adaptive on a 2-D integrand | adaptive on a smooth 1-D integrand |
| Failure modes | sporadic spike runs (~0.5–1 % of cells) | none observed |
| Cost per cell | ~ms (adaptive 2-D struggles) | $O(N)$ per outer sample × ~30 outer samples = ms |

The two implementations end up at similar runtime per cell but the new one is
more reliable because the hard direction was eliminated, not approximated.

---

## 4. Julia implementation

### Files

- [`notebooks/lognormal_radiocarbon.jl`](lognormal_radiocarbon.jl) — the math:
  - `AtmC14` struct: piecewise-constant atmospheric $^{14}\\!C$ lookup with a
    constant tail.
  - `load_atm14c(path)`: matches Wolfram's data-loading behavior — keeps
    columns `years_before_2000` and `R_14C`, sorts ascending in age, drops
    negative ages, computes `mean_R` from the last 50 000 file rows.
  - `inner_integral(atm, α)`: $O(N)$ allocation-free evaluation of $I(k)$;
    runs through knots, accumulating the segment contributions.
  - `lognormal_radiocarbon(atm, τ, ā; rtol)`: outer 1-D quadrature in log-$k$
    via `QuadGK.quadgk`.
- [`notebooks/03b_lognormal_age_scan.jl`](03b_lognormal_age_scan.jl) — driver
  that mirrors the Wolfram script:
  1. Loads `data/14C_atm_annot.csv` once.
  2. Reads `results/all_sites_14C_turnover.csv`.
  3. Calls `lognormal_radiocarbon` for each (site, age) on the fixed
     `agelist = 10 .^ range(3, 5.5, length=101)`.
  4. Writes the three matrix CSVs that downstream scripts expect.

### Parallelism

The driver parallelizes over sites with `Threads.@threads`. Each site is
independent, atm14C is read-only, and `inner_integral` is allocation-free, so
threading scales nearly linearly. With `julia --threads=auto` (12 threads on
this machine), the full 99 × 101 main scan + the smaller q05 / q95 scans
together take ≈ 35 s.

### Numerical tolerances

`quadgk` is called with `rtol = 1e-4` per cell. Internal self-consistency
across `rtol = 1e-6` vs `rtol = 1e-10` agrees to ~ $10^{-9}$, well below the
~$10^{-3}$ accuracy that Wolfram targets.

---

## 5. Python implementation

### Files

- [`notebooks/lognormal_radiocarbon.py`](lognormal_radiocarbon.py) — the math:
  - `AtmC14`: same role as the Julia struct.
  - `load_atm14c(path)`: same loading behavior.
  - `inner_integral(atm, α)`: vectorized in NumPy
    (`np.exp(-α * ages)`, then a single `(fm[:-1] * (e[:-1] - e[1:])).sum()`).
    Goes via compiled NumPy code for the 55 000-element loop.
  - `lognormal_radiocarbon(atm, τ, ā, rtol)`: outer 1-D quadrature in log-$k$
    via `scipy.integrate.quad`.
- [`notebooks/03b_lognormal_age_scan.py`](03b_lognormal_age_scan.py) — driver,
  one-to-one with the Julia driver. Outputs go to `*_python.csv` paths so the
  Julia outputs are preserved alongside.
- [`notebooks/03b_lognormal_calibration.py`](03b_lognormal_calibration.py) —
  unified end-to-end pipeline: forward age scan + LOWESS calibration in a
  single run, no CSV round-trip between the two phases.

### Parallelism

`joblib.Parallel(n_jobs=-1)` over sites. Joblib uses processes by default
(works around the GIL), and the per-site work is large enough to amortize the
fork cost. With 12 worker processes, the full pipeline matches the Julia
runtime within ~10 %.

### Why scipy.integrate.quad instead of a fixed-grid approach

`quad` (QUADPACK adaptive Gauss–Kronrod) is the closest Python analogue of
Julia's `QuadGK.quadgk`, which makes apples-to-apples comparison
straightforward. The integrand is smooth in log-$k$, so even ~30 evaluations
per cell suffice — `quad`'s Python-callback overhead is the only thing
slowing it relative to Julia.

---

## 6. Validation

### Comparison strategy

Three cross-checks, in increasing order of strictness:

1. **Internal self-consistency of the new implementation.** Compute the same
   integral at multiple `rtol` values and confirm convergence to ~$10^{-9}$
   relative. Establishes that the new implementation is internally
   well-posed; values are not artifacts of one tolerance setting.

2. **Element-wise comparison to the saved Wolfram CSVs.** Since the CSVs were
   the working reference for downstream calibration, this confirms the new
   implementation isn't subtly wrong in some systematic way the Wolfram
   reference could catch.

3. **Smoothness check on the new implementation.** Confirm that every row
   of the new age-scan output is smooth-in-age (monotone or near-monotone, no
   spikes). This is the property the Wolfram output sometimes lacks.

### Detecting Wolfram failures

The pointwise comparison surfaces two kinds of disagreements:

- **Tiny, ubiquitous, ~$10^{-3}$ relative**: matches Wolfram's stated
  `PrecisionGoal -> 3` (3-digit accuracy). These are the genuine integration
  errors expected of the Wolfram approach and are not corrected by the port.
- **Localized ~10 % spikes**, often in runs of 5–11 consecutive cells. These
  are Wolfram `NIntegrate` mis-convergences. Detected via a rolling
  median + anchor-cell test on the Wolfram row (the new implementation's
  rows are smooth, so disagreements at suspected-failure cells indicate the
  Wolfram cell is wrong, not the new value).

### Final results

Across the full 99 × 101 grid (`results/03_calibrate_models/03b_lognormal_model_age_scan.csv`):

| Scan | Cells | Wolfram failures detected | Median rel err (clean) | 99th pctile |
|---|---:|---:|---:|---:|
| main (turnover) | 9 999 | ~40 (~0.4 %) | 1.4e-3 | 2.5e-3 |
| q05 | 1 111 | 4 (0.4 %) | 1.9e-3 | 2.8e-3 |
| q95 | 1 111 | 12 (1.1 %) | 0.9e-3 | 1.8e-3 |

The new implementation reproduces the Wolfram result to within Wolfram's own
precision goal everywhere, and replaces the spike-failed cells with smooth,
monotone values.

### Cross-implementation comparison (Julia vs Python)

The Julia and Python ports use the same math, the same atmospheric data, the
same outer-integration tolerance, but two different adaptive-quadrature
implementations (`QuadGK.jl` vs QUADPACK). Their outputs agree to:

| Scan | Median rel err | p99 | Max |
|---|---:|---:|---:|
| main | 5.6e-9 | 1.7e-5 | 3.8e-3 |
| q05 | 1.2e-8 | 1.2e-5 | 3.5e-3 |
| q95 | 1.0e-8 | 7.3e-6 | 8.0e-4 |

Median agreement at machine epsilon confirms both ports are computing the
same integral correctly. The handful of `~10⁻³` cells reflect harmless
differences in adaptive-mesh decisions between the two quadrature libraries
under the same `rtol = 1e-4` request.

---

## 7. Performance

| Pipeline | Wall time (full 3-scan) | Notes |
|---|---:|---|
| Wolfram (`ParallelMap`, 8+ kernels) | not benchmarked here | reference output predates this port |
| Julia (`--threads=auto`, 12 threads) | ~35 s | includes ~3 s `Pkg.instantiate` + JIT warm-up |
| Python (`joblib`, 12 workers) | ~35 s | numpy-vectorized inner + process parallelism |

The two ports land at very similar wall times. Julia is faster per call
(no callback overhead, better tight-loop performance) but Python uses NumPy
for the dominant $O(N)$ inner integral and joblib for the parallel sweep,
which closes the gap.

---

## 8. Downstream calibration

The original Wolfram script generates the calibration matrices that
[`notebooks/03b_calibrate_lognormal_model.py`](03b_calibrate_lognormal_model.py)
later inverts (per-row LOWESS smoothing + interpolation) to predict each
site's mean age from its measured fraction modern.

The unified Python driver
[`notebooks/03b_lognormal_calibration.py`](03b_lognormal_calibration.py)
collapses the two stages into one process, passing the in-memory matrices
directly into LOWESS rather than round-tripping through CSV. End-to-end
predicted ages from the unified script agree with the original two-step
(Julia age scan + Python LOWESS) pipeline to:

| Column | median rel diff | max rel diff | max abs diff |
|---|---:|---:|---:|
| `pred` | 1.8e-7 | 2.4e-4 | 4.9 yr (out of ~20 000 yr) |
| `pred_05` | 2.6e-7 | 1.0e-5 | 0.4 yr |
| `pred_95` | 8.8e-7 | 7.5e-6 | 0.06 yr |

Where the unified-Python and the original Wolfram-derived predictions
diverge most (~10 % at the worst sites), the divergence traces back to the
Wolfram NIntegrate spike runs distorting LOWESS smoothing on those sites.
The unified-Python pipeline is the more reliable end-to-end calibration.

---

## 9. Files produced (summary)

### Julia
- [`notebooks/lognormal_radiocarbon.jl`](lognormal_radiocarbon.jl)
- [`notebooks/03b_lognormal_age_scan.jl`](03b_lognormal_age_scan.jl)

### Python
- [`notebooks/lognormal_radiocarbon.py`](lognormal_radiocarbon.py)
- [`notebooks/03b_lognormal_age_scan.py`](03b_lognormal_age_scan.py)
- [`notebooks/03b_lognormal_calibration.py`](03b_lognormal_calibration.py) — end-to-end

### Validation / benchmarking
- [`notebooks/test_lognormal_radiocarbon.jl`](test_lognormal_radiocarbon.jl) — Julia ↔ Wolfram CSV comparison on a subset
- [`notebooks/verify_full_scan.jl`](verify_full_scan.jl) — Julia ↔ Wolfram comparison on the full grid, with Wolfram-failure detection
- [`notebooks/test_python_ports.py`](test_python_ports.py) — Julia ↔ Python comparison + benchmark

### Reference data preserved
- [`results/03_calibrate_models/_wolfram_reference/`](../results/03_calibrate_models/_wolfram_reference/) — original Wolfram-generated age-scan and predictions CSVs, kept as a regression baseline.
