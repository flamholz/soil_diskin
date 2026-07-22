# Fast steady-state age distributions in `age_dist_utils`

Companion to [`soil_diskin/age_dist_utils.py`](../../../soil_diskin/age_dist_utils.py).
Tests live in [`tests/test_age_dist_util_funcs.py`](../../../tests/test_age_dist_util_funcs.py).

This note records a rewrite of the two steady-state age-distribution functions,
`calc_age_dist_cdf` and `box_model_ss_age_dist`. The mathematics they implement
is unchanged; only how it is evaluated changed. The result is ~59x faster and
accepts a scalar age.

---

## 1. What the functions compute

Both describe a linear compartmental model at steady state,

$$
\frac{dX}{dt} = u + A\,X,
\qquad
X_{ss} = -A^{-1}u,
\qquad
\eta = \frac{X_{ss}}{\mathbf{1}^{\mathsf T}X_{ss}},
$$

following Sierra et al. (2018). `calc_age_dist_cdf` returns the fraction of
carbon younger than age $a$,

$$
F(a) \,=\, 1 - \mathbf{1}^{\mathsf T}\,e^{A a}\,\eta ,
$$

and `box_model_ss_age_dist` returns the corresponding density, which differs
only in the left vector ($z^{\mathsf T} = -\mathbf{1}^{\mathsf T}A$ instead of
$\mathbf{1}^{\mathsf T}$).

Note that $X_{ss} = -A^{-1}u$ is exactly the semi-analytical (SASU) steady
state. Feeding these functions an annual-mean operator and mean input therefore
yields a CDF normalized to the SASU steady state with no extra work — this is
what the CLM4.5 and JSBACH prediction scripts rely on (see
[`clm45_fnew_sasu_normalization.md`](clm45_fnew_sasu_normalization.md) and
[`jsbach_fnew_sasu_normalization.md`](jsbach_fnew_sasu_normalization.md)).

---

## 2. The inefficiency

Both functions built a full dense matrix exponential for every age:

```python
age_dens = 1 - np.array([zT @ sp.linalg.expm(A * a) @ eta for a in ages])
```

For a 70x70 operator over 1000 ages this is 1000 independent `expm` calls. But
the quantity actually needed, $z^{\mathsf T}e^{Aa}v$, is a **scalar**: the full
matrix is computed and then immediately collapsed.

---

## 3. The rewrite

A new private helper `_expm_bilinear(A, z, v, ages)` evaluates that scalar for
all ages at once. Diagonalizing $A = P\,\mathrm{diag}(\lambda)\,P^{-1}$ once,

$$
z^{\mathsf T}e^{Aa}v
\,=\, (z^{\mathsf T}P)\,\mathrm{diag}(e^{\lambda a})\,(P^{-1}v)
\,=\, \sum_j w_j e^{\lambda_j a},
\qquad
w_j = (z^{\mathsf T}P)_j\,(P^{-1}v)_j .
$$

The weights $w_j$ do not depend on age, so all ages reduce to one
`(n_ages x d)` elementwise exponential and a row sum. No per-age matrix work
remains.

Both public functions now delegate to it, and both switched from
`inv(A) @ u` to `np.linalg.solve(-A, u)` — same result, faster and better
conditioned since it never forms an explicit inverse. Two dead locals in
`calc_age_dist_cdf` (`beta`, `X`) were removed.

### Robustness fallback

Diagonalization is invalid for defective matrices, so the helper falls back to
the original `scipy.linalg.expm` loop when any of the following holds:

- the eigenvector basis is ill-conditioned, `cond(P) > _COND_LIMIT` (1e10);
- the result is not finite;
- the bilinear form does not come out real;
- `numpy` raises `LinAlgError`.

For the CLM4.5 operator this never triggers: `cond(P)` measured 10–141 across a
sample of 11 sites, with eigenvalues real to ~1e-16. There are repeated
eigenvalues (60 distinct of 70), but the small `cond(P)` confirms the operator
is diagonalizable rather than defective.

### Scalar ages

`ages` is now normalized with `np.atleast_1d`. Previously a scalar became a 0-d
array, which cannot be `len()`'d or iterated, raising
`TypeError: len() of unsized object`. A scalar now behaves like a length-1
sequence. This lets callers evaluate the CDF directly at one age instead of
building a grid and interpolating.

### Output shape

Unchanged: one entry per age, shaped as `z @ M @ v` would be, i.e. `(n_ages, 1)`
for the standard call. Callers therefore need no changes. (Returning
`(n_ages,)` would be tidier but is a breaking change, so it was left alone.)

---

## 4. Performance

Measured on the CLM4.5 operator (70x70) over 1000 log-spaced ages:

| implementation | time | vs. reference |
| --- | --- | --- |
| original `expm` loop | 0.146 s | 1.0x |
| eigendecomposition + vectorized | 0.003 s | **59x** |
| `scipy.sparse.linalg.expm_multiply` loop | did not finish in 120 s | rejected |

Agreement with the original: **1.2e-13**.

`expm_multiply` was tested and rejected: it computes the action of the
exponential on a vector without forming the matrix, but for ages up to 1e5 the
matrix $Aa$ has an enormous norm, so its scaling/Krylov machinery needs far too
many steps.

**Caveat.** The eigendecomposition is a fixed $O(d^3)$ cost amortized over
ages, so the speedup shrinks as the number of ages falls; at roughly 20 ages or
fewer the two implementations are comparable.

---

## 5. Tests

Five tests in
[`tests/test_age_dist_util_funcs.py`](../../../tests/test_age_dist_util_funcs.py):

| test | pins |
| --- | --- |
| `test_age_dist` (pre-existing) | `box_model_ss_age_dist` against published Sierra et al. 2018 RothC values |
| `test_expm_bilinear_matches_direct_expm` | fast path vs dense-`expm` reference, both $z$ forms, rtol 1e-9 |
| `test_expm_bilinear_preserves_output_shape` | `(n_ages, 1)` for the helper and both public functions |
| `test_calc_age_dist_cdf_is_a_cdf` | starts near 0, monotonic, saturates at 1 |
| `test_expm_bilinear_falls_back_when_not_diagonalizable` | Jordan block routes to the fallback |

The fallback test deserves a note. Its first version used $v = e_1$ and passed
**without the fallback doing anything** — that vector aligns with the Jordan
block's single eigenvector, so defectiveness never surfaced. It now uses
$v = e_3$, where the eigendecomposition path is wrong by ~1.9 absolute, and
additionally asserts `cond(P) > _COND_LIMIT` so the test fails loudly if the
guard ever stops firing rather than passing vacuously.
