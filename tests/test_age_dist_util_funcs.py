import numpy as np
import pandas as pd
from soil_diskin import age_dist_utils as util

def test_age_dist():
    # Test the validity of the age distribution function by comparing the results of a specific model in 
    # Sierra et al. 2018. Specifically, we look at the RothC results.
    A = np.array([[-10, 0, 0, 0],
                [0, -0.3, 0, 0],
                [1.02, 0.03, -0.59, 0],
                [1.2, 0.04, 0.08, -0.02]
                ])
    ksRC = np.array([10, 0.3, 0.66, 0.02])
    FYMsplit = np.array([0.49, 0.49, 0.02])
    DR=1.44; In=1.7; FYM=0; clay=23.4
    x = 1.67 * (1.85 + 1.60 * np.exp(-0.0786 * clay))
    B = 0.46 / (x + 1) # Proportion that goes to the BIO pool
    H = 0.54 / (x + 1) # Proportion that goes to the HUM pool

    ai3 = B * ksRC
    ai4 = H * ksRC

    ARC = np.diag(-ksRC)
    ARC[2,:] = ARC[2,:] + ai3
    ARC[3,:] = ARC[3,:] + ai4

    RcI=np.array([In * (DR / (DR + 1)) + (FYM * FYMsplit[0]), In * (1 / (DR + 1)) + (FYM * FYMsplit[1]), 0, (FYM * FYMsplit[2])])

    ages = np.arange(1,1001)
    pA = util.box_model_ss_age_dist(ARC,RcI,ages)

    sierra = pd.read_csv('tests/test_data/sierra_2018_RothC.csv')
    np.testing.assert_almost_equal(pA.flatten(), sierra['age_pdf'].values)


# --- _expm_bilinear fast path -------------------------------------------------
# box_model_ss_age_dist and calc_age_dist_cdf evaluate z @ expm(A a) @ v via an
# eigendecomposition instead of one dense expm per age. These tests pin that
# fast path to the direct expm reference it replaced.

import scipy as sp
from soil_diskin.age_dist_utils import _expm_bilinear


def _reference_bilinear(A, z, v, ages):
    """The original implementation: one dense matrix exponential per age."""
    return np.array([np.asarray(z @ sp.linalg.expm(A * a) @ v).item() for a in ages])


def _rothc_operator():
    """A small, well-conditioned compartmental operator and its input vector."""
    ksRC = np.array([10, 0.3, 0.66, 0.02])
    clay = 23.4
    x = 1.67 * (1.85 + 1.60 * np.exp(-0.0786 * clay))
    B, H = 0.46 / (x + 1), 0.54 / (x + 1)
    A = np.diag(-ksRC)
    A[2, :] = A[2, :] + B * ksRC
    A[3, :] = A[3, :] + H * ksRC
    u = np.array([1.7 * (1.44 / 2.44), 1.7 * (1 / 2.44), 0.0, 0.0])
    return A, u


def test_expm_bilinear_matches_direct_expm():
    """The eigendecomposition path reproduces the dense-expm reference."""
    A, u = _rothc_operator()
    d = A.shape[0]
    ages = np.logspace(-1, 3, 200)
    eta = np.linalg.solve(-A, u)
    eta = eta / eta.sum()

    for z in (np.ones((1, d)), -1 * np.ones((1, d)) @ A):
        fast = _expm_bilinear(A, z, eta, ages)
        ref = _reference_bilinear(A, z.reshape(-1), eta, ages)
        np.testing.assert_allclose(fast.flatten(), ref, rtol=1e-9, atol=1e-12)


def test_expm_bilinear_preserves_output_shape():
    """Callers relied on one (1,)-shaped entry per age, i.e. (n_ages, 1)."""
    A, u = _rothc_operator()
    ages = np.arange(1, 11)
    eta = np.linalg.solve(-A, u)
    eta = eta / eta.sum()

    assert _expm_bilinear(A, np.ones((1, A.shape[0])), eta, ages).shape == (10, 1)
    assert util.box_model_ss_age_dist(A, u, ages).shape == (10, 1)
    assert util.calc_age_dist_cdf(A, u, ages).shape == (10, 1)


def test_calc_age_dist_cdf_is_a_cdf():
    """CDF starts near 0, increases monotonically, and saturates at 1."""
    A, u = _rothc_operator()
    ages = np.logspace(-2, 4, 500)
    cdf = util.calc_age_dist_cdf(A, u, ages).flatten()

    assert cdf[0] < 1e-2
    assert np.all(np.diff(cdf) >= -1e-12)
    np.testing.assert_allclose(cdf[-1], 1.0, atol=1e-8)


def test_expm_bilinear_falls_back_when_not_diagonalizable():
    """A defective (non-diagonalizable) matrix must still give the right answer.

    A Jordan block has one eigenvector for its repeated eigenvalue, so
    expm(A a) carries polynomial-in-a terms a diagonalization cannot represent.
    v = e3 is chosen to excite those terms: without the fallback the
    eigendecomposition path is wrong by ~1.9 absolute here, so this test fails
    loudly if the guard ever stops firing.
    """
    from soil_diskin.age_dist_utils import _COND_LIMIT

    A = np.array([[-0.5, 1.0, 0.0],
                  [0.0, -0.5, 1.0],
                  [0.0, 0.0, -0.5]])
    z = np.ones((1, 3))
    v = np.array([0.0, 0.0, 1.0])
    ages = np.linspace(0.1, 20, 50)

    # The guard must actually trip, otherwise this test proves nothing.
    _, P = np.linalg.eig(A)
    assert np.linalg.cond(P) > _COND_LIMIT

    got = _expm_bilinear(A, z, v, ages)
    ref = _reference_bilinear(A, z.reshape(-1), v, ages)
    np.testing.assert_allclose(got.flatten(), ref, rtol=1e-8, atol=1e-12)
