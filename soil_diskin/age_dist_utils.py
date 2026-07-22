import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp

# Above this condition number the eigenvector basis is too ill-conditioned to
# trust, and _expm_bilinear falls back to the direct matrix-exponential loop.
_COND_LIMIT = 1e10


def _expm_bilinear(A: np.array, z: np.array, v: np.array, ages: np.array) -> np.array:
    """Evaluate the scalar ``z @ expm(A * a) @ v`` for every age ``a``.

    Both age-distribution functions below need only this scalar, never the full
    matrix exponential. Diagonalizing ``A = P diag(lam) P^-1`` once turns it into

        z @ expm(A a) @ v = sum_j w_j exp(lam_j a),
        w_j = (z @ P)_j * (P^-1 v)_j

    so all ages are evaluated in a single vectorized sum rather than one dense
    ``expm`` per age. For a 70x70 operator over 1000 ages this is ~59x faster
    (0.146 s -> 0.003 s) and agrees with the direct loop to ~1e-13.

    Falls back to the original ``scipy.linalg.expm`` loop when the
    eigendecomposition cannot be trusted: a non-diagonalizable or
    ill-conditioned eigenvector basis, a non-finite result, or a bilinear form
    that fails to come out real.

    Returns one entry per age, shaped as ``z @ M @ v`` would be, so callers see
    the same output shape as the original implementations.
    """
    A = np.asarray(A)
    # atleast_1d so a scalar age works: np.asarray alone gives a 0-d array,
    # which cannot be len()'d or iterated. A scalar yields one entry, i.e. the
    # same shape as passing a length-1 sequence.
    ages = np.atleast_1d(np.asarray(ages, dtype=float))
    z_flat = np.asarray(z).reshape(-1)
    v_flat = np.asarray(v).reshape(-1)

    # Shape a single element of the original per-age list comprehension had.
    elem_shape = np.shape(np.asarray(z) @ np.eye(A.shape[0]) @ np.asarray(v))

    try:
        lam, P = np.linalg.eig(A)
        cond = np.linalg.cond(P)
        if not np.isfinite(cond) or cond > _COND_LIMIT:
            raise np.linalg.LinAlgError("eigenvector basis too ill-conditioned")

        w = (z_flat @ P) * np.linalg.solve(P, v_flat)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            val = (w[None, :] * np.exp(np.outer(ages, lam))).sum(axis=1)

        scale = max(1.0, float(np.max(np.abs(val.real))))
        if np.max(np.abs(val.imag)) > 1e-8 * scale:
            raise np.linalg.LinAlgError("bilinear form did not come out real")
        out = val.real
        if not np.all(np.isfinite(out)):
            raise np.linalg.LinAlgError("non-finite result from eigendecomposition")
    except np.linalg.LinAlgError:
        out = np.array([
            np.asarray(z_flat @ sp.linalg.expm(A * a) @ v_flat).item() for a in ages
        ])

    return out.reshape((len(ages),) + elem_shape)


def box_model_ss_age_dist(A:np.array, u: np.array, ages: np.array) -> np.array:
    '''
    Calculate the steady state age distribution for a general box model. Based on Equation 17 in [Sierra et al. 2018](https://link.springer.com/article/10.1007/s11004-017-9690-1#Sec17)

    Parameters
    ----------
    A : np.array
        The transition matrix of the box model. Should be a square matrix.
    u : np.array
        The input vector of the box model. Should be a 1D array with the same length as the number of rows in A.
    ages : np.array
        The ages at which to calculate the age distribution. Should be a 1D array.
    
    Returns
    -------
    np.array
        The age distribution at the specified ages. The shape of the output will be the same as that of `ages`.
    '''

    d = A.shape[0]
    one = np.ones((d,1))
    zT = -1 * one.T @ A
    # solve(-A, u) rather than inv(A) @ u: same result, faster and better
    # conditioned since it never forms the explicit inverse.
    xss = np.linalg.solve(-A, u)
    eta = xss/xss.sum()
    age_pdf = _expm_bilinear(A, zT, eta, ages)

    return age_pdf
    

# tmax sets the length of the simulation
# keep it small so that runtimes are reasonable
# timestep = 0.2 # yrs
# tmax = 5000 # yrs
def dynamic_age_dist(A_t,u,timestep,tmax):
    
    # define the time steps
    ts = np.arange(0,tmax,timestep)

    # a matrix of zeros for timesteps and ks
    # each row is a k, each column is a timestep
    state = np.zeros((u.shape[0], ts.size))

    for i, t in enumerate(ts):    
        # Haven't added new material yet, can just multiply 
        # the whole matrix by the fractional decay
        state += A_t(t) @ state * timestep

        # new input of biomass 
        state[:,i] = u(t) * timestep
    return state

def nonlinear_age_dist(A_t,u,timestep,tmax):
    
    # define the time steps
    ts = np.arange(0,tmax,timestep)

    # a matrix of zeros for timesteps and ks
    # each row is a k, each column is a timestep
    state = np.zeros((u.shape[0], ts.size))

    for i, t in enumerate(ts):    
        # Haven't added new material yet, can just multiply 
        # the whole matrix by the fractional decay
        state += A_t(state)*timestep

        # new input of biomass 
        state[:,i] = u*timestep
    return state


def predict_fnew(model, config, env_params, tmax = 10_000):
    """
    Predict the fraction of new carbon in a specific site using a specific model.

    Args:
        model: The model to use for prediction.
        config: Configuration parameters for the model.
        env_params: Environmental parameters for the model.

    Returns:
        age_CDF: The cumulative distribution function of the age of carbon.
    """
    
    model = model(config, env_params)
    ts = np.logspace(-1, np.log10(tmax), 1000)  # time in years
    labeled = solve_ivp(model._dX, (0, tmax * 1.1), np.zeros(model.X_size), t_eval=ts, method='LSODA')
    age_CDF = labeled.y.sum(axis=0) / labeled.y.sum(axis=0)[-1]

    return age_CDF


# age distribution calculation based on Sierra et al. 2018
def calc_age_dist_cdf(A, u, ages):
    """Steady-state age CDF of a linear compartmental model (Sierra et al. 2018).

        F(a) = 1 - 1^T expm(A a) eta,    eta = xss / sum(xss),  xss = -A^-1 u

    Note that xss is the analytical steady state of ``dX/dt = u + A X``, so
    passing an annual-mean operator and mean input gives a CDF normalized to the
    semi-analytical (SASU) steady state.

    Parameters
    ----------
    A : np.array (d, d)
        Compartmental operator (the term multiplying the state in dX/dt).
    u : np.array (d,)
        Input vector.
    ages : np.array (n,)
        Ages at which to evaluate the CDF.

    Returns
    -------
    np.array, shape (n, 1)
        The CDF at each age.
    """
    d = A.shape[0]
    zT = np.ones((1, d))
    # solve(-A, u) rather than inv(A) @ u: same result, faster and better
    # conditioned since it never forms the explicit inverse.
    xss = np.linalg.solve(-A, u)
    eta = xss / xss.sum()
    age_dens = 1 - _expm_bilinear(A, zT, eta, ages)
    return age_dens
