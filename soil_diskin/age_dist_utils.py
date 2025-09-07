import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp

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
    xss = (-1 * np.linalg.inv(A)) @ u
    eta = xss/xss.sum()
    age_pdf = np.array([zT @ sp.linalg.expm(A * a) @ eta for a in ages])
    
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
def calc_age_dist_cdf(A,u,ages):
    d = A.shape[0]
    one = np.ones((d,1))
    beta = u/u.sum()
    zT = one.T
    xss = (-1 * np.linalg.inv(A)) @ u
    X = np.diag(xss)
    eta = xss/xss.sum()
    age_dens = 1 - np.array([zT @ sp.linalg.expm(A * a) @ eta for a in ages])
    return age_dens
