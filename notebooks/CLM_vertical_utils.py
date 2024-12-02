import numpy as np
from scipy.linalg import block_diag
import warnings

SECS_PER_DAY = 86400
DAYS_PER_YEAR = 365


def make_V_matrix(Gamma_soil, F_soil, npools, nlevels,
                  dz, dz_node, zsoi, zisoi):
    """
    Create a tridiagonal matrix for soil carbon pools

    Parameters
    ----------
    Gamma_soil : float
        Diffusion coefficient
    F_soil : float
        Advection coefficient
    npools : int
        Number of soil carbon pools
    nlevels : int
        Number of soil layers
    dz : np.array
        Thickness of soil layers (m)
    dz_node : np.array
        Distance between layer interfaces (m)
    zsoi : np.array
        Depth of soil layers (m)
    zisoi : np.array
        Depth of soil layer interfaces (m)

    Returns
    -------
    np.array
        Tridiagonal matrix
    """

    # A function from Patankar, Table 5.2, pg 95
    aaa = np.vectorize(lambda pe: np.max ([0, (1 - 0.1 * np.abs(pe))**5]))

    Gamma_vec = np.ones(nlevels+1)*Gamma_soil
    F_vec = np.ones(nlevels+1)*F_soil

    # Calculate the weighting between lfactors for the diffusion and advection terms
    w_e = np.zeros(nlevels+1)
    w_e[1:] = (zisoi[:nlevels] - zsoi[:nlevels]) / dz_node[1:nlevels+1]
    Gamma_e = np.zeros(nlevels+1)
    Gamma_e[1:] = 1 / ((1 - w_e[1:nlevels+1]) / Gamma_vec[1:nlevels+1] + w_e[1:nlevels+1] / Gamma_vec[nlevels]); # Harmonic mean of diffus

    # Calculate the weighting between lfactors for the diffusion and advection terms
    w_p = np.zeros(nlevels+1)
    w_p[:nlevels] = (zsoi[1:nlevels+1] - zisoi[:nlevels]) / dz_node[1:nlevels+1]
    Gamma_p = np.zeros(nlevels+1)
    Gamma_p[:nlevels] = 1 / ((1 - w_p[:nlevels]) / Gamma_vec[:nlevels] + w_p[:nlevels] / Gamma_vec[1:nlevels+1]); # Harmonic mean of diffus

    ## TODO - pop the above code into a separate function and compare againt the output from the matlab code

    # Define the D and F values for each layer the according to Eq. 5.9 in Patankar, pg. 82
    D_e = Gamma_e / dz_node[:nlevels+1]
    D_p = np.zeros(nlevels+1)
    D_p[:nlevels] = Gamma_p[:nlevels] / dz_node[1:nlevels+1]
    D_p[-1] = D_e[-1]
    F_e = F_vec
    F_p = F_vec
    F_p[-1] = 0


    # Define the Peclet number - ignore the warning 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Pe_e = F_e / D_e
    Pe_e[0] = 0;
    Pe_p = F_p / D_p

    # Define the vectors for the tridiagonal matrix
    a_tri_e =  -( D_e * aaa(Pe_e) + np.max([F_e, np.zeros(nlevels+1)],axis=0))
    c_tri_e =  - (D_p * aaa(Pe_p) + np.max([-F_p, np.zeros(nlevels+1)],axis=0))
    b_tri_e = - a_tri_e - c_tri_e

    # Define the upper and lower bounaries
    b_tri_e[0] = - c_tri_e[0]
    b_tri_e[-2] = - a_tri_e[-2]

    # Define the tridiagonal matrix
    tri_matrix = np.diag(a_tri_e[:-1],k=-1)[1:,1:] + np.diag(b_tri_e[:-1],k=0) + np.diag(c_tri_e[:-1],k=1)[:-1,:-1]

    # Define the block diagonal matrix
    tri_matrix = block_diag(*[tri_matrix]*npools)
    
    # Set the first pool to zero
    tri_matrix[:nlevels,:nlevels] = 0

    # Divide the matrix by dz
    tri_matrix = (tri_matrix.T/np.tile(dz[:nlevels],npools)).T

    return tri_matrix

def make_A_matrix(sand_content, nlevels):
    """
    Create a A matrix for soil carbon pools for the CENTURY model with 7 pools. 
    The order of the pools is CWD, Litter1, Litter2, Litter3, SOM1, SOM2, SOM3
    #TODO - change when we change the order of the pools

    Parameters
    ----------
    sand_content : np.array (nlevels)
        sand content
    nlevels : int
        Number of soil layers

    Returns
    -------
    np.array
        The A matrix
    """

    assert len(sand_content) == nlevels

    # The functional parameterization of the transfer coefficients is based on # this is based on the
    # original CENTURY model parameterization in Parton et al. 1998 (https://link.springer.com/article/10.1007/BF02180320) in Figure 1

    # t is a number dependent on the sand content that determines the transfer coefficient fraction of carbon that is lost to respiration
    t = 0.85 - 0.68 * 0.01 * (100 - sand_content)

    # f is the fraction of carbon from a specific pool that is transferred to another pool
    f_s1s2 = 1 - 0.004 / (1 - t)
    f_s1s3 = 0.004 / (1 - t)
    f_s2s1 = 0.42 / 0.45 * np.ones(nlevels)
    f_s2s3 = 0.03 / 0.45 * np.ones(nlevels)

    # rf is the fractio of carbon in a specific flux between pools that is lost to respiration (1-CUE)
    rf_s1s2 = t
    rf_s1s3 = t

    # Using the formalism a_i,j = (1-rf_i,j) * f_i,j, where a_i,j are the coefficients in the A matrix 
    # Implementation accroding to Eq. 3 in Huang et al. 2017 (https://onlinelibrary.wiley.com/doi/10.1111/gcb.13948)
    Adiag = -np.eye(nlevels) #A11-A77
    A_zero = np.zeros((nlevels,nlevels))
    A31 = 0.76 * np.eye(nlevels) # CWD -> Litter2
    A41 = 0.24 * np.eye(nlevels) # CWD -> Litter3
    A52 = (1-0.55) * np.eye(nlevels) # Litter1 -> SOM2
    A53 = (1-0.5) * np.eye(nlevels) # Litter2 -> SOM1
    A56 = np.diag((1-0.55) * f_s2s1) # SOM1 -> SOM3
    A57 = (1-0.55) * np.eye(nlevels) # SOM3 -> SOM1
    A64 = (1-0.5) * np.eye(nlevels) # Litter3 -> SOM2
    A65 = np.diag((1 - rf_s1s2) * f_s1s2) # SOM1 -> SOM2
    A75 = np.diag((1 - rf_s1s3) * f_s1s3) # SOM1 -> SOM3
    A76 = np.diag((1-0.55) * f_s2s3) # SOM2 -> SOM3

    A_matrix = np.block([
        [Adiag     , A_zero    , A_zero    , A_zero    , A_zero    , A_zero    , A_zero    ],
        [A_zero    , Adiag     , A_zero    , A_zero    , A_zero    , A_zero    , A_zero    ],
        [A31       , A_zero    , Adiag     , A_zero    , A_zero    , A_zero    , A_zero    ],
        [A41       , A_zero    , A_zero    , Adiag     , A_zero    , A_zero    , A_zero    ],
        [A_zero    , A52       , A53       , A_zero    , Adiag     , A56       , A57       ],
        [A_zero    , A_zero    , A_zero    , A64       , A65       , Adiag     , A_zero    ],
        [A_zero    , A_zero    , A_zero    , A_zero    , A75       , A76       , Adiag     ]
            ])

    return A_matrix
    

def make_K_matrix(taus, zsoi,
                  w_scalar, t_scalar, o_scalar, n_scalar,
                  decomp_depth_efolding, nlevels):
    """Makes the matrix of rate constants for each of the soil carbon pools.

    Parameters
    ----------
    taus : np.array
        Turnover times for each soil carbon pool in units of seconds
    zsoi : np.array
        Depth of soil layers (m)
    w_scalar : float
        Water scalar (0-1)
    t_scalar : float 
        Temperature scalar (0-1)
    o_scalar : float
        Oxygen scalar (0-1)
    n_scalar : float
        Nitrogen scalar (0-1)
    decomp_depth_efolding : float
        Depth of the decomposition efolding (units of 1/m)
    nlevels : int
        Number of soil layers -- 10 in practice

    Returns
    -------
    np.array
        The K matrix
    """
    # calculate k's from tau's
    ks = 1 / (taus)
    depth_scalar = np.exp(-zsoi[:nlevels]/decomp_depth_efolding)
    k_modifier = (t_scalar * w_scalar * o_scalar * depth_scalar)
    return block_diag(*[np.diag(k * k_modifier * n_scalar if i in [1,2,3] else k * k_modifier) for i,k in enumerate(ks)]) # only for the litter pools (pools 2,3,4) do we multiply by n_scalar





