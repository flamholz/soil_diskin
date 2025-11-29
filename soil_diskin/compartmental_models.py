import warnings
import numpy as np
import xarray as xr
import jax.numpy as jnp
import jax

from collections import namedtuple
from jax import numpy as jnp
from jax import scipy as jsp
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from soil_diskin.constants import SECS_PER_DAY, DAYS_PER_YEAR
from soil_diskin.age_dist_utils import box_model_ss_age_dist, dynamic_age_dist
from soil_diskin.age_dist_utils import calc_age_dist_cdf


# TODO: remove JULES, CABLE, CLM5_jax models as they are not used.
# TODO: add fuller testing of CLM5, JSBACH, and ReducedComplexModel.
# TODO: remove commented-out and dead code.
# TODO: the configuration of these models is very obtuse, as was made 
# clear by trying to write simple tests. This should be improved.


# Data structures
# Global configuration parameters
ConfigParams = namedtuple('CP', ['decomp_depth_efolding', 'taus', 'Gamma_soil', 'F_soil',
                                 'zsoi', 'zisoi', 'dz', 'dz_node', 'nlevels','npools'])

# Data that are specific to a location
# Q: in the documents below this is called LocalData - is this the same thing?
LocDependentData = namedtuple('LDD', ['w', 't', 'o', 'n', 'sand', 'I','X0'])


# Q: is this used only by CLM5?
class GlobalData:
    """Class for storing global data, factory for creating LocDependentData instances."""

    def __init__(self, global_da):
        """Construct an instance of the global dataset.

        Puts the data into a usable format for the model.
        """
        self.global_da = global_da

        # unpack the environmental variables
        self.sand_da = global_da['CELLSAND'][0] # 0th axis is time - the value is constant in time so we are taking the first value
        self.w_scalar_da = global_da['W_SCALAR']
        self.t_scalar_da = global_da['T_SCALAR']
        self.o_scalar_da = global_da['O_SCALAR']
        self.n_scalar_da = global_da['FPI_VR']

        # upack the initial values
        CWD = global_da['CWDC_VR']
        LITR1 = global_da['LITR1C_VR']
        LITR2 = global_da['LITR2C_VR']
        LITR3 = global_da['LITR3C_VR']
        SOIL1 = global_da['SOIL1C_VR']
        SOIL2 = global_da['SOIL2C_VR']
        SOIL3 = global_da['SOIL3C_VR']
        X = xr.concat([CWD, LITR1, LITR2, LITR3, SOIL1, SOIL2, SOIL3], dim='pools')
        self.X0 = X[:,0,:,:].stack(pooldepth=['pools','LEVDCMP1_10'])
        
        # unpack the inputs
        CWD_input = global_da['TOTC2CWDC_VR']
        LITR1_input = global_da['TOTC2LITRMETC_VR']
        LITR2_input = global_da['TOTC2LITRCELC_VR']
        LITR3_input = global_da['TOTC2LITRLIGC_VR']
        zero_da = xr.zeros_like(LITR1_input)
        inputs = xr.concat([CWD_input, LITR1_input, LITR2_input, LITR3_input, zero_da, zero_da, zero_da], dim='pools')
        self.inputs = inputs.stack(pooldepth=['pools','LEVDCMP1_10'])

        # remove negative inputs
        self.inputs = xr.where(self.inputs>=1e-36,self.inputs,0)

    def make_ldd(self, lat, lon):
        test_point = self.w_scalar_da.sel(y=lat, x=lon, method='nearest')
        if test_point.notnull().all() == False:
            w = self.w_scalar_da.coarsen(x=4, y=4).mean()
            t = self.t_scalar_da.coarsen(x=4, y=4).mean()
            o = self.o_scalar_da.coarsen(x=4, y=4).mean()
            n = self.n_scalar_da.coarsen(x=4, y=4).mean()
            sand = self.sand_da.coarsen(x=4, y=4).mean()
            I = self.inputs.coarsen(x=4, y=4).mean()
            X0 = self.X0.coarsen(x=4, y=4).mean()
            w = w.sel(y=lat, x=lon, method='nearest')
            t = t.sel(y=lat, x=lon, method='nearest')
            o = o.sel(y=lat, x=lon, method='nearest')
            n = n.sel(y=lat, x=lon, method='nearest')
            sand = sand.sel(y=lat, x=lon, method='nearest')
            I = I.sel(y=lat, x=lon, method='nearest')
            X0 = X0.sel(y=lat, x=lon, method='nearest')
        else:
            w = self.w_scalar_da.sel(y=lat, x=lon, method='nearest')
            t = self.t_scalar_da.sel(y=lat, x=lon, method='nearest')
            o = self.o_scalar_da.sel(y=lat, x=lon, method='nearest')
            n = self.n_scalar_da.sel(y=lat, x=lon, method='nearest')
            sand = self.sand_da.sel(y=lat, x=lon, method='nearest')
            I = self.inputs.sel(y=lat, x=lon, method='nearest')
            X0 = self.X0.sel(y=lat, x=lon, method='nearest')

        
        return LocDependentData(w=w, t=t, o=o, n=n, sand=sand, I=I, X0=X0)

    def make_ldd_jax(self, lat, lon):
        w = jnp.array(self.w_scalar_da.sel(y=lat, x=lon, method='nearest').values)
        t = jnp.array(self.t_scalar_da.sel(y=lat, x=lon, method='nearest').values)
        o = jnp.array(self.o_scalar_da.sel(y=lat, x=lon, method='nearest').values)
        n = jnp.array(self.n_scalar_da.sel(y=lat, x=lon, method='nearest').values)
        sand = jnp.array(self.sand_da.sel(y=lat, x=lon, method='nearest').values)
        I = jnp.array(self.inputs.sel(y=lat, x=lon, method='nearest').values)
        X0 = jnp.array(self.X0.sel(y=lat, x=lon, method='nearest').values)
        return LocDependentData(w=w, t=t, o=o, n=n, sand=sand, I=I, X0=X0)


class CompartmentalModel:
    """An abstract base class for compartmental models."""
    
    def __init__(self, config: namedtuple, env_params: namedtuple):
        """
        Args:
        config: ConfigParams
            Configuration parameters for the model.
        env_params: LocalData
            Local data object containing environmental variables at a specific location.
        """
        self.config = config
        if env_params is not None:
            self.env = env_params

    def run(self, timesteps, dt):
        """Run the model for a series of timesteps."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class CLM5(CompartmentalModel):
    """A model based on the CLM5 with vertical diffusion and advection."""
    
    def __init__(self, config: namedtuple, env_params: namedtuple):
        """Construct an instance for a single grid cell.

        Args:
            config: instance of ConfigParams.
                Contains all the global configuration parameters.
            env_params: instance of LocDependentData.
                Contains all the data that are specific to a location.
        """
        super().__init__(config, env_params)

        self.I = env_params.I * SECS_PER_DAY * DAYS_PER_YEAR  # Convert from g/m^2/s to g/m^2/year
        self.A = self.make_A_matrix(env_params.sand, config.nlevels)
        self.V = self.make_V_matrix(config.Gamma_soil, config.F_soil,
                               config.npools, config.nlevels, config.dz, 
                               config.dz_node, config.zsoi, config.zisoi)
        
        self.K_ts = np.stack(
            [self.make_K_matrix(
                self.config.taus, 
                np.array(self.config.zsoi),
                self.env.w[t,:], 
                self.env.t[t,:], 
                self.env.o[t,:], 
                self.env.n[t,:],
                self.config.decomp_depth_efolding,
                self.config.nlevels) for t in range(12)
            ]
        )

        self.X_size = self.I.shape[1]
    
    @classmethod
    def make_V_matrix(self, Gamma_soil, F_soil, npools, nlevels,
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

        Gamma_vec = np.ones(nlevels+1) * Gamma_soil
        F_vec = np.ones(nlevels+1) * F_soil

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


    @classmethod
    def make_A_matrix(self, sand_content, nlevels):
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

    @classmethod
    def make_K_matrix(self, taus, zsoi,
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

    def _dX(self, t, X, tres = 'Y'):
        """ODE to be integrated to calculate C pools via CLM style CENTURY type model. 

        K_t is time dependent, needs to be calculated at each step.
        Other matrices are constant.
        
        dX = I_t - (AK_t - V) * X

        Args:
            X: state matrix of C pools
            t: time
            tres: time resolution ('M' for monthly, 'Y' for yearly)

        Returns:
            dX: change in state matrix of C pools
        """

        assert tres in ['M','Y'], "tres must be 'M' or 'Y'"
        # print('t:', t)
        if tres == 'Y':
            t_ind = int((t % (1 / 12)) * 12 * 12) # t is in units of years, and the dt for the I, T and P is in months, so we multiply by 12 to get the index
        else:
            t_ind = int(t % 12) # t is in units of months
        I_t = self.I[t_ind,:].values
        
        taus = self.config.taus
        zsoi = self.config.zsoi
        w_scalar = self.env.w
        t_scalar = self.env.t
        o_scalar = self.env.o
        n_scalar = self.env.n
        decomp_depth_efolding = self.config.decomp_depth_efolding
        nlevels = self.config.nlevels

        # K_t = self.make_K_matrix(taus, zsoi,
        #                     w_scalar[t_ind,:], t_scalar[t_ind,:], o_scalar[t_ind,:], n_scalar[t_ind,:],
        #                     decomp_depth_efolding, nlevels)

        
        # dX = I_t + (self.A @ K_t - self.V) @ X#.values
        dX = X.copy()
        dX.data = I_t + (self.A @ self.K_ts[t_ind, :, :] - self.V) @ X.values

        return dX
    
    def run(self, timesteps, dt, tres = 'Y'):
        """
        Run the model for a series of timesteps.

        Args:
            timesteps: list of timesteps to run the model for
            dt: timestep size
            tres: time resolution ('M' for monthly, 'Y' for yearly)
        """
        assert tres in ['M','Y'], "tres must be 'M' or 'Y'"

        Xs = [self.env.X0]
        for i,ts in enumerate(timesteps):
            # print("Running timestep ", i,ts)
            # print(self._CLM_vertical(Xs[-1], ts))
            # print("ts:", ts)
            Xs.append(Xs[-1] + self._dX(ts, Xs[-1], tres) * dt)
        
        return xr.concat(Xs, dim = 'TIME')


class CLM5_jax(CompartmentalModel):
    """A model based on the CLM5 with vertical diffusion and advection."""
    
    def __init__(self, config: namedtuple, env_params: namedtuple):
        """Construct an instance for a single grid cell.

        Args:
            config: instance of ConfigParams.
                Contains all the global configuration parameters.
            env_params: instance of LocDependentData.
                Contains all the data that are specific to a location.
        """
        self.config = config
        self.env = env_params

        self.I = env_params.I * SECS_PER_DAY * DAYS_PER_YEAR  # Convert from g/m^2/s to g/m^2/year
        self.A = self.make_A_matrix(env_params.sand, config.nlevels)
        self.V = self.make_V_matrix(config.Gamma_soil, config.F_soil,
                               config.npools, config.nlevels, config.dz, 
                               config.dz_node, config.zsoi, config.zisoi)
        
        # Create the K matrix for each month
        self.K_ts = jnp.stack(
            [self.make_K_matrix(
                self.config.taus, 
                jnp.array(self.config.zsoi),
                self.env.w[t,:], 
                self.env.t[t,:], 
                self.env.o[t,:], 
                self.env.n[t,:],
                self.config.decomp_depth_efolding,
                self.config.nlevels) for t in range(12)
            ]
        )
                        

    
    def make_V_matrix(self,Gamma_soil, F_soil, npools, nlevels,
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

        Gamma_vec = np.ones(nlevels+1) * Gamma_soil
        F_vec = np.ones(nlevels+1) * F_soil

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

        return jnp.array(tri_matrix)

    def make_A_matrix(self, sand_content, nlevels):
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
        f_s2s1 = 0.42 / 0.45 * jnp.ones(nlevels)
        f_s2s3 = 0.03 / 0.45 * jnp.ones(nlevels)

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

        return jnp.array(A_matrix)

    def make_K_matrix(self, taus, zsoi,
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
        depth_scalar = jnp.exp(-zsoi[:nlevels]/decomp_depth_efolding)
        k_modifier = (t_scalar * w_scalar * o_scalar * depth_scalar)
        return jsp.linalg.block_diag(*[jnp.diag(k * k_modifier * n_scalar if i in [1,2,3] else k * k_modifier) for i,k in enumerate(ks)]) # only for the litter pools (pools 2,3,4) do we multiply by n_scalar

    @jax.jit
    def _dX(self, t, X):
        """ODE to be integrated to calculate C pools via CLM style CENTURY type model. 

        K_t is time dependent, needs to be calculated at each step.
        Other matrices are constant.
        
        dX = I_t - (AK_t - V) * X

        Args:
            X: state matrix of C pools
            t: time

        Returns:
            dX: change in state matrix of C pools
        """

        # t_ind = int((t % (1 / 12)) * 12 * 12)  # t is in units of years, and the dt for the I, T and P is in months, so we multiply by 12 to get the index
        t_ind = jnp.array((t % (1 / 12)) * 12 * 12, int)  # t is in units of years, and the dt for the I, T and P is in months, so we multiply by 12 to get the index
        I_t = self.I[t_ind,:].values
        
        # taus = self.config.taus
        # zsoi = self.config.zsoi
        # w_scalar = self.env.w
        # t_scalar = self.env.t
        # o_scalar = self.env.o
        # n_scalar = self.env.n
        # decomp_depth_efolding = self.config.decomp_depth_efolding
        # nlevels = self.config.nlevels

        # K_t = self.make_K_matrix(taus, zsoi,
                            # w_scalar[t_ind,:], t_scalar[t_ind,:], o_scalar[t_ind,:], n_scalar[t_ind,:],
                            # decomp_depth_efolding, nlevels)
        
        A_t = jnp.subtract(jnp.dot(self.A, self.K_ts[t_ind,:,:]), self.V)
        dX = jnp.add(I_t, jnp.dot(A_t , X))
        # dX = I_t + (self.A @ K_t[t_ind,:,:] - self.V) @ X#.values
        return dX
    
    def run(self, timesteps, dt):
        """
        Run the model for a series of timesteps.

        Args:
            timesteps: list of timesteps to run the model for
            dt: timestep size
        """
        Xs = [self.env.X0]
        for i,ts in enumerate(timesteps):
            # print("Running timestep ", i,ts)
            # print(self._CLM_vertical(Xs[-1], ts))
            Xs.append(Xs[-1] + self._dX(Xs[-1], ts) * dt)
        
        return xr.concat(Xs, dim = 'TIME')


class JULES(CompartmentalModel):
    """ 
    Implementation of the JULES soil carbon model. 
    Based on https://gmd.copernicus.org/articles/4/701/2011/gmd-4-701-2011.pdf#page=11.83
    
    TODO: delete this class, it is unused and untested.
    """
    
    def __init__(self, config: namedtuple, env_params: namedtuple):
        """Construct an instance for a single grid cell.

        Args:
            config: instance of ConfigParams.
                Contains all the global configuration parameters.
            env_params: instance of LocDependentData.
                Contains all the data that are specific to a location.
        """
        super().__init__(config, env_params)
        self.K = self.make_K_matrix(config.ks, env_params.T, env_params.s_limit, env_params.u)
        self.A = self.make_A_matrix(env_params.clay)
        self.u = self.make_u(env_params.alpha_dr)

    def make_K_matrix(ks, T, s_limit, u):
        """Make the K matrix for the JULES model.

        Parameters
        ----------
        ks : np.array
            The rate constants for each soil carbon pool in units of 1/s
        T : float
            The temperature in degrees Celsius
        s_limit : float
            The soil moisture limit (0-1)
        
        u : float
            Fractional vegetation cover (0-1)
        
        Returns
        -------
        np.array
            The K matrix
        """


        ks = np.diag(ks *SECS_PER_DAY * DAYS_PER_YEAR)  # convert to seconds

        F_T = 2**(T - 298.15) / 10
        F_u = 0.6 + 0.4 * (1 - u)
        K = ks * s_limit * F_T * F_u

        return K

    def make_A_matrix(self, clay, K):
        """Make the A matrix for the JULES model.
        
        Parameters
        ----------
        clay : float
            The clay content of the soil (0-1)
        K : np.array
            The K matrix for the JULES model
        
        Returns
        -------
        np.array
            The A matrix
        """

        x = 1.67 * (1.21 + 2.24 * np.exp(-0.085 * 0.45 * clay / 100)) # from the Carlos Sierra paper https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018GB005950
        beta_r = 1 / (x + 1)

        A = - np.diag([1,1,1,1])
        A[2,:] += 0.46 * beta_r * K.diagonal() 
        A[3,:] += 0.54 * beta_r * K.diagonal()

        return A
    
    def make_u(alpha_dr):
        f_dpm = alpha_dr / (1 + alpha_dr)  # fraction of NPP that goes to DPM
        return np.array([f_dpm, 1 - f_dpm, 0, 0])  # Initialize u for each pool

    def _dX(self, X, t):
        
        I_t = self.I[t,:].values

        K_t = self.make_K_matrix(self.config.ks, self.env.T[t], self.env.s_limit, self.env.u)
        A_t = self.make_A_matrix(self.env.clay, K_t)
        dX = I_t * self.u + (A_t @ K_t) @ X.values
        return dX
    
    def run(self, timesteps, dt):
        """
        Run the model for a series of timesteps.

        Args:
            timesteps: list of timesteps to run the model for
            dt: timestep size
        """
        Xs = [self.env.X0]
        for i,ts in enumerate(timesteps):
            Xs.append(Xs[-1] + self._dX(Xs[-1], ts) * dt)
        
        return xr.concat(Xs, dim = 'TIME')
    
    def pA_ss(self):
        """Calculate the steady-state age distribution."""
        return box_model_ss_age_dist(self.K @ self.A, self.u)
    
    def pA_dynamic(self, dt, tmax):

        def A_t(self, t):
            K_t = self.make_K_matrix(self.config.ks, self.env.T[t], self.env.s_limit, self.env.u)
            A_t = self.make_A_matrix(self.env.clay, K_t)
            return A_t @ K_t
        
        def I_t(t):
            """Get the input at time t."""
            return self.I[t,:].values * self.u
        
        """Calculate the age distribution at a given time."""
        return dynamic_age_dist(A_t, I_t, dt, tmax)


class CABLE(CompartmentalModel):
    def __init__(self, config: namedtuple):
        """Construct an instance for a single grid cell.

        Args:
            config: instance of ConfigParams.
                Contains all the global configuration parameters.
            env_params: instance of LocDependentData.
                Contains all the data that are specific to a location.
        """
        super().__init__(config, None)
        self.K = np.diag(config.K * config.eta)
        self.A = config.A
        self.u = config.u
        
        # check if config has I, if not define it as 
        if not hasattr(self.config, 'I'):
            self.I = lambda t: np.ones(len(config.K))
    
    def _dX(self, X, t):
        """ODE to be integrated to calculate C pools via CABLE model. 

        K_t is time dependent, needs to be calculated at each step.
        Other matrices are constant.
        
        dX = I_t - (AK_t - V) * X

        Args:
            X: state matrix of C pools
            t: time

        Returns:
            dX: change in state matrix of C pools
        """
        I_t = self.I(t) * self.u
        
        dX = I_t + (self.A @ self.K) @ X.values
        return dX
    
    def pA_ss(self, ages):
        """Calculate the steady-state age distribution."""
        return box_model_ss_age_dist(self.K @ self.A, self.u, ages)


class JSBACH(CompartmentalModel):

    """A model based on JSBACH with vertical diffusion and advection."""
    
    def __init__(self, config: namedtuple, env_params: namedtuple):
        """Construct an instance for a single grid cell.

        Args:
            config: instance of ConfigParams.
                Contains all the global configuration parameters.
            env_params: instance of LocDependentData.
                Contains all the data that are specific to a location.
                For JSBACH, this includes the d of the CWD diameter, 
                the temperature, the precipitation time series.
        """
        super().__init__(config, env_params)
        self.I = env_params.I
        self.A = self.make_A_matrix()
        self.K = np.stack([self.make_K_matrix(env_params.T[i], env_params.P[i], env_params.d) for i in range(12)]) # calculate K for each month
        self.u = self.make_B()

        self.X_size = self.u.shape
        
    def make_A_matrix(self):
        """Make the A matrix for the JSBACH model.
        """
        # Define the transfer coefficients based on the JSBACH model
        # from https://pure.mpg.de/rest/items/item_3279802_26/component/file_3316522/content#page=107.51 and 
        # https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/src/carbon/mo_carbon_process.f90
        # Eq 6.32
        A_W = 0.99; A_E = 0.; A_N = 0.;
        W_A = 0.48; W_E = 0.; W_N = 0.015;
        E_A = 0.01; E_W = 0.; E_N = 0.95;
        N_A = 0.83; N_W = 0.01; N_E = 0.02;
        mu_H = 0.0045; # Eq. 6.33

        # Construct the A matrix for the 4 litter pools, excluding the humus pool
        A_pool = np.array([[-1, W_A, E_A, N_A],
                             [A_W, -1, E_W, N_W],
                             [A_E, W_E, -1, N_E],
                             [A_N, W_N, E_N, -1]])
        
        # Create a block diagonal matrix for the pools for aboveground and belowground litter
        A_total = np.diag(-np.ones(9))
        A_total[:8,:8]= block_diag(A_pool,A_pool)
        
        # Add the humus pool
        # A_total[:8,8] = mu_H  # transfer from all pools to humus
        A_total[8,:8] = mu_H  # transfer from all pools to humus
        
        # Build a matrix for woody and non-woody litter pools

        A = block_diag(A_total, A_total)

        return A
    
    def make_K_matrix(self, T, P, d):
        """Make the K matrix for the JSBACH model.

        Parameters
        ----------
        T : float
            The temperature in degrees Celsius
        P : float
            The precipitation in m/yr
        d : float
            The diameter of CWD - from WoodLitterSize in https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/data/lctlib_nlct21.def it is 4

        Returns
        -------
        np.array
            The K matrix
        """
        # from https://pure.mpg.de/rest/items/item_3279802_26/component/file_3316522/content#page=107.51 and 
        # https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/src/carbon/mo_carbon_process.f90
        a_i=np.array([0.72, 5.9, 0.28, 0.031]) # Eq. 6.28
        a_h = 0.0016 # Eq. 6.34
        b1 = 9.5e-2; b2 = -1.4e-3; gamma = -1.21; # Eq. 6.30 - T in C and P in m/yr
        phi1 = -1.71; phi2 = 0.86; r = -0.306; # Eq. 6.31

        k_clim = (np.exp(b1 * T + b2 * T**2)) * (1 - np.exp(gamma * P)) # Eq. 6.30

        h_s = np.min([1, (1 + phi1 * d + phi2 * d**2)**r]) # Eq. 6.31

        K_pools = np.hstack([a_i , a_i , a_h, a_i * h_s, a_i * h_s, a_h]) * k_clim # Eq. 6.29 + Eq. 6.34 - first for non-woody pools and then for woody pools

        return np.diag(K_pools)
    
    def make_B(self):
        """Make the B matrix for the JSBACH model.
        """
        
        # distribute NPP
        fract_npp_2_woodPool = 0.3 # fraction of NPP that goes to wood pool
        fract_npp_2_reservePool = 0.05 # fraction of NPP that goes to reserve pool
        fract_npp_2_exudates = 0.05 # fraction of NPP that goes to root exudates
        # NPP_2_woodPool    = fract_npp_2_woodPool * NPP # NPP mol(C)/m2/yr
        # NPP_2_reservePool = fract_npp_2_reservePool * NPP
        # NPP_2_rootExudates= fract_npp_2_exudates * NPP
        # NPP_2_greenPool = (1 - fract_npp_2_woodPool - fract_npp_2_reservePool - fract_npp_2_exudates) * NPP

    
        LeafLit_coef = np.array([0.4651, 0.304, 0.0942, 0.1367, 0.]) #coefficient to distribute leaf litter into 5 classes of chemical composition
        WoodLit_coef = np.array([0.65, 0.025, 0.025, 0.3, 0.]) #coefficient to distribute woody litter into 5 classes of chemical composition

        fract_wood_aboveGround = 0.7 # !< Fraction of C above ground in wood pool (for separation of woody litter into above and below ground litter pools)
        fract_green_aboveGround = 0.5 # !< Fraction of C above ground in green pool (for separation of green litter into above and below ground litter pools)



        # greenC2leafC = 4 # ratio of carbon green pool (leaves, fine roots, starches, sugars) to leaf carbon
        # leaf_shedding = 0.000342 # Time in which leaves are constantly shedded [days-1] 
        # specific_leaf_area_C = 0.264 # Carbon content per leaf area in [m2(leaf)/mol(Carbon)] 
        # C_2_litter_greenPool = greenC2leafC * leaf_shedding / specific_leaf_area_C
        
        
        # C_2_litter_woodPool = c_woods / tau_c_woods
        # cflux_c_greenwood_2_litter = cflux_c_greenwood_2_litter + C_2_litter_woodPool

        # # Trace back leafLitter
        
        
        # C_2_litter_greenPool =  MIN(greenC2leafC * leaf_shedding / specific_leaf_area_C, c_green)
        # cflux_c_greenwood_2_litter = C_2_litter_greenPool

        # C_2_litter_woodPool = c_woods / tau_c_woods
        # cflux_c_greenwood_2_litter = cflux_c_greenwood_2_litter + C_2_litter_woodPool
        
        # C_2_litter_greenPool = c_reserve / tau_c_reserve
        # cflux_c_greenwood_2_litter = cflux_c_greenwood_2_litter + C_2_litter_greenPool

        # leafLitter    = cflux_c_greenwood_2_litter - C_2_litter_woodPool + Cflx_faeces_2_LG - Cflx_2_crop_harvest


        # Summary

        # NPP goes to green pool, wood pool, reserve pool and root exudates by multiplying the fractions by NPP_pot_yDayMean, this is the input into the 
        # 3 vegetation pools - green, wood and reserve pools

        # from there, the carbon istransferred to the litter pools
        # 1. For green pool, it is based on the LAI and the shedding rate - probably can be casted as a turnover time
        # 2. For wood pool, it is based on the turnover time of the wood
        # 3. For reserve pool, it is based on the turnover time of the reserve
        # 4. root exudates flux is transferred to the belowground water soluble pool
        # 5. The litter pools are then distributed into the aboveground and belowground pools based on the fraction of aboveground and belowground litter
        #    and to the different litter pools based on the coefficients for leaf and wood litter
        
        # As a first approximation, I'll just assume the living pools are in steady state, so the output from each pool is equal to the input into the pool.
        # This means that the input into the litter pools is:
        NPP_fractions = np.array([1. - (fract_npp_2_woodPool + fract_npp_2_exudates), fract_npp_2_woodPool, fract_npp_2_exudates])
        above_below = np.array([[fract_green_aboveGround, 1 - fract_green_aboveGround],
                                [fract_wood_aboveGround, 1 - fract_wood_aboveGround],
                                [0, 1]])
        litter_pool_split = np.stack([LeafLit_coef,WoodLit_coef,np.array([0,1,0,0,0])])
        #NPP * NPP_fractions (3,1 - green, wood, exudates) * litter_fractions (4,3)  * above_below (3,2) 
        # ((NPP_fractions * litter_pool_split[:,:-1].T) @ above_below).T.flatten()
        NPP_to_litter_pools = (NPP_fractions * litter_pool_split[:,:-1].T) # size 4,3

        NPP_pools_above_below = np.stack([NPP_to_litter_pools,NPP_to_litter_pools],axis=2) * above_below # size 4,3,2
        veg_to_litter = np.array([[1, 0, 1], [0, 1, 0]]) # the vegetation pools that contribute to the non-woody and woody litter pools
        B_temp = (NPP_pools_above_below.transpose(0,2,1) @ veg_to_litter.T) # size 4,2,2
        B = np.concatenate([B_temp.transpose(1,0,2).reshape(8,2),  np.zeros((1,2))]).T.flatten() # size 18,
        
        return B
        

        # The output vector should be size 18, composed of:
        # 4 pools of non-woody litter for aboveground, 4 pools of non-woody litter for belowground, and one pool of non-woody humus
        # 4 pools of woody litter for aboveground, 4 pools of woody litter for belowground, and one pool of woody humus

    def _dX(self, t, X):
        
        t_ind = int((t % (1 / 12)) * 12 * 12) # t is in units of years, and the dt for the I, T and P is in months, so we multiply by 12 to get the index
        I_t = self.I[t_ind]
        
        dX = I_t * self.u + (self.A @ self.K[t_ind,:]) @ X
        return dX
    
    def _impulse(self, t, X):
        """Define the impulse response for the JSBACH model.

        Args:
            t: time in years
            t_ind: index of the time in the input data
        """
        # Define the impulse input as a function of time
        # This is a placeholder, you can define your own impulse input function
        t_ind = int((t % (1 / 12)) * 12 * 12) # t is in units of years, and the dt for the I, T and P is in months, so we multiply by 12 to get the index
        
        dX = (self.A @ self.K[t_ind,:]) @ X
        return dX

    def pA(self, tmax: np.ndarray) -> np.ndarray:
        """Calculate the age distribution for the model.

        Args:
            tmax: maximum time in years to calculate the age distribution for

        Returns:
            np.ndarray: age distribution for the model
        """
        
        ts = np.logspace(-2,np.log10(tmax),1000)
        solution = solve_ivp(self._impulse, t_span=(0,tmax*1.01), y0 = self.u, method="LSODA", t_eval=ts)
        ys = solution.y.sum(axis=0)

        t_inds = ((ts % (1 / 12)) * 12 * 12).astype(int) # t is in units of years, and the dt for the I, T and P is in months, so we multiply by 12 to get the index
        pA = ys * self.I[t_inds]
        return pA
    

    
    def run(self, timesteps, dt):
        """
        Run the model for a series of timesteps.

        Args:
            timesteps: list of timesteps to run the model for
            dt: timestep size
        """
        Xs = [self.env.X0]
        for i,ts in enumerate(timesteps):
            # print("Running timestep ", i,ts)
            # print(self._CLM_vertical(Xs[-1], ts))
            Xs.append(Xs[-1] + self._dX(Xs[-1], ts) * dt)
        
        return xr.concat(Xs, dim = 'TIME')


class ReducedComplexModel(CompartmentalModel):
    def __init__(self, config, params):
        super().__init__(config, params)

        self.A = self.make_A_matrix()
        self.u = self.make_u()

    def make_A_matrix(self):
        model = self.config.model
        params = self.env.params
        rs_fac = self.config.rs_fac[model]
        tau_fac = self.config.tau_fac[model]

        A = np.diag(-1 / params[:3])
        if self.config.correct:
            A[1,0] = params[3] / params[0] if model !='MRI' else 0.46 * params[3] / params[0]
            A[2,2] = -1 / (params[2] * tau_fac)
            A[2,1] = rs_fac * params[4] / params[1]
        else:
            A[1,0] = params[3] / params[0]
            A[2,1] = params[4] / params[1]
        
        return A
    
    def make_u(self):
        """Make the u vector for the ReducedComplexModel."""
        # The u vector is the fraction of NPP that goes to each pool
        return np.array([self.env.params[5],0.0,0.0])
    
    def cdf(self, t):
        """Calculate the cumulative distribution function for the model."""
        # This is a placeholder, you can define your own CDF function
        return calc_age_dist_cdf(self.A, self.u, t)
