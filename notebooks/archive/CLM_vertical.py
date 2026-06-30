import xarray as xr
from CLM_vertical_utils import *
from collections import namedtuple


# Data structures
# Global configuration parameters
ConfigParams = namedtuple('CP', ['decomp_depth_efolding', 'taus', 'Gamma_soil', 'F_soil',
                                 'zsoi', 'zisoi', 'dz', 'dz_node', 'nlevels','npools'])
# Data that are specific to a location
LocDependentData = namedtuple('LDD', ['w', 't', 'o', 'n', 'sand', 'I','X0'])


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

    def make_ldd(self, lat, lon):
        w = self.w_scalar_da.sel(LAT=lat, LON=lon)
        t = self.t_scalar_da.sel(LAT=lat, LON=lon)
        o = self.o_scalar_da.sel(LAT=lat, LON=lon)
        n = self.n_scalar_da.sel(LAT=lat, LON=lon)
        sand = self.sand_da.sel(LAT=lat, LON=lon)
        I = self.inputs.sel(LAT=lat, LON=lon)
        X0 = self.X0.sel(LAT=lat, LON=lon)
        return LocDependentData(w=w, t=t, o=o, n=n, sand=sand, I=I, X0=X0)


class CLM_vertical:
    """Class for simulating a single site using the CLM vertical model."""

    def __init__(self, cp_instance, ldd_instance):
        """Construct an instance for a single grid cell.

        Args:
            cp_instance: instance of ConfigParams.
                Contains all the global configuration parameters.
            ldd_instance: instance of LocDependentData.
                Contains all the data that are specific to a location.
        """
        self.cp = cp_instance
        self.ldd = ldd_instance

        self.I = ldd_instance.I
        self.A = make_A_matrix(ldd_instance.sand, cp_instance.nlevels)
        self.V = make_V_matrix(cp_instance.Gamma_soil, cp_instance.F_soil,
                               cp_instance.npools, cp_instance.nlevels, cp_instance.dz, 
                               cp_instance.dz_node, cp_instance.zsoi, cp_instance.zisoi)
    
    def _CLM_vertical(self, X, t):
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
        I_t = self.I[t,:].values

        taus = self.cp.taus
        zsoi = self.cp.zsoi
        w_scalar = self.ldd.w
        t_scalar = self.ldd.t
        o_scalar = self.ldd.o
        n_scalar = self.ldd.n
        decomp_depth_efolding = self.cp.decomp_depth_efolding
        nlevels = self.cp.nlevels

        K_t = make_K_matrix(taus, zsoi,
                            w_scalar[t,:], t_scalar[t,:], o_scalar[t,:], n_scalar[t,:],
                            decomp_depth_efolding, nlevels)
        
        # print("I_t shape ", I_t.shape)
        # print("K_t shape ", K_t.shape)
        # print("A shape ", self.A.shape)
        # print("V shape ", self.V.shape)
        # print("X shape ", X.shape)

        # Created an error - the I_t was not copied and then was modiefied in place
        # dX = I_t
        # print(self.ldd.I[0,:].values)
        # dX -= self.A @ K_t @ X.values
        # dX += self.V @ X.values
        
        dX = I_t + (self.A @ K_t - self.V) @ X.values
        return dX
    
    def run(self, timesteps, dt):
        """
        Run the model for a series of timesteps.

        Args:
            timesteps: list of timesteps to run the model for
            dt: timestep size
        """
        Xs = [self.ldd.X0]
        for i,ts in enumerate(timesteps):
            # print("Running timestep ", i,ts)
            # print(self._CLM_vertical(Xs[-1], ts))
            Xs.append(Xs[-1] + self._CLM_vertical(Xs[-1], ts) * dt)
        
        return xr.concat(Xs, dim = 'TIME')