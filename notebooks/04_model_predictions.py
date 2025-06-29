#%% import libraries
import numpy as np
import pandas as pd
from models import * 
from collections import namedtuple
import pickle


#%% Run CABLE (CASA-CNP) model

# Create the config object for the model based on parameters for Evergreen broadleaf foress in Test S5. 
# in [Xia et al. 2013](https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.12172)

# We use only the 6 pools for litter and soil carbon. To calulcate u for the soil pools, we use the
# u values for the plant pools, and thre transfer coefficients from the plant pools to the litter pools
CABLE_config = namedtuple('Config', ['K','eta','A','u'])
CABLE_config.K = np.array([10., 0.95, 0.49, 2.19, 0.12, 0.0027])
CABLE_config.eta = 0.4 * np.ones(6)  
CABLE_config.u = np.array([0.249 * 0.69 + 0.551 * 0.6, 0.249 * 0.31 + 0.551 * 0.4, 0.2, 0, 0, 0])
CABLE_config.A = -np.diag(np.ones(6))  # Assuming 5 pools for simplicity
CABLE_config.A[3,0] = 0.45
CABLE_config.A[3,1] = 0.36
CABLE_config.A[3,2] = 0.24
CABLE_config.A[4,1] = 0.14
CABLE_config.A[4,2] = 0.28
CABLE_config.A[4,3] = 0.39
CABLE_config.A[5,3] = 0.006
CABLE_config.A[5,4] = 0.003

# Create the CABLE model instance
cable_model = CABLE(CABLE_config)
ages_cable = np.arange(0,100_000,0.1)
pA_cable = cable_model.pA_ss(ages_cable)

# Save the model predictions
with open(f'../results/model_predictions/CABLE_{pd.Timestamp.now().date().strftime("%d-%m-%Y")}.pkl','wb') as f:
    pickle.dump([ages_cable, pA_cable.squeeze()],f)

#%% Run the CLM5 model

#%% Run the JSBACH model
# from https://pure.mpg.de/rest/items/item_3279802_26/component/file_3316522/content#page=107.51 and 
# https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/src/carbon/mo_carbon_process.f90
a_i=np.array([0.72, 5.9, 0.28, 0.031]) # Eq. 6.28
aH = 0.0016 # for the humus pool
b1 = 9.5e-2; b2 = -1.4e-3; gamma = -1.21; # Eq. 6.30 - T in C and P in m/yr
phi1 = -1.71; phi2 = 0.86; r = -0.306; # Eq. 6.31

