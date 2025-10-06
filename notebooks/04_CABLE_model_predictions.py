import numpy as np
import pandas as pd
from notebooks.models import * 
from collections import namedtuple
import pickle
from notebooks.constants import *

"""
Runs the CABLE model for the sites in the Balesdent dataset and saves the predictions to a pickle file.
"""

# Load the site data
site_data = pd.read_csv('results/processed_balesdent_2018.csv')
turnover_14C = pd.read_csv('results/all_sites_14C_turnover.csv')

# Create the config object for the model based on parameters for Evergreen broadleaf foress in Test S5. 
# in [Xia et al. 2013](https://onlinelibrary.wiley.com/doi/full/10.1111/gcb.12172)
print("Generating CABLE model predictions...")

# We use only the 6 pools for litter and soil carbon. To calculate u for the soil pools, we use the
# u values for the plant pools, and three transfer coefficients from the plant pools to the litter pools
# TODO: load these from a config file.
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

out_fname = f'results/04_model_predictions/CABLE.pkl'
print(f"Saving CABLE model predictions to {out_fname}")

# Save the model predictions
with open(out_fname,'wb') as f:
    pickle.dump([ages_cable, pA_cable.squeeze()],f)