import numpy as np
import pandas as pd
from soil_diskin.compartmental_models import JSBACH 
from collections import namedtuple
from soil_diskin.constants import DAYS_PER_YEAR
import subprocess

# from https://pure.mpg.de/rest/items/item_3279802_26/component/file_3316522/content#page=107.51 and 
# https://gitlab.dkrz.de/icon/icon-model/-/blob/release-2024.10-public/externals/jsbach/src/carbon/mo_carbon_process.f90
a_i=np.array([0.72, 5.9, 0.28, 0.031]) # Eq. 6.28
a_h = 0.0016 # Eq. 6.34
b1 = 9.5e-2; b2 = -1.4e-3; gamma = -1.21; # Eq. 6.30 - T in C and P in m/yr
phi1 = -1.71; phi2 = 0.86; r = -0.306; # Eq. 6.31

JSBACH_config = namedtuple('Config', ['a_i', 'a_h', 'b1', 'b2', 'gamma', 'phi1', 'phi2', 'r'])
JSBACH_config.a_i = a_i
JSBACH_config.a_h = a_h
JSBACH_config.b1 = b1
JSBACH_config.b2 = b2
JSBACH_config.gamma = gamma
JSBACH_config.phi1 = phi1
JSBACH_config.phi2 = phi2
JSBACH_config.r = r

# run the bash command inside the script 
# subprocess.run(['cd', '../../jsbach'])
# subprocess.run(['cd' , '../../jsbach', ' &&' ,' echo' ,' "n"' ,'|' ,' ./run_yasso_test_YMB.sh'])

# distribute NPP
fract_npp_2_woodPool = 0.3 # fraction of NPP that goes to wood pool
fract_npp_2_reservePool = 0.05 # fraction of NPP that goes to reserve pool
fract_npp_2_exudates = 0.05 # fraction of NPP that goes to root exudates

LeafLit_coef = np.array([0.4651, 0.304, 0.0942, 0.1367, 0.]) #coefficient to distribute leaf litter into 5 classes of chemical composition
WoodLit_coef = np.array([0.65, 0.025, 0.025, 0.3, 0.]) #coefficient to distribute woody litter into 5 classes of chemical composition

fract_wood_aboveGround = 0.7 # !< Fraction of C above ground in wood pool (for separation of woody litter into above and below ground litter pools)
fract_green_aboveGround = 0.5 # !< Fraction of C above ground in green pool (for separation of green litter into above and below ground litter pools)


one_vec = np.ones(12) # 12 pools in the JSBACH model
JSBACH_env_params = namedtuple('EnvParams', ['I', 'T', 'P', 'd'])(one_vec * DAYS_PER_YEAR, 25 * one_vec, one_vec, 4)

JSBACH_model = JSBACH(config=JSBACH_config,
                 env_params=JSBACH_env_params)

JSBACH_output = JSBACH_model._dX(t = 0, X = np.ones(18))[:9]
# fortran_output = pd.read_csv('../../jsbach/yasso_output.csv')
fortran_output = pd.read_csv('tests/test_data/jsbach/yasso_output.csv')
print(fortran_output['Value'].values[:9])
print(JSBACH_output[:9] * (1/DAYS_PER_YEAR) + np.ones(9))
assert np.allclose(fortran_output['Value'].values[:9],JSBACH_output[:9] * (1/DAYS_PER_YEAR) + np.ones(9), rtol=1e-3, atol=1e-3), "JSBACH model output does not match Fortran implementation output"
print("JSBACH model output matches Fortran implementation output")

# run using 
# cd "/Users/yinonmb/Weizmann Institute Dropbox/YinonMoise Bar-On/git/soil_diskin/notebooks" python -m tests.test_JSBACH