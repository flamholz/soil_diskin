import os
import pandas as pd

from os import path

from scipy.interpolate import interp1d

# Description: Constants used in the notebooks

CWD = os.getcwd()

# Kinetic constant associated with radiocarbon decay
LAMBDA_14C = 1/8267.0 # per year units

C14_DATA_PATH = path.join(CWD, '../data/14C_atm_annot.csv')
C14_DATA = pd.read_csv(C14_DATA_PATH)

# Interpolated 14C data loaded globally to avoid recomputing it
# Interpolator wants a monotonically increasing x series
INTERP_R_14C = interp1d(C14_DATA.years_before_2000, C14_DATA.R_14C,
                        kind='zero', fill_value='extrapolate')
# default interpolation is zero order, which is just nearest neighbor