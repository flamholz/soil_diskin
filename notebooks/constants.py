import pandas as pd
from clamped_interpolator import ClampedInterpolator

# Description: Constants used in the notebooks

# Kinetic constant associated with radiocarbon decay
LAMBDA_14C = 1/8267.0 # per year units

C14_DATA = pd.read_csv('../data/14C_atm_annot.csv')

# Interpolated 14C data loaded globally to avoid recomputing it
# Interpolator wants an monotonically increasing x
INTERP_R_14C = ClampedInterpolator(C14_DATA.years_before_2000, C14_DATA.R_14C)