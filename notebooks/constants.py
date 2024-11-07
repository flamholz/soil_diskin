import pandas as pd
from clamped_interpolator import ClampedInterpolator

# Description: Constants used in the notebooks

# Kinetic constant associated with radiocarbon decay
LAMBDA_14C = 1/8267 # per year units

# Load the 14C data
# TODO: save a sorted version with years before present and column names
C14_DATA = pd.read_csv('../data/14C_atm.csv')
C14_DATA.columns = ['year', 'Delta_14C']
C14_DATA = C14_DATA.dropna().sort_values('year', ascending=True)
C14_DATA['years_before_present'] = (C14_DATA.year - C14_DATA.year.max()).abs()
C14_DATA = C14_DATA.sort_values('year', ascending=False)

# Interpolated 14C data loaded globally to avoid recomputing it
# Interpolator wants an monotonically increasing x
INTERP_14C = ClampedInterpolator(C14_DATA.years_before_present, C14_DATA.Delta_14C)