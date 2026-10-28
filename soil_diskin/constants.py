import os
import pandas as pd
import sympy

from os import path

# Description: Constants used in the notebooks

CWD = os.getcwd()

# Kinetic constant associated with radiocarbon decay
LAMBDA_14C = 1/8267.0 # per year units

# Seconds in a day and days in a year for time conversions
SECS_PER_DAY = 86400
DAYS_PER_YEAR = 365
T_MELT = 273.15  # Kelvin at which water melts

C14_DATA_PATH = path.join(CWD, 'data/14C_atm_annot.csv')
# Preserve the public data/interpolator names, loading only on first access.
# Importing model classes with a caller-supplied atmosphere needs no local CSV.
# INTERP_R_14C is the default atmosphere's lookup, radiocarbon_utils.AtmC14.R.
C14_DATA: pd.DataFrame
INTERP_R_14C: callable
__all__ = ['CWD', 'LAMBDA_14C', 'SECS_PER_DAY', 'DAYS_PER_YEAR', 'T_MELT',
           'C14_DATA_PATH', 'C14_DATA', 'INTERP_R_14C', 'GAMMA']


def __getattr__(name):
    if name not in ('C14_DATA', 'INTERP_R_14C'):
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    if name == 'C14_DATA':
        globals()['C14_DATA'] = pd.read_csv(C14_DATA_PATH)
    else:
        from soil_diskin.radiocarbon_utils import default_atmosphere
        globals()['INTERP_R_14C'] = default_atmosphere().R
    return globals()[name]


# SymPy Euler-Mascheroni constant for use in the model
GAMMA = float(sympy.EulerGamma.evalf())