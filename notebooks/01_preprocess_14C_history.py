import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

"""
Annotates and preprocesses historical atmospheric 14C levels
for the convenience of downstream analysis.
"""

# Load the 14C data
# TODO: document the provenance of this data
C14_DATA = pd.read_csv('data/14C_atm.csv')
C14_DATA.columns = ['year', 'Delta_14C']
C14_DATA = C14_DATA.dropna().sort_values('year', ascending=True)

# Annotate the 14C data with years before present and years before 2000
C14_DATA['years_before_present'] = (C14_DATA.year - C14_DATA.year.max()).abs()
C14_DATA['years_before_2000'] = (2000 - C14_DATA.year)
# Calculate R_14C, which is the ratio of 14C/12C relative in per mil
C14_DATA['R_14C'] = C14_DATA.Delta_14C / 1000 + 1
C14_DATA = C14_DATA.sort_values('year', ascending=False)
C14_DATA.to_csv('data/14C_atm_annot.csv', index=False)

# Save an interpolated version? 
# INTERP_R_14C = interp1d(C14_DATA.years_before_2000, C14_DATA.R_14C,
#                         kind='zero', fill_value='extrapolate')