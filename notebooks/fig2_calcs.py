# %% 
# if currently in notebooks/ directory, go up one level
import os
import sys

# Add the notebooks/ directory to the sys.path
print(os.getcwd())

# If we are in the notebooks/ directory, go up one level
if os.path.basename(os.getcwd()) == 'notebooks':
    sys.path.append(os.getcwd())
    os.chdir('..')
    print(f'Changed working directory to {os.getcwd()}')

# %% imports
import numpy as np

from soil_diskin.continuum_models import PowerLawDisKin
from tqdm import tqdm

# %% calculations needed to make figure 2
# first scan over t_min and t_max to calculate T and R_14C
# TODO: move all below here a new fig2.py once the 
# simulation is importable. 
# TODO: run this separately from the figure 2 plotting code
# and save the results to a file, since it takes a while to run.

# Exemplify calibration of the power-law model.
# vary t_min and t_max and calculate the turnover T and
# steady-state radiocarbon ratio. Store these in matrices
# and plot as contour plots.
t_min_values = np.linspace(0.1, 10, 100)
t_max_values = np.linspace(1000, 100000, 100)
t_min_grid, t_max_grid = np.meshgrid(
    t_min_values, t_max_values, indexing='ij')

T_matrix = np.zeros((len(t_min_values), len(t_max_values)))
R_14C_matrix = np.zeros((len(t_min_values), len(t_max_values)))

for i, t_min in enumerate(tqdm(t_min_values, desc='t_min loop')):
    for j, t_max in enumerate(tqdm(t_max_values, desc='t_max loop', leave=False)):
        my_model = PowerLawDisKin(t_min=t_min, t_max=t_max)
        T_matrix[i, j] = my_model.T
        R_14_C, _err = my_model.calc_radiocarbon_ratio_ss()

        R_14C_matrix[i, j] = R_14_C

# %% Save t_min, t_max, T_matrix, R_14C_matrix to a npz file
np.savez('results/fig2_calcs.npz',
         t_min_values=t_min_values,
         t_max_values=t_max_values,
         T_matrix=T_matrix,
         R_14C_matrix=R_14C_matrix)

