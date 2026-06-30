# %%
# If currently in notebooks/ directory, go up one level.
import os
import sys

if os.path.basename(os.getcwd()) == 'notebooks':
    sys.path.append(os.getcwd())
    os.chdir('..')
    print(f'Changed working directory to {os.getcwd()}')

# %% imports
import numpy as np

from soil_diskin.continuum_models import LognormalDisKinFast
from soil_diskin.radiocarbon_utils import load_atm14c
import multiprocessing
from tqdm import tqdm

# %% calculations needed to make figure 2 (lognormal version)
# Scan over mu and sigma and calculate T and R_14C.
mu_values = np.linspace(-2.5, 0.0, 100)
sigma_values = np.linspace(0.1, 2.0, 100)
MU, SIGMA = np.meshgrid(mu_values, sigma_values, indexing='ij')

atm = load_atm14c()

T_matrix = np.zeros((len(mu_values), len(sigma_values)))
R_14C_matrix = np.zeros((len(mu_values), len(sigma_values)))


def _init_worker(atm_path):
    global WORKER_ATM
    WORKER_ATM = load_atm14c(atm_path)


def _compute_cell(task):
    i, j, mu, sigma = task
    model = LognormalDisKinFast(mu=mu, sigma=sigma, atm=WORKER_ATM)
    r14c, _ = model.calc_radiocarbon_ratio_ss()
    return (i, j, model.T, r14c)


# Use multiprocessing when run as a script; otherwise fall back to serial loop
if __name__ == "__main__":
    atm_path = "data/14C_atm_annot.csv"
    # build list of tasks
    tasks = [(i, j, mu_values[i], sigma_values[j]) for i in range(len(mu_values)) for j in range(len(sigma_values))]

    cpu_count = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=cpu_count, initializer=_init_worker, initargs=(atm_path,)) as pool:
        pbar = tqdm(total=len(tasks), desc='cells', unit='cell')
        for i, j, Tval, rval in pool.imap_unordered(_compute_cell, tasks, chunksize=16):
            T_matrix[i, j] = Tval
            R_14C_matrix[i, j] = rval
            pbar.update(1)
        pbar.close()
    # %% Save grids and outputs to an npz file.
    np.savez(
        'results/fig2_calcs.npz',
        mu_values=mu_values,
        sigma_values=sigma_values,
        MU=MU,
        SIGMA=SIGMA,
        T_matrix=T_matrix,
        R_14C_matrix=R_14C_matrix,
    )
