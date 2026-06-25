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
from soil_diskin.continuum_models import PowerLawDisKin

# %% calculations needed to make figure 1 (power law simulation)
# Generate impulse train and run simulation
np.random.seed(1234)
n_inputs = 50000
J_t = np.random.normal(10, 2, n_inputs)

# Use the power law model to simulate decays for figures.
my_sim = PowerLawDisKin(t_min=1, t_max=1000)
ts = np.arange(n_inputs)
g_ts = my_sim.run_simulation(ts, J_t)

# %% Save to an npz file.
if __name__ == "__main__":
    np.savez(
        'results/fig1_calcs.npz',
        J_t=J_t,
        ts=ts,
        g_ts=g_ts,
        n_inputs=n_inputs,
    )
    print(f"Saved fig1_calcs.npz: n_inputs={n_inputs}, g_ts.shape={g_ts.shape}")
