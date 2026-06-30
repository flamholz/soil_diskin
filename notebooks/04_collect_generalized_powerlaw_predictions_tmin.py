import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from soil_diskin.continuum_models import GammaDisKin, PowerLawDisKin, GeneralPowerLawDisKin
from glob import glob
"""
Collects the continuum model predictions for all sites and saves them to CSV files.
"""

# Load the site data
site_data = pd.read_csv('results/processed_balesdent_2018.csv')
turnover_14C = pd.read_csv('results/all_sites_14C_turnover.csv')

#%% Generalized Power-law model with beta = np.exp(-GAMMA)

# find all files with the tmin
general_power_law_files = glob('results/03_calibrate_models/general_powerlaw_model_optimization_results_tmin_*.csv')
# for each file perform the same process of generating predictions and saving them to a csv file
for file in general_power_law_files:
    print(f"Processing file: {file}")
    general_power_law_params = pd.read_csv(file)

    print("Generating generalized power-law model predictions...")
    predictions = []
    for i, row in general_power_law_params.iterrows():
        model = GeneralPowerLawDisKin(t_min=row['t_min'], t_max=row['t_max'], beta=row['beta'])
        if not model.params_valid():
            print(f"Invalid parameters for site {i}: t_min={row['t_min']}, t_max={row['t_max']}, beta={row['beta']}")
        
        predictions.append(model.cdfA(site_data.loc[i, 'Duration_labeling']))
    predictions = np.array(predictions)
    
    # Save the model predictions
    t_min_token = file.split('tmin_')[1].split('.csv')[0]
    output_file = f'results/04_model_predictions/general_power_law_tmin_{t_min_token}.csv'
    np.savetxt(output_file, predictions)
    print(f"Saved predictions to {output_file}")

    