import os
import pandas as pd
from soil_diskin.data_wrangling import process_balesdent_data

"""
All scripts to be run from project root directory.

This script downloads and processes the raw data from Balesdent et al. 2018,
which is used to calculate soil carbon turnover times. The processed data is
saved to a CSV file in the results folder.
"""

if __name__ == "__main__":
    # Data should be downloaded by snakemake already
    folder = "data/balesdent_2018/"
    raw_data_filename = "balesdent_2018_raw.xlsx"
    raw_data_path = os.path.join(folder, raw_data_filename)
    raw_data = pd.read_excel(raw_data_path, skiprows=7)

    final_data = process_balesdent_data(raw_data)
    final_data.to_csv(os.path.join('results/processed_balesdent_2018.csv'), index=False)

## TODO merge with 01_preprocess_balesdent_data_all.py
