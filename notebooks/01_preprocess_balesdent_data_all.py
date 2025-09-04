#%% 00 - load libraries
import os

import pandas as pd
import numpy as np
from soil_diskin.utils import download_file
from soil_diskin.data_wrangling_all import process_balesdent_data

"""
All scripts to be run from project root directory.

This script downloads and processes the raw data from Balesdent et al. 2018,
which is used to calculate soil carbon turnover times. The processed data is
saved to a CSV file in the results folder.
"""

if __name__ == "__main__":
    folder = "../data/balesdant_2018/"
    raw_data_filename = "41586_2018_328_MOESM3_ESM.xlsx"
    raw_data_url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0328-3/MediaObjects/41586_2018_328_MOESM3_ESM.xlsx"
    download_file(raw_data_url, folder, raw_data_filename)

    raw_data = pd.read_excel(os.path.join(folder,raw_data_filename),skiprows=7)
    final_data = process_balesdent_data(raw_data)
    final_data.to_csv(os.path.join('results/processed_balesdant_2018_all.csv'), index=False)

## TODO

# 1. Test that the processing works as expected for a predefined set of data, including data with missing values in deeper layers
# %%
