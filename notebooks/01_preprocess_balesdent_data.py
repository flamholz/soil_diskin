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

    # Count the number of unique locations in the raw data
    print("Loaded raw data...")
    unique_locations = raw_data[['Latitude', 'Longitude']].drop_duplicates()
    print(f"\t{len(unique_locations)} unique locations in raw data.")
    # Count the number of unique locations + duration labeling combinations
    unique_loc_duration = raw_data[['Latitude', 'Longitude', 'Duration_labeling']].drop_duplicates()
    print(f"\t{len(unique_loc_duration)} unique locations + durations in the raw data.")

    print("Processing Balesdent et al. 2018 data...")
    final_data = process_balesdent_data(raw_data)
    final_data.to_csv(os.path.join('results/processed_balesdent_2018.csv'), index=False)

    # Count the number of unique locations in the processed data
    unique_locations_processed = final_data[['Latitude', 'Longitude']].drop_duplicates()
    print(f"\t{len(unique_locations_processed)} unique locations in processed data.")
    # Count the number of unique locations + duration labeling combinations
    unique_loc_duration_processed = final_data[['Latitude', 'Longitude', 'Duration_labeling']].drop_duplicates()
    print(f"\t{len(unique_loc_duration_processed)} unique locations + durations in the processed data.")