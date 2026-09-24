import argparse
import pandas as pd
from soil_diskin.data_wrangling import process_balesdent_data, balesdent_layers, balesdent_sampled_layers
from soil_diskin.soilgrids_utils_w_unc import backfill_missing_soc

"""
All scripts to be run from project root directory.

This script downloads and processes the raw data from Balesdent et al. 2018,
which is used to calculate soil carbon turnover times. The processed data is
saved to a CSV file in the results folder.

For sites lacking SOC data, values can optionally be backfilled from SoilGrids 
using Google Earth Engine.

Usage:
    python notebooks/01_preprocess_balesdent_data.py -i data/balesdent_2018/balesdent_2018_raw.xlsx --backfill
    python notebooks/01_preprocess_balesdent_data.py -i input.xlsx -o output.csv --backfill
    python notebooks/01_preprocess_balesdent_data.py --no-backfill
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process Balesdent et al. 2018 soil carbon data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--depth-resolved', action='store_true',
                        help='Keep original profiles and ten 10-cm layers; no stock backfilling')
    parser.add_argument('--sampled-layers', action='store_true',
                        help='Use reported intervals, stocks, f_new and zmid from the Layers sheet')
    
    parser.add_argument(
        '-i', '--raw-file-path',
        type=str,
        default='data/balesdent_2018/balesdent_2018_raw.xlsx',
        help='Path to the raw data file'
    )
    
    parser.add_argument(
        '--backfill',
        action='store_true',
        default=False,
        help='Backfill missing SOC data from SoilGrids using Google Earth Engine'
    )
    
    parser.add_argument(
        '--no-backfill',
        action='store_false',
        dest='backfill',
        help='Do not backfill missing SOC data (default behavior)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for processed data'
    )
    
    args = parser.parse_args()
    args.depth_resolved = args.depth_resolved or args.sampled_layers
    if args.depth_resolved and args.backfill:
        parser.error('--backfill is only supported for bulk stocks')
    args.output = args.output or ('results/processed_balesdent_2018_sampled.csv' if args.sampled_layers else
                                 'results/processed_balesdent_2018_depth.csv' if args.depth_resolved
                                 else 'results/processed_balesdent_2018.csv')
    return args


if __name__ == "__main__":
    args = parse_args()
    
    print(f"Loading raw data from {args.raw_file_path}...")
    raw_data = pd.read_excel(args.raw_file_path, skiprows=7)
    if args.depth_resolved:
        layers = (balesdent_sampled_layers(raw_data, pd.read_excel(args.raw_file_path, sheet_name='Layers', header=9))
                  if args.sampled_layers else balesdent_layers(raw_data))
        layers.to_csv(args.output, index=False)
        print(f'Saved depth-resolved observations to {args.output}')
        raise SystemExit

    # Count the number of unique locations in the raw data
    print("Loaded raw data...")
    unique_locations = raw_data[['Latitude', 'Longitude']].drop_duplicates()
    print(f"\t{len(unique_locations)} unique locations in raw data.")
    # Count the number of unique locations + duration labeling combinations
    unique_loc_duration = raw_data[['Latitude', 'Longitude', 'Duration_labeling']].drop_duplicates()
    print(f"\t{len(unique_loc_duration)} unique locations + durations in the raw data.")

    print("Processing Balesdent et al. 2018 data...")
    # Keep sites with missing SOC data if backfilling is enabled
    final_data = process_balesdent_data(raw_data, keep_missing_soc=args.backfill)
    
    # Add a column for the source of the data
    final_data['C_data_source'] = 'Balesdent et al. 2018'

    # Count the number of unique locations in the processed data
    unique_locations_processed = final_data[['Latitude', 'Longitude']].drop_duplicates()
    print(f"\t{len(unique_locations_processed)} unique locations in processed data.")
    # Count the number of unique locations + duration labeling combinations
    unique_loc_duration_processed = final_data[['Latitude', 'Longitude', 'Duration_labeling']].drop_duplicates()
    print(f"\t{len(unique_loc_duration_processed)} unique locations + durations in the processed data.")
    
    # Backfill missing SOC data from SoilGrids if requested
    if args.backfill:
        print("\nBackfilling missing SOC data from SoilGrids...")
        final_data, backfill_stats = backfill_missing_soc(
            final_data,
            lat_col='Latitude',
            lon_col='Longitude', 
            soc_col='Ctotal_0-100estim',
            source_col='C_data_source',
            use_bulk_density=True,
            calc_uncertainty=True
        )
    else:
        print("\nSkipping backfill (use --backfill to enable)")
    
    # Save the final data
    print(f"\nSaving processed data to {args.output}...")
    final_data.to_csv(args.output, index=False)
    print("Done!")
