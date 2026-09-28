import argparse
import json
from pathlib import Path
import pandas as pd
from soil_diskin.data_wrangling import process_balesdent_data, balesdent_sampled_layers
from soil_diskin.soilgrids_utils_w_unc import backfill_missing_soc

"""
All scripts to be run from project root directory.

This script downloads and processes the raw data from Balesdent et al. 2018,
which is used to calculate soil carbon turnover times. The processed data is
saved to a CSV file in the results folder.

For sites lacking SOC data, values can optionally be backfilled from SoilGrids 
using the SoilGrids WCS service.

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
    parser.add_argument('--depth-resolved', '--sampled-layers', action='store_true',
                        help='Use reported depth intervals, stocks and f_new from the Layers sheet')
    parser.add_argument('--soilgrids-cache', default='results/soilgrids_layer_cache.json',
                        help='Cache SoilGrids means and quantiles by location and depth')
    
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
        help='Backfill missing SOC data from SoilGrids using WCS'
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
    suffix = '_sampled' if args.depth_resolved else ''
    if args.depth_resolved and args.backfill:
        suffix += '_soilgrids'
    args.output = args.output or f'results/processed_balesdent_2018{suffix}.csv'
    return args


if __name__ == "__main__":
    args = parse_args()
    
    print(f"Loading raw data from {args.raw_file_path}...")
    raw_data = pd.read_excel(args.raw_file_path, skiprows=7)
    if args.depth_resolved:
        layers = balesdent_sampled_layers(raw_data, pd.read_excel(args.raw_file_path, sheet_name='Layers', header=9))
        if args.backfill:
            from soil_diskin.utils import file_digest
            layers, metadata = backfill_missing_soc(layers, args.soilgrids_cache)
            metadata['workbook_sha256'] = file_digest(args.raw_file_path)
            Path(args.output).with_suffix('.json').write_text(json.dumps({'stock_backfill': metadata}, indent=2)+'\n')
            print(metadata)
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
        final_data, backfill_stats = backfill_missing_soc(final_data, args.soilgrids_cache)
    else:
        print("\nSkipping backfill (use --backfill to enable)")
    
    # Save the final data
    print(f"\nSaving processed data to {args.output}...")
    final_data.to_csv(args.output, index=False)
    print("Done!")
