#!/usr/bin/env python
"""
Convert HDF5 lognormal age scan results to CSV format.

The HDF5 files contain:
- /results: 2D array (sites × ages)
- /ages: 1D array of age values
- /site_indices: 1D array of site indices

The output CSV has ages as columns and sites as rows.
"""

import argparse
import h5py
import pandas as pd


def convert_h5_to_csv(input_h5_path, output_csv_path):
    """Convert HDF5 scan results to CSV format."""
    # Read HDF5 file
    with h5py.File(input_h5_path, 'r') as f:
        results = f['/results'][:]
        ages = f['/ages'][:]
        site_indices = f['/site_indices'][:]
    
    # Create DataFrame with ages as columns and site_indices as index
    df = pd.DataFrame(
        results,
        index=site_indices,
        columns=ages
    )
    
    # Set index name for clarity
    df.index.name = 'site_index'
    
    # Export to CSV
    df.to_csv(output_csv_path)
    print(f"Converted {input_h5_path} to {output_csv_path}")
    print(f"Shape: {df.shape} ({len(site_indices)} sites × {len(ages)} ages)")


def main():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 lognormal age scan results to CSV format'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input HDF5 file path'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    convert_h5_to_csv(args.input, args.output)


if __name__ == '__main__':
    main()
