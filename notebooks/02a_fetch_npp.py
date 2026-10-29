"""Step 02a: fetch NPP from MODIS via Earth Engine for the locations of some site tables.

NPP is fetched once per location for the union of the locations in the given
tables, and written to a table that the turnover scripts read. For the Balesdent
sites (the default) these are the bulk and depth-resolved site tables from
script 01, read by both step 02 scripts; the Snakefile runs it again for the
ISRaD profiles. This keeps the Earth Engine query in one place, gives tables of
the same sites identical NPP, and lets the rest of the pipeline rerun without
Earth Engine access.

NPP is the mean of the annual MOD17A3HGF images of --years FIRST LAST (default
soil_diskin.npp.YEARS, 2001-2025), averaged over the native-resolution pixels
within 5 km of each location, in g C/m²/yr, with fill codes masked (see
soil_diskin.npp.fetch_modis_npp).

Outputs:
  --output (default results/02_npp/site_npp.csv): latitude, longitude, NPP (g C/m²/yr),
  n_valid_pixels and valid_share (the share of the disk with data), and with
  --sd, NPP_sd: the standard deviation across the annual images (interannual
  variability, not the error of the product).
  The same path with .json: the query settings, the image IDs averaged, the
  retrieval time and the input checksums.

Run from the project root directory:
    python notebooks/02a_fetch_npp.py [--locations TABLE ...] [-o OUTPUT] [--years 2001 2025]
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from soil_diskin import npp as modis
from soil_diskin.site_inputs import COORDS, location_keys, valid_coordinates
from soil_diskin.utils import file_digest

LOCATION_PATHS = ['results/processed_balesdent_2018.csv', 'results/processed_balesdent_2018_sampled_soilgrids.csv']
OUTPUT_PATH = 'results/02_npp/site_npp.csv'


def read_locations(path):
    """The latitude and longitude columns of a table, whatever their capitalization."""
    table = pd.read_csv(path, low_memory=False, float_precision='round_trip')
    columns = {c.lower(): c for c in table.columns}
    return table[[columns[c] for c in COORDS]].set_axis(COORDS, axis=1)


def locations(paths):
    """Unique valid locations of the tables, in order of first appearance."""
    union = pd.concat([read_locations(path) for path in paths]).reset_index(drop=True)
    # Tables written by different scripts can hold one location with different last
    # bits; deduplicate on rounded coordinates, keeping the first value.
    union = union[~location_keys(union).duplicated().to_numpy()].reset_index(drop=True)
    return union[valid_coordinates(union)].reset_index(drop=True)


def main(paths=LOCATION_PATHS, output=OUTPUT_PATH, years=modis.YEARS, sd=False):
    output = Path(output)
    sites = locations(paths)
    modis.initialize()
    table = modis.fetch_modis_npp(sites, years=years, sd=sd)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    metadata = {
        'product': modis.PRODUCT, 'band': modis.BAND,
        'years': list(years) if years else 'all', 'image_ids': modis.image_ids(years),
        'mask_fill': True, 'radius_m': modis.RADIUS_M, 'decimals': modis.DECIMALS, 'sd': sd,
        'statistic': 'temporal mean of the annual images, then mean of the native pixels within radius_m',
        'units': 'g C/m²/yr', 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'sources': {Path(path).stem: {'path': str(Path(path).resolve()), 'sha256': file_digest(path)}
                    for path in paths},
        'n_locations': len(table), 'n_missing_npp': int(table.NPP.isna().sum()),
        'min_valid_share': float(table.valid_share.min()),
        'table_sha256': file_digest(output)}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved NPP for {len(table)} locations to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--locations', nargs='+', default=LOCATION_PATHS,
                        help='tables with latitude and longitude columns (any capitalization); '
                             'default: the Balesdent bulk and layer tables from script 01')
    parser.add_argument('-o', '--output', default=OUTPUT_PATH, help='output NPP table')
    parser.add_argument('--years', type=int, nargs=2, metavar=('FIRST', 'LAST'), default=modis.YEARS,
                        help=f'calendar years to average (default: {modis.YEARS[0]} {modis.YEARS[1]})')
    parser.add_argument('--sd', action='store_true', help='also record NPP_sd, the SD across annual images')
    args = parser.parse_args()
    main(args.locations, args.output, tuple(args.years), args.sd)
