"""Step 02a: fetch NPP for every Balesdent location from MODIS via Earth Engine.

NPP is fetched once per location for the union of the locations in the bulk
and depth-resolved site tables from script 01, and written to a table that
both step 02 scripts read. This keeps the Earth Engine query in one place,
gives both tables identical NPP, and lets the rest of the pipeline rerun
without Earth Engine access.

NPP is the mean of the annual MOD17A3HGF images (all of them, or --years
FIRST LAST), averaged over 10 km around each location, in g C/m²/yr (see
soil_diskin.npp.fetch_modis_npp). By default fill codes are not masked, as in
the original script 02; --mask-fill masks them.

Outputs:
  --output (default results/02_npp/site_npp.csv): latitude, longitude, NPP.
  The same path with .json: the query settings, the image IDs averaged, the
  retrieval time and the input checksums.

Run from the project root directory:
    python notebooks/02a_fetch_npp.py [--years 2001 2023] [--mask-fill]
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from soil_diskin import npp as modis
from soil_diskin.site_inputs import COORDS, location_keys, valid_coordinates
from soil_diskin.utils import file_digest

SITES_PATH = 'results/processed_balesdent_2018.csv'
LAYERS_PATH = 'results/processed_balesdent_2018_sampled_soilgrids.csv'
OUTPUT_PATH = 'results/02_npp/site_npp.csv'


def locations(sites_path, layers_path):
    """Unique valid locations of the bulk sites (Latitude/Longitude) and the layers (latitude/longitude)."""
    sites = pd.read_csv(sites_path)[['Latitude', 'Longitude']].set_axis(COORDS, axis=1)
    layers = pd.read_csv(layers_path, dtype={'profile_id': str}, float_precision='round_trip')[COORDS]
    union = pd.concat([sites, layers]).reset_index(drop=True)
    # The two tables are parsed differently, so one location can differ in its last
    # bits between them; deduplicate on rounded coordinates, keeping the bulk value.
    union = union[~location_keys(union).duplicated().to_numpy()].reset_index(drop=True)
    return union[valid_coordinates(union)].reset_index(drop=True)


def main(sites_path=SITES_PATH, layers_path=LAYERS_PATH, output=OUTPUT_PATH, years=None, mask_fill=False):
    output = Path(output)
    sites = locations(sites_path, layers_path)
    modis.initialize()
    table = modis.fetch_modis_npp(sites, years=years, mask_fill=mask_fill)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    metadata = {
        'product': modis.PRODUCT, 'band': modis.BAND,
        'years': list(years) if years else 'all', 'image_ids': modis.image_ids(years),
        'mask_fill': mask_fill, 'footprint_m': modis.FOOTPRINT_M,
        'statistic': 'temporal mean of the annual images, then spatial mean over the footprint',
        'units': 'g C/m²/yr', 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'sources': {name: {'path': str(Path(path).resolve()), 'sha256': file_digest(path)}
                    for name, path in [('sites', sites_path), ('layers', layers_path)]},
        'n_locations': len(table), 'n_missing_npp': int(table.NPP.isna().sum()),
        'table_sha256': file_digest(output)}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved NPP for {len(table)} locations to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sites', default=SITES_PATH, help='bulk site table from script 01')
    parser.add_argument('--layers', default=LAYERS_PATH, help='depth-resolved layer table from script 01')
    parser.add_argument('-o', '--output', default=OUTPUT_PATH, help='output NPP table')
    parser.add_argument('--years', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help='average only these calendar years (default: all images)')
    parser.add_argument('--mask-fill', action='store_true', help='mask MODIS fill codes before averaging')
    args = parser.parse_args()
    main(args.sites, args.layers, args.output, args.years, args.mask_fill)
