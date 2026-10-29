"""Step 01 (ISRaD): pseudo-bulk 0-100 cm soil profiles with radiocarbon from ISRaD.

Reads the ISRaD flat layer table and, with soil_diskin.israd, keeps the layers
that have measured radiocarbon and a carbon stock, then combines each profile and
sampling year whose layers cover 0-100 cm completely into one bulk soil column:
its total carbon stock and stock-weighted radiocarbon. This is the ISRaD
counterpart of the bulk Balesdent table of script 01, with the same units.

Profile-years whose stock exceeds --max-stock are dropped as implausible; e.g.
Finstad_2020 reports lyr_soc ~100x what its carbon % implies, likely in Mg C/ha
rather than g C/cm². Entries in EXCLUDED_ENTRIES are dropped whole, for the
reasons given there. Profiles sampled after the end of the atmospheric 14C
record (2013) are kept, with their sampling year, and are left to the
calibration to exclude. Profiles sampled before 1950 (e.g. archived samples) are
kept and listed in the JSON for review.

Outputs:
  --output (default results/israd/israd_profiles.csv): one row per profile-year
  with entry_name, site_name, pro_name, sampling_year, latitude, longitude,
  n_layers, n_fallback_layers (layers whose stock used total carbon or bulk
  density with coarse fragments), stock_kg_m2 (0-100 cm, kg C/m²), 14C (per mil),
  fm_obs (R = 14C/1000 + 1) and km_to_nearest_balesdent_site.
  The same path with .json: the source and its checksum, the settings, and the
  profile-years dropped or flagged.

Run from the project root directory:
    python notebooks/01_preprocess_israd.py --israd ISRAD_LAYER_CSV [-o OUTPUT] [--max-stock 150]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from soil_diskin.israd import DEPTH, load_layers, pseudo_bulk
from soil_diskin.site_inputs import distance_km, source_entry
from soil_diskin.utils import file_digest

BALESDENT_PATH = 'results/processed_balesdent_2018.csv'
OUTPUT_PATH = 'results/israd/israd_profiles.csv'
MAX_STOCK_KG_M2 = 150.
# ISRaD entries dropped whole, with the reason (recorded in the output JSON).
EXCLUDED_ENTRIES = {
    'Min_2025': (
        'Implausible carbon contents (checked 2026-10-29 in ISRaD v2.9.9). No lyr_soc is reported, so stocks '
        'come from carbon % x bulk density, but lyr_c_tot is exactly twice lyr_c_org in every layer, and '
        'lyr_c_org runs e.g. 0.41% at 0-20 cm, 0.00% at 20-40 cm and 1.79% at 180-210 cm. The Michigan GLBRC '
        'site gets 0-100 cm stocks of 0.07-2.1 kg C/m² (turnover 0.1-3 yr; no other ISRaD profile is below '
        '1.3 kg C/m²). The Wisconsin site\'s stocks (7.8-9.4 kg C/m²) look plausible but come from the same '
        'entry, so it is dropped too.'),
}


def nearest_km(profiles, sites):
    """Distance in km from each profile to the nearest of `sites` (Latitude, Longitude)."""
    lat, lon = profiles[['latitude', 'longitude']].to_numpy(float).T
    km = distance_km(lat[:, None], lon[:, None], sites.Latitude.to_numpy()[None, :], sites.Longitude.to_numpy()[None, :])
    return km.min(axis=1)


def preprocess(israd_path, balesdent_path=BALESDENT_PATH, max_stock=MAX_STOCK_KG_M2):
    """The ISRaD profile table and a dict describing what was dropped or flagged."""
    layers = load_layers(israd_path)
    profiles = pseudo_bulk(layers)
    excluded = profiles.entry_name.isin(EXCLUDED_ENTRIES)
    excluded_entries = {entry: {'profile_years': int(len(group)), 'reason': EXCLUDED_ENTRIES[entry]}
                        for entry, group in profiles[excluded].groupby('entry_name')}
    complete = len(profiles)
    profiles = profiles[~excluded]
    implausible = profiles.stock_kg_m2 > max_stock
    dropped = {entry: {'profile_years': int(len(group)), 'median_stock_kg_m2': float(group.stock_kg_m2.median())}
               for entry, group in profiles[implausible].groupby('entry_name')}
    profiles = profiles[~implausible].rename(columns={'lyr_obs_date_y': 'sampling_year', 'R': 'fm_obs'})
    profiles['sampling_year'] = profiles.sampling_year.astype(int)
    profiles['14C'] = (profiles.fm_obs - 1)*1000
    sites = pd.read_csv(balesdent_path)[['Latitude', 'Longitude']].drop_duplicates()
    profiles['km_to_nearest_balesdent_site'] = nearest_km(profiles, sites)
    columns = ['entry_name', 'site_name', 'pro_name', 'sampling_year', 'latitude', 'longitude', 'n_layers',
               'n_fallback_layers', 'stock_kg_m2', '14C', 'fm_obs', 'km_to_nearest_balesdent_site']
    profiles = profiles[columns].reset_index(drop=True)
    early = profiles[profiles.sampling_year < 1950]
    summary = {'layers_with_radiocarbon_and_stock': int(len(layers)),
               'complete_profile_years': complete, 'excluded_entries': excluded_entries,
               'dropped_stock_above_max': dropped, 'profile_years': int(len(profiles)),
               'locations': int(len(profiles[['latitude', 'longitude']].drop_duplicates())),
               'entries': int(profiles.entry_name.nunique()),
               'sampling_years': [int(profiles.sampling_year.min()), int(profiles.sampling_year.max())],
               'sampled_before_1950': early[['entry_name', 'site_name', 'pro_name', 'sampling_year']].to_dict('records')}
    return profiles, summary


def main(argv=None):
    """Write the ISRaD profile table and its JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--israd', required=True, help='ISRaD flat layer table (ISRaD_data_flat_layer_*.csv)')
    parser.add_argument('--balesdent', default=BALESDENT_PATH, help='Balesdent site table, for distances to its sites')
    parser.add_argument('-o', '--output', default=OUTPUT_PATH, help='output table')
    parser.add_argument('--max-stock', type=float, default=MAX_STOCK_KG_M2,
                        help='drop profile-years whose 0-100 cm stock exceeds this (kg C/m²)')
    args = parser.parse_args(argv)
    output = Path(args.output)
    profiles, summary = preprocess(args.israd, args.balesdent, args.max_stock)
    output.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output, index=False)
    metadata = {'sources': {'israd_layers': source_entry(args.israd), 'balesdent_sites': source_entry(args.balesdent)},
                'depth_cm': DEPTH, 'max_stock_kg_m2': args.max_stock,
                'stock': 'lyr_soc, else carbon % x bulk density x thickness (no coarse-fragment correction); kg C/m²',
                'radiocarbon': 'lyr_14c/1000 + 1, else fraction modern x exp(-lambda (year - 1950)); stock-weighted',
                **summary, 'preparation_source_sha256': file_digest(__file__), 'table_sha256': file_digest(output)}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, default=str)+'\n')
    print(f'Saved {len(profiles)} ISRaD profile-years to {output}')


if __name__ == '__main__':
    main()
