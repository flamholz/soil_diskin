"""Step 02 (ISRaD): NPP and turnover times for the pseudo-bulk ISRaD profiles.

The ISRaD counterpart of 02_get_turnover_14C.py. For each profile-year of
results/israd/israd_profiles.csv (01_preprocess_israd.py), this script adds the
carbon input, taken to be the location's NPP from the table written by script
02a for the ISRaD locations (MODIS MOD17A3HGF via Earth Engine), and the turnover
time stock / NPP, with all of NPP entering 0-100 cm as in the bulk Balesdent
analysis. The radiocarbon (14C, fm_obs) comes from ISRaD itself, at each
profile's sampling year, rather than from a global map.

Only profile-years with NPP > 0 are kept. No turnover_q05/_q95 are given: the
only NPP spread available, the standard deviation across annual images
(npp_sd_kg_m2_yr), is interannual variability rather than an error of the mean.

Outputs:
  --output (default results/israd/israd_turnover.csv): the profile table plus
  npp_kg_m2_yr, npp_sd_kg_m2_yr (kg C/m²/yr) and turnover (yr).
  The same path with .json: the sources, their checksums, the NPP query, and the
  profile-years dropped for lack of NPP.

Run from the project root directory:
    python notebooks/02_get_israd_turnover.py [-i INPUT] [-o OUTPUT] [--npp NPP]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from soil_diskin.site_inputs import npp_at, npp_provenance, read_npp, source_entry
from soil_diskin.utils import file_digest

INPUT_PATH = 'results/israd/israd_profiles.csv'
OUTPUT_PATH = 'results/israd/israd_turnover.csv'
NPP_PATH = 'results/02_npp/israd_npp.csv'


def israd_turnover(profiles, npp):
    """Add NPP (kg C/m²/yr) and turnover (yr) to the ISRaD profiles; keep those with NPP > 0.

    Returns:
        The table, and the profile-years dropped for lack of a positive NPP.
    """
    result = profiles.copy()
    result['npp_kg_m2_yr'] = npp_at(result, npp)/1000
    result['npp_sd_kg_m2_yr'] = npp_at(result, npp, column='NPP_sd')/1000
    result['turnover'] = result.stock_kg_m2/result.npp_kg_m2_yr
    usable = result.npp_kg_m2_yr.gt(0)
    return result[usable].reset_index(drop=True), result[~usable]


def main(argv=None):
    """Write the ISRaD turnover table and its JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', default=INPUT_PATH, help='ISRaD profiles from 01_preprocess_israd.py')
    parser.add_argument('-o', '--output', default=OUTPUT_PATH, help='output table')
    parser.add_argument('--npp', default=NPP_PATH, help='NPP table of the ISRaD locations from script 02a (with --sd)')
    args = parser.parse_args(argv)
    source, output = Path(args.input), Path(args.output)
    result, dropped = israd_turnover(pd.read_csv(source), read_npp(args.npp))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    metadata = {'sources': {'israd_profiles': source_entry(source)},
                'npp': npp_provenance(args.npp), 'npp_conversion': 'g C/m²/yr divided by 1000 to kg C/m²/yr',
                'turnover': 'stock_kg_m2 / npp_kg_m2_yr; all NPP enters 0-100 cm',
                'dropped_without_positive_npp': dropped[['entry_name', 'site_name', 'pro_name', 'sampling_year']].to_dict('records'),
                'profile_years': int(len(result)),
                'preparation_source_sha256': file_digest(__file__), 'table_sha256': file_digest(output)}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved turnover for {len(result)} ISRaD profile-years to {output}')


if __name__ == '__main__':
    main()
