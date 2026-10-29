"""Step 02: bulk soil radiocarbon and turnover times for the Balesdent sites.

For each row of results/processed_balesdent_2018.csv (a site and labeling
duration), this script combines
  * the 0-100 cm soil carbon stock, from script 01, with its q05/q95 estimates;
  * the carbon input, taken to be the site's NPP, from the table written by
    script 02a (MODIS MOD17A3HGF via Earth Engine); and
  * the soil radiocarbon, from the global Delta14C map of Shi et al. (2020)
    (data/shi_2020/global_delta_14C.nc), which has one-cm levels from 0 to 100 cm.
    Gaps in the map are filled with the nearest value at the same depth, so a
    site in an empty cell (e.g. on a small island) takes its profile from another
    cell, which is recorded.

Delta14C is averaged over 10-cm bins and weighted by the row's share of carbon in
each bin (weight_0 ... weight_90 from script 01). The turnover time is
stock / NPP. Radiocarbon is reported as Delta14C in per mil (column `14C`) and as
R = Delta14C/1000 + 1 (column `fm_obs`), the convention of
data/14C_atm_annot.csv and of the calibration scripts.

The depth-resolved counterpart, one row per reported layer, is
02c_get_layer_turnover_14C.py; the code they share is in soil_diskin.site_inputs.

Outputs:
  --output (default results/all_sites_14C_turnover.csv): Latitude, Longitude,
  14C, NPP (g C/m²/yr), the stocks (kg C/m²), fm_obs, turnover/_q05/_q95 (yr), and
  the grid cell the radiocarbon comes from (radiocarbon_spatially_filled,
  radiocarbon_source_latitude/_longitude, radiocarbon_source_km).
  The same path with .json: the sources, their checksums and the NPP query.

Run from the project root directory:
    python notebooks/02_get_turnover_14C.py [-i INPUT] [-o OUTPUT] [--shi SHI] [--npp NPP]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from soil_diskin.site_inputs import (
    npp_at, npp_provenance, radiocarbon_at, read_npp, shi_provenance, source_entry)
from soil_diskin.utils import file_digest

INPUT_PATH = 'results/processed_balesdent_2018.csv'
OUTPUT_PATH = 'results/all_sites_14C_turnover.csv'
SHI_PATH = 'data/shi_2020/global_delta_14C.nc'
NPP_PATH = 'results/02_npp/site_npp.csv'
NAMES = {'Latitude': 'latitude', 'Longitude': 'longitude'}
SUFFIXES = ['', '_q05', '_q95']


def bulk_turnover(data, shi, npp):
    """Add radiocarbon, NPP and turnover times to the processed Balesdent sites.

    Args:
        data: processed sites from script 01, with Latitude, Longitude,
            weight_0 ... weight_90 and Ctotal_0-100estim (and _q05/_q95).
        shi: the Shi et al. (2020) Delta14C dataset (see site_inputs.sample_radiocarbon).
        npp: NPP table from site_inputs.read_npp.

    Returns:
        One row per row of `data` with Latitude, Longitude, 14C, NPP, the stocks,
        fm_obs, turnover/_q05/_q95, and where the radiocarbon comes from:
        radiocarbon_spatially_filled (from a gap-filled cell, see
        site_inputs.sample_radiocarbon), radiocarbon_source_latitude/_longitude
        (the cell) and radiocarbon_source_km (its distance from the site). Rows
        without NPP > 0 get NaN turnover; rows without any radiocarbon data get
        NaN 14C.
    """
    result = data.rename(columns=NAMES)
    delta, filled, source = radiocarbon_at(result, shi)
    result['NPP'] = npp_at(result, npp)
    npp_kg_m2_yr = result.NPP/1000
    # Weight the 10-cm bin means of the row's location by the row's own carbon
    # weights. Bins below a profile's deepest measured layer have no weight and are
    # skipped (script 01 normalizes the weights over the rest).
    weights = result.filter(regex=r'^weight_\d+').to_numpy()
    weighted = delta.reshape(-1, 10, 10).mean(axis=2)*weights
    result['14C'] = np.where(np.isfinite(weighted).any(axis=1), np.nansum(weighted, axis=1), np.nan)
    result['fm_obs'] = result['14C']/1000+1
    # Gap-filled if any level of a bin with a weight came from another cell.
    result['radiocarbon_spatially_filled'] = (filled.reshape(-1, 10, 10).any(axis=2) & np.isfinite(weights)).any(axis=1)
    result = result.join(source)
    stocks = ['Ctotal_0-100estim'+suffix for suffix in SUFFIXES]
    for stock, suffix in zip(stocks, SUFFIXES):
        result['turnover'+suffix] = result[stock]/npp_kg_m2_yr.where(npp_kg_m2_yr.gt(0))
    columns = (list(NAMES.values())+['14C', 'NPP', *stocks, 'fm_obs']+['turnover'+suffix for suffix in SUFFIXES]
               + ['radiocarbon_spatially_filled', *source.columns])
    return result[columns].rename(columns={v: k for k, v in NAMES.items()})


def main(argv=None):
    """Read the processed sites, add radiocarbon, NPP and turnover, and write the table and its JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', default=INPUT_PATH, help='processed sites from script 01')
    parser.add_argument('-o', '--output', default=OUTPUT_PATH, help='output table')
    parser.add_argument('--shi', default=SHI_PATH, help='Shi et al. (2020) Delta14C map')
    parser.add_argument('--npp', default=NPP_PATH, help='NPP table from script 02a')
    args = parser.parse_args(argv)
    source, output = Path(args.input), Path(args.output)
    data = pd.read_csv(source, float_precision='round_trip')
    with xr.open_dataset(args.shi) as shi:
        result = bulk_turnover(data, shi, read_npp(args.npp))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    metadata = {'sources': {'sites': source_entry(source), 'shi': source_entry(args.shi)}, **shi_provenance(),
                'radiocarbon_depth_aggregation': '10-cm bin means weighted by the site carbon weights of script 01; bins without a weight skipped',
                'npp': npp_provenance(args.npp), 'npp_conversion': 'g C/m²/yr divided by 1000 to kg C/m²/yr',
                'turnover': 'Ctotal_0-100estim (and _q05/_q95) / NPP; all NPP enters 0-100 cm',
                'preparation_source_sha256': file_digest(__file__), 'table_sha256': file_digest(output)}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved turnover to {output}')


if __name__ == '__main__':
    main()
