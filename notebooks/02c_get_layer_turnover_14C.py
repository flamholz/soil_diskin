"""Step 02c: soil radiocarbon, carbon inputs and turnover times for each Balesdent layer.

The depth-resolved counterpart of 02_get_turnover_14C.py. For each reported layer
of results/processed_balesdent_2018_sampled_soilgrids.csv (script 01 with
--depth-resolved --backfill), this script combines
  * the layer's carbon stock, from script 01, with its q05/q95 estimates if present;
  * the layer's carbon input: the site's NPP, from the table written by script
    02a, distributed over 0-100 cm by an InputAllocation (by default half in
    0-10 cm and half following the Jackson et al. global root profile); and
  * the soil radiocarbon, from the global Delta14C map of Shi et al. (2020),
    averaged over the layer's depth range (one-cm levels [z, z+1) cm, no
    extrapolation below 100 cm). Gaps in the map are filled with the nearest
    value at the same depth, and filled layers are flagged.

The turnover time is stock / input. Radiocarbon is reported as Delta14C in per mil
(column `14C`) and as R = Delta14C/1000 + 1 (column `fm_obs`).

Outputs (both refused if they already exist):
  --output (default results/all_sites_14C_turnover_depth_resolved.csv): the layer
  table plus npp_kg_m2_yr, 14C, fm_obs, radiocarbon_spatially_filled, the
  allocation settings, input_kg_m2_yr and turnover/_q05/_q95 (yr).
  The same path with .json: the sources, their checksums, the NPP query and the
  settings, plus script 01's JSON if there is one beside the input.

Run from the project root directory:
    python notebooks/02c_get_layer_turnover_14C.py [-i INPUT] [-o OUTPUT] [--shi SHI] [--npp NPP]
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from soil_diskin.site_inputs import (
    DEFAULT_ALLOCATION, InputAllocation, npp_at, npp_provenance, radiocarbon_at, read_npp,
    shi_provenance, source_entry, verify_shi)
from soil_diskin.utils import file_digest

INPUT_PATH = 'results/processed_balesdent_2018_sampled_soilgrids.csv'
OUTPUT_PATH = 'results/all_sites_14C_turnover_depth_resolved.csv'
SHI_PATH = 'data/shi_2020/global_delta_14C.nc'
NPP_PATH = 'results/02_npp/site_npp.csv'


def layer_turnover(data, shi, npp, allocation=DEFAULT_ALLOCATION):
    """Add radiocarbon, carbon inputs and turnover times to the processed layers.

    Args:
        data: processed layers from script 01, with latitude, longitude,
            z_top_cm, z_bottom_cm and stock_kg_m2 (and _q05/_q95, if present).
        shi: the Shi et al. (2020) Delta14C dataset (see site_inputs.sample_radiocarbon).
        npp: NPP table from site_inputs.read_npp.
        allocation: InputAllocation of the site's NPP over depth.

    Returns:
        `data` plus npp_kg_m2_yr, 14C, fm_obs, radiocarbon_spatially_filled, the
        allocation settings, npp_distribution, input_kg_m2_yr and turnover/_q05/_q95.
        Layers with bounds outside 0-100 cm, or a missing depth inside them, get
        NaN 14C; layers without a positive input get NaN turnover.
    """
    result = data.copy()
    delta, filled = radiocarbon_at(result, shi)
    result['npp_kg_m2_yr'] = npp_at(result, npp)/1000
    top, bottom = result[['z_top_cm', 'z_bottom_cm']].to_numpy(float).T
    depth = np.arange(100)  # Level z represents [z, z+1) cm, as in the bulk means.
    overlap = np.maximum(0, np.minimum(bottom[:, None], depth+1)-np.maximum(top[:, None], depth))
    supported = (top >= 0) & (bottom <= 100) & (bottom > top)
    thickness = np.where(supported, bottom-top, np.nan)
    # Missing contributing depths invalidate the mean; missing depths elsewhere do not.
    result['14C'] = (np.where(overlap > 0, delta, 0)*overlap).sum(axis=1)/thickness
    result['fm_obs'] = result['14C']/1000+1
    result['radiocarbon_spatially_filled'] = (filled & (overlap > 0)).any(axis=1) & supported
    result = result.assign(**asdict(allocation), npp_distribution='InputAllocation')
    result['input_kg_m2_yr'] = allocation.layer_input(result.npp_kg_m2_yr, result.z_top_cm, result.z_bottom_cm)
    for suffix in ['', '_q05', '_q95']:
        if 'stock_kg_m2'+suffix in result:
            result['turnover'+suffix] = result['stock_kg_m2'+suffix]/result.input_kg_m2_yr.where(result.input_kg_m2_yr.gt(0))
    return result


def main(argv=None):
    """Read the processed layers, add radiocarbon, inputs and turnover, and write the table and its JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', default=INPUT_PATH, help='processed layers from script 01')
    parser.add_argument('-o', '--output', default=OUTPUT_PATH, help='output table (must not exist)')
    parser.add_argument('--shi', default=SHI_PATH, help='Shi et al. (2020) Delta14C map')
    parser.add_argument('--npp', default=NPP_PATH, help='NPP table from script 02a')
    parser.add_argument('--input-depth', type=float, default=DEFAULT_ALLOCATION.input_depth_cm,
                        help='e-folding depth of the input profile (cm)')
    parser.add_argument('--surface-fraction', type=float, default=DEFAULT_ALLOCATION.surface_fraction,
                        help='share of the input entering in 0-10 cm')
    parser.add_argument('--soil-npp-fraction', type=float, default=1., help='share of NPP entering the soil')
    args = parser.parse_args(argv)
    source, output = Path(args.input), Path(args.output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'use a new output path: {output}')
    md5 = verify_shi(args.shi)
    allocation = InputAllocation(args.input_depth, args.surface_fraction, args.soil_npp_fraction)
    data = pd.read_csv(source, dtype={'profile_id': str}, float_precision='round_trip')
    with xr.open_dataset(args.shi) as shi:
        result = layer_turnover(data, shi, read_npp(args.npp), allocation)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    metadata = {'sources': {'processed_layers': source_entry(source), 'shi': source_entry(args.shi)},
                **shi_provenance(md5),
                'radiocarbon_depth_aggregation': 'thickness-weighted mean over reported bounds; one-cm bins [level, level+1); no depth extrapolation',
                'npp': npp_provenance(args.npp), 'npp_conversion': 'g C/m²/yr divided by 1000 to kg C/m²/yr',
                'npp_imputed': False, 'input_allocation': asdict(allocation),
                'fnew_observation': 'Layers sheet ratio_newCtoC; stocks from Cstock in kg C/m²',
                'preparation_source_sha256': file_digest(__file__), 'table_sha256': file_digest(output)}
    if source.with_suffix('.json').exists():
        metadata['preprocessing'] = json.loads(source.with_suffix('.json').read_text())
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved turnover to {output}')


if __name__ == '__main__':
    main()
