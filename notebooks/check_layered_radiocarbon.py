"""Verify layered targets against 02_get_turnover_14C.py lines 36–67, without Earth Engine.

Run: python -m notebooks.check_layered_radiocarbon --output-dir results/my_radio_check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rio
import xarray as xr

from soil_diskin.layered_data import file_digest, load_profiles


def check_radiocarbon(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a new or empty output directory')
    prepared = load_profiles(allow_partial=True)
    sources = prepared.metadata['sources']
    processed = pd.read_csv(sources['processed_layers']['path'], dtype={'profile_id': str}, float_precision='round_trip')
    raw = processed.drop_duplicates('profile_id').rename(columns={'profile_id': 'Internal_profile_ID',
                                                                 'latitude': 'Latitude', 'longitude': 'Longitude'})
    raw = raw.reset_index(drop=True)
    # Independently reproduce the original rasterio path, rather than reuse the adapter.
    raster = rio.open_rasterio(sources['shi']['path'], masked=True)
    if not isinstance(raster, xr.DataArray):
        raise ValueError('expected one Shi raster')
    with raster:
        c14_data = raster.rio.write_crs('EPSG:4326').rio.write_nodata(np.nan)
        c14_data_extrapolated = c14_data.rio.interpolate_na(method='nearest')

        def extract_sites(ds):
            return xr.concat([ds.sel(y=row['Latitude'], x=row['Longitude'], method='nearest')
                              for _, row in raw.iterrows()], dim='site')

        reference = extract_sites(c14_data_extrapolated).values.reshape((-1, 10, 10)).mean(axis=2)
        native = extract_sites(c14_data).values.reshape((-1, 10, 10)).mean(axis=2)

    audit = raw.loc[raw.index.repeat(10), ['Internal_profile_ID', 'Latitude', 'Longitude']].rename(
        columns={'Internal_profile_ID': 'profile_id'}).reset_index(drop=True)
    audit['layer'] = np.tile(np.arange(10), len(raw))
    audit['delta14c_original_permil'] = reference.ravel()
    audit['delta14c_unfilled_permil'] = native.ravel()
    audit['fm_original'] = 1+reference.ravel()/1000
    audit = audit.merge(prepared.profiles[['profile_id', 'layer', 'fm_obs', 'radiocarbon_spatially_filled']],
                        on=['profile_id', 'layer'], how='left', validate='one_to_one')
    audit['eligible_for_layered_fit'] = audit.fm_obs.notna()
    audit['fm_difference'] = audit.fm_obs-audit.fm_original
    compared = audit[audit.eligible_for_layered_fit]
    assert len(compared) == len(prepared.profiles)
    np.testing.assert_array_equal(compared.fm_obs.to_numpy(), compared.fm_original.to_numpy())
    native_difference = np.abs(reference-native)
    previous = compared[compared.delta14c_unfilled_permil.notna()]
    summary = {
        'status': 'complete', 'reference': '02_get_turnover_14C.py lines 36–67; per-layer values, not weighted column mean',
        'raw_profiles': len(raw), 'raw_profile_layers': len(audit),
        'original_finite_layer_values': int(np.isfinite(reference).sum()),
        'native_missing_layer_values': int(np.isnan(native).sum()),
        'maximum_existing_delta_difference_permil': float(np.nanmax(native_difference)),
        'eligible_profiles': prepared.profiles.profile_id.nunique(), 'eligible_layers': len(compared),
        'eligible_locations': len(prepared.profiles[['latitude', 'longitude']].drop_duplicates()),
        'previous_native_profiles': previous.profile_id.nunique(), 'previous_native_layers': len(previous),
        'filled_eligible_layers': int(prepared.profiles.radiocarbon_spatially_filled.sum()),
        'maximum_abs_fm_difference': float(compared.fm_difference.abs().max()),
        'model_fits_rerun': False, 'data': prepared.metadata,
        'source_sha256': {name: file_digest(name) for name in [
            'notebooks/check_layered_radiocarbon.py', 'notebooks/02_get_turnover_14C.py', 'soil_diskin/layered_data.py']}}
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output/'radiocarbon_comparison.csv', index=False)
    prepared.profiles.to_csv(output/'prepared_layers.csv', index=False)
    prepared.excluded.to_csv(output/'exclusions.csv', index=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({key: value for key, value in summary.items() if key not in ['data', 'source_sha256']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results/layered_radiocarbon_parity'))
    check_radiocarbon(parser.parse_args().output_dir)
