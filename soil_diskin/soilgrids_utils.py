"""
Utilities for accessing SoilGrids data via Google Earth Engine.

SoilGrids provides global soil property predictions at 250m resolution.
This module provides functions to query soil organic carbon (SOC) data
for specific locations.

TODO: add unit tests with a mocked Earth Engine interface.
"""

import ee
import yaml

import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
def initialize_earth_engine():
    """Initialize Earth Engine. Call this before using any EE functions."""
    try:
        ee.Authenticate()  # Authenticate Earth Engine
        ee.Initialize(project=config['earth_engine']['project'])
    except Exception as e:
        print("Earth Engine initialization failed. Attempting authentication...")
        ee.Authenticate()
        ee.Initialize()


def get_soc_at_point(lat, lon, scale=250):
    """
    Get soil organic carbon (g/kg) at a point from SoilGrids.
    
    Args:
        lat (float): Latitude in decimal degrees
        lon (float): Longitude in decimal degrees
        scale (int): Resolution in meters (default 250m for SoilGrids)
    
    Returns:
        dict: SOC values for each depth layer in g/kg
            Keys are depth ranges: '0-5cm', '5-15cm', '15-30cm', 
                                   '30-60cm', '60-100cm', '100-200cm'
    """
    # Load SoilGrids SOC layer
    soc_image = ee.Image("projects/soilgrids-isric/soc_mean")
    
    # Create point geometry
    point = ee.Geometry.Point([lon, lat])
    
    # Sample the image at the point
    sample = soc_image.sample(
        region=point,
        scale=scale,
        geometries=True
    ).first()
    
    # Extract values for each band
    if sample is None:
        return None
    
    band_names = ['soc_0-5cm_mean', 'soc_5-15cm_mean', 'soc_15-30cm_mean',
                  'soc_30-60cm_mean', 'soc_60-100cm_mean', 'soc_100-200cm_mean']
    depth_names = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
    
    soc_values = {}
    for band, depth in zip(band_names, depth_names):
        try:
            # SoilGrids SOC is in g/kg, need to divide by 10 to get actual values
            value = sample.get(band).getInfo()
            soc_values[depth] = value / 10.0 if value is not None else None
        except:
            soc_values[depth] = None
    
    return soc_values


def calculate_total_soc_0_100(soc_dict):
    """
    Calculate total SOC for 0-100cm depth from SoilGrids data.
    
    This approximates the Ctotal_0-100estim column in the Balesdent dataset.
    
    Args:
        soc_dict (dict): Dictionary of SOC values by depth from get_soc_at_point
    
    Returns:
        float: Estimated total SOC in kg/m² for 0-100cm depth, or None if insufficient data
    """
    if soc_dict is None:
        return None
    
    # Depth intervals in cm
    depths = {
        '0-5cm': (0, 5),
        '5-15cm': (5, 15),
        '15-30cm': (15, 30),
        '30-60cm': (30, 60),
        '60-100cm': (60, 100)
    }
    
    total_soc = 0.0
    
    for depth_name, (top, bottom) in depths.items():
        soc_gkg = soc_dict.get(depth_name)
        if soc_gkg is None:
            return None  # Can't calculate if any layer is missing
        
        thickness_cm = bottom - top
        
        # Convert g/kg to kg/m²
        # Assuming typical bulk density of 1.3 g/cm³ = 1300 kg/m³
        # This could be improved by also fetching bulk density from SoilGrids
        bulk_density = 1.3  # g/cm³ = kg/dm³
        
        # SOC (kg/m²) = SOC (g/g) × bulk_density (g/cm³) × thickness (cm) * 1e4 (to convert m² to cm²) / 1e3 (to convert g to kg)
        layer_soc = (soc_gkg / 1000.0) * bulk_density * thickness_cm * 1e4 / 1e3
        total_soc += layer_soc
    
    return total_soc


def get_soc_with_bulk_density(lat, lon, scale=250):
    """
    Get SOC and bulk density at a point, then calculate total SOC more accurately.
    
    Args:
        lat (float): Latitude in decimal degrees
        lon (float): Longitude in decimal degrees
        scale (int): Resolution in meters (default 250m)
    
    Returns:
        float: Estimated total SOC in kg/m² for 0-100cm depth
    """
    # Load both SOC and bulk density images
    soc_image = ee.Image("projects/soilgrids-isric/soc_mean")
    bdod_image = ee.Image("projects/soilgrids-isric/bdod_mean")
    
    # Create point geometry
    point = ee.Geometry.Point([lon, lat])
    
    # Sample both images
    soc_sample = soc_image.sample(region=point, scale=scale).first()
    bdod_sample = bdod_image.sample(region=point, scale=scale).first()
    
    if soc_sample is None or bdod_sample is None:
        return None
    
    # Define bands and depths. Note that mapped units are dg/kg for SOC and cg/cm3 for
    # bulk density. SOC should be divided by 10 to get g/kg, bulk density by 100 to get kg/dm3.
    soc_bands = ['soc_0-5cm_mean', 'soc_5-15cm_mean', 'soc_15-30cm_mean',
                 'soc_30-60cm_mean', 'soc_60-100cm_mean']
    bdod_bands = ['bdod_0-5cm_mean', 'bdod_5-15cm_mean', 'bdod_15-30cm_mean',
                  'bdod_30-60cm_mean', 'bdod_60-100cm_mean']
    thicknesses = [5, 10, 15, 30, 40]  # cm
    
    total_soc = 0.0
    soc_conversion = 10.0  # from dg/kg to g/kg
    bdod_conversion = 100.0  # from cg/cm3 to g/cm3
    
    try:
        for soc_band, bdod_band, thickness in zip(soc_bands, bdod_bands, thicknesses):
            # Get SOC in g/kg (divide by 10)
            soc_value = soc_sample.get(soc_band).getInfo()
            if soc_value is None:
                return None
            # Convert from dg/kg to g/kg
            soc_gkg = soc_value / soc_conversion
            
            # Get bulk density in kg/dm³ (divide by 100)
            # Note: kg/dm³ = g/cm³
            bdod_value = bdod_sample.get(bdod_band).getInfo()
            if bdod_value is None:
                return None
            bulk_density_kgdm = bdod_value / bdod_conversion
            
            # Calculate SOC for this layer in kg/m²
            # SOC (kg/m²) = SOC (g/g) × bulk_density (g/cm³) × thickness (cm) * 1e4 (to convert m² to cm²) / 1e3 (to convert g to kg)
            layer_soc = (soc_gkg / 1000.0) * bulk_density_kgdm * thickness * 1e4 / 1e3
            total_soc += layer_soc
            
    except Exception as e:
        print(f"Error calculating SOC at ({lat}, {lon}): {e}")
        return None
    
    return total_soc


def backfill_missing_soc(df, lat_col='Latitude', lon_col='Longitude', 
        soc_col='Ctotal_0-100estim', source_col='C_data_source', use_bulk_density=True):
    """
    Backfill missing SOC values in a DataFrame using SoilGrids data.
    
    Args:
        df (pd.DataFrame): DataFrame with location and SOC data
        lat_col (str): Name of latitude column
        lon_col (str): Name of longitude column
        soc_col (str): Name of SOC column to backfill
        source_col (str): Name of column indicating data source
        use_bulk_density (bool): If True, fetch bulk density for more accurate estimates
    
    Returns:
        pd.DataFrame: DataFrame with backfilled SOC values
        dict: Statistics about the backfilling operation
    """
    # Initialize Earth Engine
    initialize_earth_engine()
    
    df = df.copy()
    
    # Find rows with missing SOC
    missing_mask = df[soc_col].isna()
    n_missing = missing_mask.sum()
    
    print(f"Found {n_missing} rows with missing {soc_col}")
    
    if n_missing == 0:
        return df, {'n_missing': 0, 'n_filled': 0, 'n_failed': 0}
    
    # Get unique locations with missing SOC
    missing_locs = df[missing_mask][[lat_col, lon_col]].drop_duplicates()
    
    print(f"Querying SoilGrids for {len(missing_locs)} unique locations...")
    
    # Query SoilGrids for each unique location
    soc_lookup = {}
    n_filled = 0
    n_failed = 0
    
    for idx, row in missing_locs.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        key = (lat, lon)
        
        try:
            if use_bulk_density:
                soc_value = get_soc_with_bulk_density(lat, lon)
            else:
                soc_dict = get_soc_at_point(lat, lon)
                soc_value = calculate_total_soc_0_100(soc_dict)
            
            if soc_value is not None:
                soc_lookup[key] = soc_value
                n_filled += 1
                print(f"  ({lat:.4f}, {lon:.4f}): {soc_value:.2f} kg/m²")
            else:
                n_failed += 1
                print(f"  ({lat:.4f}, {lon:.4f}): Failed to retrieve data")
        except Exception as e:
            n_failed += 1
            print(f"  ({lat:.4f}, {lon:.4f}): Error - {e}")
    
    # Backfill the DataFrame
    for idx, row in df[missing_mask].iterrows():
        key = (row[lat_col], row[lon_col])
        if key in soc_lookup:
            df.loc[idx, soc_col] = soc_lookup[key]
            df.loc[idx, source_col] = 'SoilGrids backfill'
    
    stats = {
        'n_missing': n_missing,
        'n_filled': n_filled,
        'n_failed': n_failed,
        'fill_rate': n_filled / len(missing_locs) if len(missing_locs) > 0 else 0
    }
    
    print(f"\nBackfilling complete:")
    print(f"  {n_filled} locations successfully filled")
    print(f"  {n_failed} locations failed")
    print(f"  Fill rate: {stats['fill_rate']:.1%}")
    
    # Remove points with missing data
    df = df[~df[soc_col].isna()]

    return df, stats
