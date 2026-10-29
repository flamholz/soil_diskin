"""Bulk 0-100 cm soil profiles from the International Soil Radiocarbon Database (ISRaD).

Reads the flat layer table of ISRaD (data/ISRaD/ISRaD_data_flat_layer_*.csv) and
combines each profile's layers into one pseudo-bulk soil column, as needed to
compare ISRaD with the bulk Balesdent analysis. Used by
notebooks/01_preprocess_israd.py and notebooks/experimental/calibrate_lognormal_israd.py.

Units follow the Balesdent tables: stocks in kg C/m², depths in cm, and
radiocarbon as R = Delta14C/1000 + 1, the convention of data/14C_atm_annot.csv.
ISRaD's own units (from its template documentation): lyr_soc in g C/cm², bulk
density in g/cm³, carbon in weight %, lyr_14c in per mil, and fraction modern
relative to 95% of the 1950 NBS oxalic acid standard.
"""
import numpy as np
import pandas as pd

from soil_diskin.constants import LAMBDA_14C

DEPTH = 100.  # cm
PROFILE = ['entry_name', 'site_name', 'pro_name', 'lyr_obs_date_y']  # one pseudo-bulk column each
G_CM2_TO_KG_M2 = 10.  # 1 g/cm² = 10 kg/m²


def load_layers(path):
    """ISRaD layers with radiocarbon (R) and a carbon stock (stock_kg_m2), from measured values only.

    R is lyr_14c/1000 + 1, or, where only fraction modern is reported, Fm
    exp(-lambda (year - 1950)): ISRaD's Fm is decay corrected to 1950, while R
    (like Delta14C) refers to the sampling year. The stock is the reported
    lyr_soc, or else carbon % x bulk density x thickness, which is how ISRaD
    computes lyr_soc (no coarse-fragment correction). Carbon % is lyr_c_org, else
    lyr_c_tot; bulk density is lyr_bd_samp (coarse fragments removed), else
    lyr_bd_tot. `fallback` marks layers that used lyr_c_tot or lyr_bd_tot.
    Layers without R, a stock, bounds, a sampling year or coordinates, or with
    zero thickness or a negative stock, are dropped.
    """
    numeric = ['site_lat', 'site_long', 'lyr_top', 'lyr_bot', 'lyr_obs_date_y', 'lyr_c_org',
               'lyr_c_tot', 'lyr_soc', 'lyr_bd_samp', 'lyr_bd_tot', 'lyr_14c', 'lyr_fraction_modern']
    layers = pd.read_csv(path, usecols=PROFILE[:3]+['lyr_name']+numeric,
                         encoding_errors='replace', low_memory=False)
    layers[numeric] = layers[numeric].apply(pd.to_numeric, errors='coerce')
    year = layers.lyr_obs_date_y
    layers['R'] = (layers.lyr_14c/1000 + 1).fillna(
        layers.lyr_fraction_modern*np.exp(-(year - 1950)*LAMBDA_14C))
    thickness = layers.lyr_bot - layers.lyr_top
    carbon = layers.lyr_c_org.fillna(layers.lyr_c_tot)
    density = layers.lyr_bd_samp.fillna(layers.lyr_bd_tot)
    stock_g_cm2 = layers.lyr_soc.fillna(carbon/100*density*thickness)
    layers['fallback'] = layers.lyr_soc.isna() & (
        (layers.lyr_c_org.isna() & layers.lyr_c_tot.notna())
        | (layers.lyr_bd_samp.isna() & layers.lyr_bd_tot.notna()))
    layers['stock_kg_m2'] = stock_g_cm2*G_CM2_TO_KG_M2
    valid = layers[['R', 'stock_kg_m2', 'lyr_top', 'lyr_bot', 'lyr_obs_date_y', 'site_lat', 'site_long']].notna().all(axis=1)
    return layers[valid & (thickness > 0) & (layers.stock_kg_m2 >= 0)]


def pseudo_bulk(layers, depth=DEPTH):
    """Stock and stock-weighted R over [0, depth] cm for each completely covered profile-year.

    A profile-year (PROFILE) is kept only if its layers, clipped to [0, depth],
    abut from 0 to `depth` with no gaps or overlaps. Organic horizons (negative
    depths) are excluded, and layers crossing `depth` are prorated by thickness.

    Returns:
        One row per profile-year: the PROFILE columns, latitude, longitude,
        n_layers, n_fallback_layers, stock_kg_m2 (total over [0, depth]) and R
        (stock-weighted mean).
    """
    rows = []
    for key, profile in layers.groupby(PROFILE, dropna=False):
        top, bottom = profile.lyr_top.clip(lower=0), profile.lyr_bot.clip(upper=depth)
        inside = bottom > top
        if not inside.any():
            continue
        profile, top, bottom = profile[inside], top[inside], bottom[inside]
        order = np.argsort(top.to_numpy())
        top, bottom = top.to_numpy()[order], bottom.to_numpy()[order]
        # Complete coverage without overlap: layers abut from 0 to depth.
        if top[0] > 0 or bottom[-1] < depth or np.any(top[1:] != bottom[:-1]):
            continue
        share = (bottom - top)/(profile.lyr_bot - profile.lyr_top).to_numpy()[order]
        stock = profile.stock_kg_m2.to_numpy()[order]*share
        if stock.sum() <= 0:
            continue
        rows.append(dict(zip(PROFILE, key), latitude=profile.site_lat.iloc[0],
                         longitude=profile.site_long.iloc[0], n_layers=len(profile),
                         n_fallback_layers=int(profile.fallback.sum()),
                         stock_kg_m2=stock.sum(), R=(stock*profile.R.to_numpy()[order]).sum()/stock.sum()))
    columns = PROFILE + ['latitude', 'longitude', 'n_layers', 'n_fallback_layers', 'stock_kg_m2', 'R']
    return pd.DataFrame(rows, columns=columns)
